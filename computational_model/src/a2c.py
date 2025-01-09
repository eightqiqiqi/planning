import numpy as np
import torch
import torch.nn.functional as F
import scipy
from torch.distributions import Categorical
from .priors import prior_loss

GRUind = 1

# Sample actions from the policy
def sample_actions(mod, policy_logits):
    # Don't differentiate through this sampling process
    batch = policy_logits.shape[1]  # batch size
    a = np.zeros((1, batch), dtype=np.int32)  # initialize action array

    πt = np.exp(policy_logits.astype(np.float64))  # probability of actions (up/down/right/left/stay)
    πt /= np.sum(πt, axis=0, keepdims=True)  # normalize over actions

    if np.any(np.isnan(πt)) or np.any(np.isinf(πt)):  # set to argmax(log pi) if exponentiating gave infs
        for b in range(batch):
            if np.any(np.isnan(πt[:, b])) or np.any(np.isinf(πt[:, b])):
                πt[:, b] = np.zeros_like(πt[:, b])
                πt[np.argmax(policy_logits[:, b]), b] = 1

    if mod.model_properties.greedy_actions:  # optionally select greedy action
        a[:] = np.argmax(πt, axis=0).astype(np.int32)
    else:  # sample action from policy
        for b in range(batch):
            a[0, b] = np.random.choice(len(πt[:, b]), p=πt[:, b])

    return a

# Zero pad data for finished episodes
def zeropad_data(rew, agent_input, a, active):
    with torch.no_grad():
        finished = torch.nonzero(~active).squeeze(1)
        if len(finished) > 0.5:  # if there are finished episodes
            rew[:, finished] = 0.0
            agent_input[:, finished] = 0.0
            a[:, finished] = 0.0
        return rew, agent_input, a


# Calculate prediction loss for the transition component
def calc_prediction_loss(agent_output, Naction, Nstates, s_index, active):
    new_Lpred = 0.0  # initialize prediction loss
    spred = agent_output[(Naction + 2):(Naction + 1 + Nstates), :]  # predicted next states (Nstates x batch)
    spred = spred - scipy.special.logsumexp(spred, axis=0, keepdims=True)  # softmax over states
    for b in range(active.size(0)):  # only active episodes
        if active[b]:
            new_Lpred -= spred[s_index[b], b]  # -log p(s_true)

    return new_Lpred


# Calculate prediction loss for the reward component
def calc_rew_prediction_loss(agent_output, Naction, Nstates, r_index, active):
    new_Lpred = 0.0  # initialize prediction loss
    i1 = Naction + Nstates + 3
    i2 = Naction + Nstates + 2 + Nstates
    rpred = agent_output[i1:i2, :]  # output distribution
    rpred -= torch.logsumexp(rpred, axis=0, keepdim=True)  # softmax over states
    for b in np.where(active)[0]:  # only active episodes
        new_Lpred -= rpred[r_index[b], b]  # -log p(r_true)

    return new_Lpred


# Convert action indices to one-hot representation
def construct_ahot(a, Naction):
    # no gradients through this
    batch = a.shape[1]
    ahot = np.zeros((Naction, batch), dtype=np.float32)
    for b in range(1, batch + 1):
        ahot[int(a[0, b - 1]), b - 1] = 1.0  # set each element to 1
    return ahot.astype(np.float32)

def forward_modular(mod, ed, x, h_rnn):
    Naction, Nstates, batch = ed.Naction, ed.Nstates, x.shape[1]  # Useful variables

    # Forward pass through recurrent part of RNN
    if isinstance(mod.network, (list, torch.nn.ModuleList)):
        gru_layer = mod.network[GRUind]
    else:
        gru_layer = mod.network

    h_rnn, ytemp = gru_layer(x, h_rnn)  # Forward pass for GRU

    # Apply policy (and value) network
    logπ_V = mod.policy(ytemp)
    logπ = logπ_V[:Naction, :]  # Policy is the first few rows
    V = logπ_V[Naction:(Naction + 1), :]  # Value function is the next row

    # Optionally impose that only physical actions can be chosen
    no_planning = mod.model_properties.no_planning
    if isinstance(no_planning, bool):
        if no_planning:
            logπ = logπ - torch.tensor([0.0, 0.0, 0.0, 0.0, float('-inf')], device=logπ.device)
    else:
        logπ = logπ - torch.logsumexp(logπ, dim=0, keepdim=True)  # softmax
        logπ = torch.cat([logπ[:4, :], logπ[4:5, :] + torch.tensor(no_planning, dtype=torch.float32, device=logπ.device)], dim=0)

    # Normalize logπ
    logπ = logπ - torch.logsumexp(logπ, dim=0, keepdim=True)  # softmax normalization
    a = sample_actions(mod, logπ)  # Sample actions
    ahot = construct_ahot(a, Naction)  # One-hot representation of actions

    # Input to prediction module (concatenation of hidden state and action)
    prediction_input = torch.cat([ytemp, ahot], dim=0)
    prediction_output = mod.prediction(prediction_input)  # Output of prediction module

    # Combine outputs and return
    output = torch.cat([logπ, V, prediction_output], dim=0)
    return h_rnn, output, a

# Calculate TD errors (deltas)
@torch.no_grad()
def calc_deltas(rews, Vs):
    batch, N = rews.shape
    δs = np.zeros((batch, N), dtype=np.float32)  # initialize TD errors
    R = np.zeros(batch)  # cumulative reward
    for t in range(N):  # for each iteration (moving backward!)
        R = rews[:, N - t - 1] + R  # cumulative reward
        δs[:, N - t - 1] = R - Vs[:, N - t - 1]  # TD error
    return δs.astype(np.float32)

# Main function for running an episode
def run_episode(
    mod,
    environment,
    loss_hp,
    reward_location=np.zeros(2),
    agent_state=np.zeros(2),
    hidden=True,
    batch=2,
    calc_loss=True,
    initial_params=[]
):
    # Extract some useful variables
    ed = environment.dimensions
    Nout = mod.model_properties.Nout
    Nhidden = mod.model_properties.Nhidden
    Nstates = ed.Nstates
    Naction = ed.Naction
    T = ed.T

    # Initialize reward location, walls, and agent state
    world_state, agent_input = environment.initialize(
        reward_location, agent_state, batch, mod.model_properties, initial_params=initial_params
    )

    # Arrays for storing outputs y
    agent_outputs = np.empty((Nout, batch, 0), dtype=np.float32)
    if hidden:  # Also store and return hidden states and world states
        hs = np.empty((Nhidden, batch, 0), dtype=np.float32)  # Hidden states
        world_states = []  # World states
    actions = np.empty((1, batch, 0), dtype=np.int32)  # List of actions
    rews = np.empty((1, batch, 0), dtype=np.float32)  # List of rewards

    # Project initial hidden state to batch size
    if isinstance(mod.network, list) or isinstance(mod.network, torch.nn.ModuleList):
        gru_layer = mod.network[GRUind]
    else:
        gru_layer = mod.network

    if hasattr(gru_layer, 'state0'):
        h_rnn = gru_layer.state0 + torch.zeros((Nhidden, batch), dtype=torch.float32)
    else:
        h_rnn = torch.zeros((Nhidden, batch), dtype=torch.float32)

    Lpred = 0.0  # Accumulate prediction loss
    Lprior = 0.0  # Accumulate regularization loss

    # Run until all episodes in the batch are finished
    while np.any(world_state.environment_state.time < (T + 1 - 1e-2)):
        if hidden:
            hs = np.concatenate((hs, h_rnn[:, :, np.newaxis]), axis=2)  # Append hidden state
            world_states.append(world_state)  # Append world state

        # Agent input is Nin x batch
        h_rnn, agent_output, a = mod.forward(mod, ed, agent_input, h_rnn)  # RNN step
        active = world_state.environment_state.time < (T + 1 - 1e-2)  # Active episodes

        # Compute regularization loss
        if calc_loss:
            Lprior += prior_loss(agent_output, world_state.agent_state, active, mod)

        # Step the environment
        rew, agent_input, world_state, predictions = environment.step(
            agent_output, a, world_state, environment.dimensions, mod.model_properties, mod, h_rnn
        )
        rew, agent_input, a = zeropad_data(rew, agent_input, a, active)  # Mask finished episodes

        agent_outputs = np.concatenate((agent_outputs, agent_output[:, :, np.newaxis]), axis=2)  # Store output
        rews = np.concatenate((rews, rew[:, :, np.newaxis]), axis=2)  # Store reward
        actions = np.concatenate((actions, a[:, :, np.newaxis]), axis=2)  # Store action

        if calc_loss:
            s_index, r_index = predictions  # True state and reward locations
            Lpred += calc_prediction_loss(agent_output, Naction, Nstates, s_index, active)
            Lpred += calc_rew_prediction_loss(agent_output, Naction, Nstates, r_index, active)

    # Calculate TD errors
    δs = calc_deltas(rews[0, :, :], agent_outputs[Naction, :, :])

    L = 0.0
    N = rews.shape[2] if calc_loss else 0  # Total number of iterations
    for t in range(N):  # Iterations
        active = (actions[0, :, t] > 0.5).astype(np.float32)  # Zero for finished episodes
        Vterm = δs[:, t] * agent_outputs[Naction, :, t]  # Value function
        L -= np.sum(loss_hp.βv * Vterm * active)  # Loss
        for b in np.where(active > 0)[0]:  # For each active episode
            RPE_term = δs[b, t] * agent_outputs[actions[0, b, t], b, t]  # PG term
            L -= loss_hp.βr * RPE_term  # Add to loss

    L += loss_hp.βp * Lpred  # Add predictive loss for internal world model
    L -= loss_hp.βe * Lprior  # Add prior loss
    L /= batch  # Normalize by batch

    if hidden:
        return L, agent_outputs, rews[0, :, :], actions[0, :, :], world_states, hs  # Return environment and hidden states
    return L, agent_outputs, rews[0, :, :], actions[0, :, :], world_state.environment_state

# Wrapper for Flux-like behavior in Python
def model_loss(mod, environment, loss_hp, batch_size):
    loss = run_episode(mod, environment, loss_hp, hidden=False, batch=batch_size)[0]  # hidden=False does not return hidden states
    return loss
