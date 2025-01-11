import numpy as np
import torch
import torch.nn.functional as F
import scipy
from .priors import prior_loss

GRUind = 1

def sample_actions(mod, policy_logits):
    batch = policy_logits.shape[0]  # 40
    a = torch.zeros((1, batch), dtype=torch.int32, device=policy_logits.device)

    πt = torch.exp(policy_logits.float())  # [40,5]
    πt /= πt.sum(dim=1, keepdim=True)  # normalize over actions

    # Handle NaN or Inf in probabilities
    invalid_mask = torch.isnan(πt) | torch.isinf(πt)
    for b in range(batch):
        if invalid_mask[b, :].any():
            πt[b, :] = 0.0
            πt[b, torch.argmax(policy_logits[b, :])] = 1.0

    # Sample actions or select greedy actions
    if mod.model_properties.greedy_actions:
        a[0, :] = torch.argmax(πt, dim=1)
    else:
        # πt 已经是 [40,5]
        print(f"πt shape: {πt.shape}")  # 应该是 [40, 5]
        sampled_actions = torch.multinomial(πt, num_samples=1).squeeze(-1)  # [40]
        a[0, :] = sampled_actions

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
    batch = a.shape[1]
    ahot = torch.zeros((Naction, batch), device=a.device, dtype=torch.float32)
    ahot[a[0, :], torch.arange(batch)] = 1.0  # Efficient indexing
    return ahot

def forward_modular(mod, ed, x, h_rnn):
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32, device=next(mod.network.parameters()).device)
    if isinstance(h_rnn, np.ndarray):
        h_rnn = torch.tensor(h_rnn, dtype=torch.float32, device=next(mod.network.parameters()).device)
    
    Naction, Nstates, batch = ed.Naction, ed.Nstates, x.shape[1]
    
    if isinstance(mod.network, (list, torch.nn.ModuleList)):
        gru_layer = mod.network[GRUind]
    else:
        gru_layer = mod.network
    
    if x.ndim == 2:
        x = x.unsqueeze(0)  # [1, 88, 40]
        x = x.transpose(1, 2)  # [1, 40, 88]
    
    print(f"Input to GRU shape after transpose: {x.shape}")  # [1, 40, 88]
    
    h_rnn, ytemp = gru_layer(x, h_rnn)  
    print(f"GRU output shape: {ytemp.shape}")  # 预期: [1, 40, 100]
    
    # 应用策略网络（和价值网络）
    logpi_V = mod.policy(ytemp)  # [1, 40, 6]
    print("logpi_V", logpi_V.shape)
    
    # 正确切片 logits，将动作 logits 和价值分开
    logpi = logpi_V[:, :, :Naction].squeeze(0)  # [40, 5]
    V = logpi_V[:, :, Naction].squeeze(0)       # [40]
    
    # 归一化 logpi
    logpi = logpi - torch.logsumexp(logpi, dim=1, keepdim=True)  # [40, 5]
    print(f"logpi shape before processing: {logpi.shape}")       # [40, 5]
    
    # 确保 logpi 是 2D 张量 [batch_size, Naction]
    if logpi.ndim > 2:
        logpi = logpi.squeeze(0).transpose(0, 1)  # 不应发生
    elif logpi.ndim == 2:
        pass  # 已经正确
    else:
        raise ValueError(f"Unexpected logpi dimensions: {logpi.shape}")
    
    print(f"logpi shape after processing: {logpi.shape}")  # [40, 5]
    
    # 无需转置，因为 sample_actions 期望输入为 [40, 5]
    logpi_transposed = logpi  # [40, 5]
    print(f"logpi_transposed shape: {logpi_transposed.shape}")  # [40, 5]
    print(f"πt shape: {torch.exp(logpi_transposed).shape}")  # [40, 5]
    
    # 采样动作
    a = sample_actions(mod, logpi_transposed)  # [1, 40]
    # 构造 one-hot 表示
    ahot = construct_ahot(a, Naction)  # [5, 40]
    print(f"ahot shape: {ahot.shape}")  # [5, 40]
    
    # 调整 ahot 的形状为 [40, 5]
    ahot = ahot.transpose(0, 1)  # [40, 5]
    print(f"ahot transposed shape: {ahot.shape}")  # [40, 5]
    
    # 调整 ytemp 的形状为 [40, 100]
    ytemp = ytemp.squeeze(0)  # [40, 100]
    print(f"ytemp squeezed shape: {ytemp.shape}")  # [40, 100]
    
    # 在特征维度上拼接 ytemp 和 ahot，得到 [40, 105]
    prediction_input = torch.cat([ytemp, ahot], dim=1)  # [40, 105]
    print(f"prediction_input shape: {prediction_input.shape}")  # [40, 105]
    
    # 通过预测模块
    prediction_output = mod.prediction(prediction_input)  # 预期输出形状根据预测模块定义
    print(f"prediction_output shape: {prediction_output.shape}")  # 根据预测模块的定义
    
    # 合并 logpi, V 和 prediction_output
    # 首先，将 V 扩展为 [40, 1] 以便拼接
    V_expanded = V.unsqueeze(1)  # [40, 1]
    print(f"V_expanded shape: {V_expanded.shape}")  # [40, 1]
    
    # 合并 logpi 和 V，得到 [40, 6]
    logpi_and_V = torch.cat([logpi, V_expanded], dim=1)  # [40, 6]
    print(f"logpi_and_V shape: {logpi_and_V.shape}")  # [40, 6]
    
    # 确保 prediction_output 的形状为 [40, prediction_features]
    # 假设 prediction_output 的形状为 [40, P]
    # 合并所有部分，得到 [40, 6 + P]
    output = torch.cat([logpi_and_V, prediction_output], dim=1)  # [40, 6 + P]
    print(f"output shape: {output.shape}")  # [40, 6 + P]
    
    return h_rnn, output, a

# Calculate TD errors (deltas)
@torch.no_grad()
def calc_deltas(rews, Vs):
    batch, N = rews.shape
    δs = torch.zeros_like(rews)
    R = torch.zeros(batch, device=rews.device)

    for t in range(N - 1, -1, -1):  # Iterate backwards
        R = rews[:, t] + R
        δs[:, t] = R - Vs[:, t]
    return δs

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

    print("naction",Naction)

    # Initialize reward location, walls, and agent state
    world_state, agent_input = environment.initialize(
        reward_location, agent_state, batch, mod.model_properties, initial_params=initial_params
    )
    print(f"Initial agent_input shape: {agent_input.shape}")

    # Convert agent_input to torch.Tensor and assign to the correct device
    device = next(mod.network.parameters()).device
    agent_input = torch.tensor(agent_input, dtype=torch.float32, device=device)

    # Initialize arrays for storing outputs y
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
        h_rnn = torch.tensor(gru_layer.state0, dtype=torch.float32, device=device) + torch.zeros((Nhidden, batch), dtype=torch.float32, device=device)
    else:
        num_layers = getattr(gru_layer, 'num_layers', 1)
        h_rnn = torch.zeros((num_layers, batch, Nhidden), dtype=torch.float32, device=device)

    Lpred, Lprior = 0.0, 0.0

    # Run until all episodes in the batch are finished
    while np.any(world_state.environment_state.time < (T + 1 - 1e-2)):
        if hidden:
            # Append hidden state
            h_rnn_np = h_rnn.detach().cpu().numpy()
            if hs.shape[2] == 0:  # If hs is empty
                hs = h_rnn_np
            else:
                hs = np.concatenate((hs, h_rnn_np), axis=2)
            # Append world state
            world_states.append(world_state)
        print("agent_input1111111111111", agent_input.shape)
        # Agent input is Nin x batch
        h_rnn, agent_output, a = mod.forward(mod, ed, agent_input, h_rnn)  # RNN step
        print("agent_output11111111", agent_output.shape)
        active = world_state.environment_state.time < (T + 1 - 1e-2)  # Active episodes

        # Compute regularization loss
        if calc_loss:
            Lprior += prior_loss(agent_output, world_state.agent_state, active, mod)

        # Step the environment
        rew, agent_input_np, world_state, predictions = environment.step(
            agent_output, a, world_state, environment.dimensions, mod.model_properties, mod, h_rnn
        )
        rew, agent_input_np, a = zeropad_data(rew, agent_input_np, a, active)  # Mask finished episodes

        # Convert agent_input back to torch.Tensor
        agent_input = torch.tensor(agent_input_np, dtype=torch.float32, device=device)

        # Store outputs
        agent_output_np = agent_output.detach().cpu().numpy()
        agent_outputs = np.concatenate((agent_outputs, agent_output_np[:, :, np.newaxis]), axis=2)  # Store output
        rews = np.concatenate((rews, rew[:, :, np.newaxis]), axis=2)  # Store reward
        actions = np.concatenate((actions, a[:, :, np.newaxis]), axis=2)  # Store action

        if calc_loss:
            s_index, r_index = predictions  # True state and reward locations
            Lpred += calc_prediction_loss(agent_output_np, Naction, Nstates, s_index, active)
            Lpred += calc_rew_prediction_loss(agent_output_np, Naction, Nstates, r_index, active)

    # Calculate TD errors
    δs = calc_deltas(rews[0, :, :], agent_outputs[Naction, :, :])

    L = 0.0
    N = rews.shape[2] if calc_loss else 0  # Total number of iterations
    for t in range(N):  # Iterations
        active = (actions[0, :, t] > 0.5).astype(np.float32)  # Zero for finished episodes
        Vterm = δs[:, t] * agent_outputs[Naction, :, t]  # Value function
        L -= np.sum(loss_hp.βv * Vterm * active)  # Loss
        for b in np.where(active > 0)[0]:  # For each active episode
            action_index = actions[0, b, t]
            if action_index < Naction:
                RPE_term = δs[b, t] * agent_outputs[action_index, b, t]  # PG term
                L -= loss_hp.βr * RPE_term  # Add to loss
            else:
                raise IndexError(f"Action index {action_index} out of bounds for Naction={Naction}")

    L += loss_hp.βp * Lpred  # Add predictive loss for internal world model
    L -= loss_hp.βe * Lprior  # Add prior loss
    L /= batch  # Normalize by batch

    if hidden:
        return L, agent_outputs, rews[0, :, :], actions[0, :, :], world_states, hs  # Return environment and hidden states
    return L, agent_outputs, rews[0, :, :], actions[0, :, :], world_state.environment_state

def model_loss(mod, environment, loss_hp, batch_size):
    loss = run_episode(mod, environment, loss_hp, hidden=False, batch=batch_size)[0] 
    return loss
