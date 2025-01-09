import numpy as np
from scipy.special import logsumexp
from random import randint
from .walls import onehot_from_loc,state_from_loc,gen_input
from .a2c import GRUind
from .planning import PlanState

def model_tree_search(goal, world_state, model, h_rnn, plan_inds, times, ed, mp, planner, Print=False):
    Larena, Naction = ed.Larena, ed.Naction
    Nstates = Larena ** 2

    batch = h_rnn.shape[1]
    path = np.zeros((4, planner.Lplan, batch))
    all_Vs = np.zeros(batch, dtype=np.float32)  # value functions
    found_rew = np.zeros(batch, dtype=np.float32)  # did I finish planning
    plan_states = np.zeros((planner.Lplan, batch), dtype=np.int32)
    wall_loc = world_state.environment_state.wall_loc

    # Only consider planning states
    h_rnn = h_rnn[:, plan_inds]
    goal = goal[plan_inds]
    times = times[plan_inds]
    wall_loc = wall_loc[:, :, plan_inds]
    ytemp = h_rnn  # Same for GRU

    agent_input = np.zeros(mp.Nin)  # Instantiate
    new_world_state = world_state

    for n_steps in range(planner.Lplan):
        batch = len(goal)  # Number of active states

        if n_steps > 1.5:  # Start from current hidden state
            # Generate new output
            h_rnn, ytemp = model.network[GRUind].cell(h_rnn, agent_input)  # Forward pass

        # Generate actions from hidden activity
        logπ_V = model.policy(ytemp)
        # Normalize over actions
        logπ = logπ_V[:4, :] - logsumexp(logπ_V[:4, :], axis=0)  # Softmax
        Vs = logπ_V[5, :] / 10.0  # Range ~ [0,1]

        πt = np.exp(logπ)
        a = np.zeros((1, batch), dtype=np.int32)  # sample actions
        for b in range(batch):
            a[0, b] = np.random.choice(len(πt[:, b]), p=πt[:, b])
        # a[:] = np.random.choice(4, size=batch)  # random action

        # Record actions
        for ib, b in enumerate(plan_inds):
            path[a[ib] - 1, n_steps, b] = 1.0  # 'a' in local coordinates, 'path' in global

        # Generate predictions
        ahot = np.zeros((Naction, batch))  # One-hot
        for b in range(batch):
            ahot[a[b] - 1, b] = 1.0
        prediction_input = np.vstack([ytemp, ahot])  # Input to prediction module
        prediction_output = model.prediction(prediction_input)  # Output from prediction module

        # Draw new states
        spred = prediction_output[:Nstates, :]  # Predicted states (Nstates x batch)
        spred = spred - logsumexp(spred, axis=0)  # Softmax over states
        state_dist = np.exp(spred)  # State distribution
        new_states = np.argmax(state_dist, axis=0).astype(np.int32)  # maximum likelihood new states

        if Print:
            print(n_steps, batch, np.mean(np.max(πt, axis=0)), np.mean(np.max(state_dist, axis=0)))

        # Record information about having finished
        not_finished = np.where(new_states != goal)[0]  # Vector of states that have not finished!
        finished = np.where(new_states == goal)[0]  # Found the goal location on these ones

        all_Vs[plan_inds] = Vs  # Store latest value
        plan_states[n_steps, plan_inds] = new_states  # Store states
        found_rew[plan_inds[finished]] += 1.0  # Record where we found the goal location

        if len(not_finished) == 0:
            return path, all_Vs, found_rew, plan_states  # Finish if all done

        # Only consider active states going forward
        h_rnn = h_rnn[:, not_finished]
        goal = goal[not_finished]
        plan_inds = plan_inds[not_finished]
        times = times[not_finished] + 1.0  # Increment time
        wall_loc = wall_loc[:, :, not_finished]
        reward_location = onehot_from_loc(Larena, goal)  # One-hot
        xplan = np.zeros((planner.Nplan_in, len(goal)))  # No planning input

        # Reward inputs
        rew = np.zeros((1, len(not_finished)))  # We continue with the ones that did not get reward

        # Update world state
        new_world_state = {
            'agent_state': state_from_loc(Larena, new_states[not_finished]),
            'environment_state': {
                'wall_loc': wall_loc,
                'reward_location': reward_location,
                'time': times
            },
            'planning_state': PlanState(xplan, None)
        }

        # Generate input
        agent_input = gen_input(new_world_state, ahot[:, not_finished], rew, ed, mp)

    return path, all_Vs, found_rew, plan_states


def model_planner(world_state, ahot, ed, agent_output, at_rew, planner, model, h_rnn, mp,
                  Print=False, returnall=False, true_transition=False):

    Larena = ed.Larena
    Naction = ed.Naction
    Nstates = ed.Nstates
    batch = ahot.shape[1]
    times = world_state['environment_state']['time']

    plan_inds = np.where(np.logical_and(ahot[4, :], ~at_rew))[0]  # Everywhere we stand still not at the reward
    rpred = agent_output[(Naction + Nstates + 2):(Naction + Nstates + 1 + Nstates), :]
    goal = np.argmax(rpred, axis=0)  # Index of ML goal location

    # Agent-driven planning
    path, all_Vs, found_rew, plan_states = model_tree_search(goal, world_state, model, h_rnn, plan_inds, times, ed, mp, planner, Print=Print)

    xplan = np.zeros((planner.Nplan_in, batch))
    for b in range(batch):
        xplan[:, b] = np.concatenate([path[:, :, b].flatten(), [found_rew[b]]])
    planning_state = PlanState(xplan, plan_states)

    if returnall:
        return planning_state, plan_inds, (path, all_Vs, found_rew, plan_states)
    return planning_state, plan_inds
