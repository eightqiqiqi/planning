import numpy as np
from scipy.stats import entropy
from collections import namedtuple
import random
from .walls import state_ind_from_state, onehot_from_state, get_rew_locs, state_from_loc, gen_input,WallState
from .planning import build_planner
from .initializations import initialize_arena
from .a2c import run_episode
from .environment import WorldState,Environment,EnvironmentDimensions
from .model import ModelProperties

# Utility functions
def useful_dimensions(Larena, planner):
    Nstates = Larena ** 2  # number of states in arena
    Nstate_rep = 2  # dimensionality of the state representation (e.g. '2' for x,y-coordinates)
    Naction = 5  # number of actions available
    Nout = Naction + 1 + Nstates  # actions and value function and prediction of state
    Nout += 1  # needed for backward compatibility (this lives between state and reward predictions)
    Nwall_in = 2 * Nstates  # provide full info
    Nin = Naction + 1 + 1 + Nstates + Nwall_in  # 5 actions, 1 rew, 1 time, L^2 states, some walls

    Nin += planner.Nplan_in  # additional inputs from planning
    Nout += planner.Nplan_out  # additional outputs for planning

    return Nstates, Nstate_rep, Naction, Nout, Nin

def update_agent_state(agent_state, amove, Larena):
    new_agent_state = agent_state + np.array([amove[0:1, :] - amove[2:3, :], amove[3:4, :] - amove[4:5, :]])  # 2xbatch
    new_agent_state = ((new_agent_state + Larena - 1) % Larena + 1).astype(np.int32)  # 1:L (2xbatch)
    return new_agent_state

def act_and_receive_reward(a, world_state, planner, environment_dimensions, agent_output, model, h_rnn, mp):
    agent_state = world_state.agent_state
    environment_state = world_state.environment_state
    reward_location = environment_state.reward_location
    wall_loc = environment_state.wall_loc
    Naction = environment_dimensions.Naction
    Larena = environment_dimensions.Larena

    agent_state_ind = state_ind_from_state(Larena, agent_state)  # extract index
    batch = a.shape[1]  # batch size
    Nstates = Larena ** 2

    ahot = np.zeros((Naction, batch))  # attempted action
    amove = np.zeros((Naction, batch))  # actual movement
    rew = np.zeros((1, batch), dtype=np.float32)  # reward collected

    # Construct array of attempted and actual movements
    for b in range(batch):
        abatch = a[0, b]  # action
        ahot[abatch, b] = 1.0  # attempted action
        if (abatch < 4.5) and bool(wall_loc[agent_state_ind[b], abatch, b]):
            rew[0, b] -= 0.0  # penalty for hitting wall?
        else:
            amove[abatch, b] = 1  # only move if we don't hit a wall

    new_agent_state = update_agent_state(agent_state, amove, Larena)  # (x, y) coordinates
    shot = onehot_from_state(Larena, new_agent_state)  # one-hot encoding (Nstates x batch)
    s_index = np.concatenate([np.argsort(-shot[:, b])[0:1] for b in range(batch)])  # corresponding index
    r_index = get_rew_locs(reward_location)  # index of reward location
    predictions = (s_index.astype(np.int32), r_index.astype(np.int32))  # things to be predicted by the agent

    found_rew = reward_location[shot.astype(bool)].astype(bool)  # moved to the reward
    s_old_hot = onehot_from_state(Larena, agent_state)  # one-hot encoding of previous agent_state
    at_rew = reward_location[s_old_hot.astype(bool)].astype(bool)  # at reward before action

    moved = np.sum(amove[0:4, :], axis=0)  # did I perform a movement? (size batch)
    rew[0, found_rew & (moved > 0.5)] = 1  # get reward if agent moved to reward location

    # teleport the agents that found the reward on the previous iteration
    for b in range(batch):
        if at_rew[b]:  # at reward
            tele_reward_location = np.ones(Nstates) / (Nstates - 1)  # where can I teleport to (not rew location)
            tele_reward_location[reward_location[:, b].astype(bool)] = 0
            new_state = np.random.choice(np.arange(Nstates), p=tele_reward_location)  # sample new state uniformly at random
            new_agent_state[:, b] = state_from_loc(Larena, [new_state])  # convert to (x, y) coordinates
            shot[:, b] = 0.0
            shot[new_state, b] = 1.0  # update one-hot location

    # run planning algorithm
    planning_state, plan_inds = planner.planning_algorithm(world_state, ahot, environment_dimensions, agent_output, at_rew, planner, model, h_rnn, mp)

    planned = np.zeros(batch, dtype=bool)
    planned[plan_inds] = True  # which agents within the batch engaged in planning

    # update the time elapsed for each episode
    new_time = np.copy(environment_state.time)
    new_time[~planned] += 1.0  # increment time for acting
    if planner.constant_rollout_time:
        new_time[planned] += planner.planning_time  # increment time for planning
    else:
        plan_states = planning_state.plan_cache
        plan_lengths = np.sum(plan_states[:, planned] > 0.5, axis=0)  # number of planning steps for each batch
        new_time[planned] += plan_lengths * planner.planning_time / 5
        if random.random() < 1e-5:
            print("variable planning time!", plan_lengths * planner.planning_time / 5)

    rew[0, planned] += planner.planning_cost  # cost of planning (in units of rewards; default 0)

    # update the state of the world
    new_world_state = WorldState(
        agent_state=new_agent_state,
        environment_state=WallState(wall_loc, reward_location, new_time),
        planning_state=planning_state
    )

    return rew.astype(np.float32), new_world_state, predictions, ahot, at_rew

def build_environment(Larena, Nhidden, T, Lplan, greedy_actions=False, no_planning=False, constant_rollout_time=True):
    planner, initial_plan_state = build_planner(Lplan, Larena, constant_rollout_time)
    Nstates, Nstate_rep, Naction, Nout, Nin = useful_dimensions(Larena, planner)  # compute some useful quantities
    model_properties = ModelProperties(Nout, Nhidden, Nin, Lplan, greedy_actions, no_planning)
    environment_dimensions = EnvironmentDimensions(Nstates, Nstate_rep, Naction, T, Larena)

    def step(agent_output, a, world_state, environment_dimensions, model_properties, model, h_rnn):
        rew, new_world_state, predictions, ahot, at_rew = act_and_receive_reward(
            a, world_state, planner, environment_dimensions, agent_output, model, h_rnn, model_properties
        )  # take a step through the environment
        agent_input = gen_input(new_world_state, ahot, rew, environment_dimensions, model_properties)
        return rew, agent_input.astype(np.float32), new_world_state, predictions

    def initialize(reward_location, agent_state, batch, mp, initial_params=[]):
        return initialize_arena(reward_location, agent_state, batch, mp, environment_dimensions, initial_plan_state, initial_params=initial_params)

    environment = Environment(initialize, step, environment_dimensions)

    def model_eval(m, batch, loss_hp):
        Nrep = 5
        means = np.zeros((Nrep, batch))
        all_actions = np.zeros((Nrep, batch))
        firstrews = np.zeros((Nrep, batch))
        preds = np.zeros((T - 1, Nrep, batch))
        Naction = environment_dimensions.Naction
        Nstates = environment_dimensions.Nstates

        for i in range(Nrep):
            _, agent_outputs, rews, actions, world_states, _ = run_episode(m, environment, loss_hp, hidden=True, batch=batch)
            agent_states = np.concatenate([x.agent_state for x in world_states], axis=2)
            means[i, :] = np.sum(rews >= 0.5, axis=1)
            all_actions[i, :] = np.mean(actions == 5, axis=1) / np.mean(actions > 0.5, axis=1)
            for b in range(batch):
                firstrews[i, b] = np.argsort(-(rews[b, :] > 0.5))[0]  # time to first reward
            for t in range(T - 1):
                for b in range(batch):
                    pred = np.argsort(-agent_outputs[(Naction + 1 + 1):(Naction + 1 + Nstates), b, t])[0]
                    agent_state = agent_states[:, b, t + 1].astype(np.int32)  # true agent_state(t+1)
                    preds[t, i, b] = onehot_from_state(Larena, agent_state)[int(pred)]

        return np.mean(means), np.mean(preds), np.mean(all_actions), np.mean(firstrews)

    return model_properties, environment, model_eval
