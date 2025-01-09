import numpy as np
import torch
from torch.autograd import Function

# WallState class to represent the environment state
class WallState:
    def __init__(self, reward_location, wall_loc, time=None):
        self.reward_location = reward_location
        self.wall_loc = wall_loc
        self.time = time if time is not None else np.zeros(1)

# Utility functions
def state_ind_from_state(Larena, state):
    """
    Convert state (2, batch) to state index (batch,)
    """
    return Larena * (state[0, :] - 1) + state[1, :]

@torch.no_grad()
def onehot_from_loc(Larena, loc):
    """
    Convert location indices (batch,) to one-hot encoding (Nstates, batch)
    """
    Nstates = Larena ** 2
    batch = len(loc)
    shot = np.zeros((Nstates, batch))
    for b in range(batch):
        shot[loc[b], b] = 1
    return shot

@torch.no_grad()
def onehot_from_state(Larena, state):
    """
    Convert state (2, batch) to one-hot encoding (Nstates, batch)
    """
    state_ind = state_ind_from_state(Larena, state)  # (batch,)
    return onehot_from_loc(Larena, state_ind)

def state_from_loc(Larena, loc):
    """
    Convert location indices (1, batch) to state (2, batch)
    """
    return np.vstack([(loc - 1) // Larena + 1, (loc - 1) % Larena + 1])

def state_from_onehot(Larena, shot):
    # input: Nstates x batch
    # output: 2 x batch
    loc = [sorted(range(len(shot[:, b])), key=lambda i: -shot[i, b])[0] for b in range(shot.shape[1])]
    loc = np.column_stack(loc)
    return state_from_loc(Larena, loc)

def get_wall_input(state, wall_loc):
    """
    Get wall information as input
    """
    input = np.vstack([wall_loc[:, 0, :], wall_loc[:, 2, :]])  # all horizontal and all vertical walls
    return input

def gen_input(world_state, ahot, rew, ed, model_properties):
    batch = rew.shape[1]
    newstate = world_state.agent_state
    wall_loc = world_state.environment_state.wall_loc
    Naction = ed.Naction
    Nstates = ed.Nstates
    shot = onehot_from_state(ed.Larena, newstate)  # one-hot encoding (Nstates x batch)
    wall_input = get_wall_input(newstate, wall_loc)  # get input about walls
    Nwall_in = wall_input.shape[0]
    Nin = model_properties.Nin

    plan_input = world_state.planning_state.plan_input
    if isinstance(plan_input, list):  # Ensure plan_input is numpy array
        plan_input = np.array(plan_input, dtype=np.float32)

    Nplan_in = plan_input.shape[0]

    x = np.zeros((Nin, batch), dtype=np.float32)
    x[:Naction, :] = ahot
    x[Naction, :] = rew[:]
    x[Naction + 1, :] = world_state.environment_state.time / 50.0  # smaller time input in [0,1]
    x[(Naction + 2):(Naction + 2 + Nstates), :] = shot
    x[(Naction + 2 + Nstates):(Naction + 2 + Nstates + Nwall_in), :] = wall_input

    if plan_input.size > 0:  # set planning input
        x[(Naction + 2 + Nstates + Nwall_in):(Naction + 2 + Nstates + Nwall_in + Nplan_in), :] = plan_input

    return x.astype(np.float32)

@torch.no_grad()
def get_rew_locs(reward_location):
    return [np.argmax(reward_location[:, i]) for i in range(reward_location.shape[1])]