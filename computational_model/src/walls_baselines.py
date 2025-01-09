import numpy as np
from .walls import state_from_onehot,state_ind_from_state,state_from_loc

def random_policy(x, md, ed, stay=True):
    # if stay is false, only uniform over actual actions
    batch = x.shape[1]
    ys = np.zeros((md.Nout, batch), dtype=np.float32)
    if stay:
        ys[:ed.Naction, :] = np.log(1 / ed.Naction)
    else:
        ys[:ed.Naction - 1, :] = np.log(1 / (ed.Naction - 1))
        ys[ed.Naction - 1, :] = -np.inf
    
    return ys

def dist_to_rew(ps, wall_loc, Larena):
    # compute geodesic distance to reward from each state (i.e., taking walls into account)
    Nstates = Larena**2
    deltas = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])  # transitions for each action
    rew_loc = state_from_onehot(Larena, ps)  # 2x1
    dists = np.full((Larena, Larena), np.nan)  # distances to goal
    dists[rew_loc[0], rew_loc[1]] = 0  # reward has zero distance
    live_states = np.zeros(Nstates, dtype=bool)
    live_states[state_ind_from_state(Larena, rew_loc)[0]] = True  # start from reward location and work backwards
    for step in range(1, Nstates):  # steps from reward
        for state_ind in np.where(live_states)[0]:  # all states I was at in (step-1) steps
            state = state_from_loc(Larena, state_ind)
            for a in range(4):  # for each action
                if not bool(wall_loc[state_ind, a, 0]):  # if I do not hit a wall
                    newstate = state + deltas[a]  # where do I end up in 'step' steps
                    newstate = ((newstate + Larena - 1) % Larena + 1).astype(int)  # 1:L (2xbatch)
                    if np.isnan(dists[newstate[0], newstate[1]]):  # if I haven't gotten here in fewer steps
                        dists[newstate[0], newstate[1]] = step  # got here in step steps
                        new_ind = state_ind_from_state(Larena, newstate)[0]
                        live_states[new_ind] = True  # need to search from here for >step steps
            live_states[state_ind] = False  # done searching for this state
    
    return dists  # return geodesics

def optimal_policy(state, wall_loc, dists, ed):
    # return uniform log policy over actions that minimize the path length to goal
    Naction, Larena = ed.Naction, ed.Larena
    deltas = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])  # transitions for each action
    nn_dists = np.full(4, np.inf)  # distance to reward for each action
    state_ind = state_ind_from_state(Larena, state)[0]  # where am I
    
    for a in range(4):  # for each action
        if not bool(wall_loc[state_ind, a, 0]):  # if I do not hit a wall
            newstate = state + deltas[a]  # where am I now
            newstate = np.mod(newstate + Larena - 1, Larena) + 1  # 1:L (2xbatch)
            nn_dists[a] = dists[newstate[0], newstate[1]]  # how far is this from reward
    
    as_optimal = np.where(nn_dists == np.min(nn_dists))[0]  # all optimal actions
    πt = np.zeros(Naction)
    πt[as_optimal] = 1 / len(as_optimal)  # uniform policy
    
    return πt  # optimal policy
