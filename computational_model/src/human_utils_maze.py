import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
import re
from .walls import state_ind_from_state,onehot_from_state

# Define adict mapping (same as in Julia)
adict = {
    '["Up"]': 3, 
    '["Down"]': 4, 
    '["Right"]': 1, 
    '["Left"]': 2
}

def find_seps(string):
    seps = [
        [m.start() for m in re.finditer(r"]],\[\[", string)],
        [m.start() for m in re.finditer(r"]],\[", string)],
        [m.start() for m in re.finditer(r"\],\[\[", string)],
        [m.start() for m in re.finditer(r"\],\[", string)]
    ]
    return sorted(sum(seps, []))

def get_wall_rep(wallstr, arena):
    seps = find_seps(wallstr)
    columns = [
        wallstr[2:(seps[3] - 2)],
        wallstr[(seps[3] + 2):(seps[7] - 2)],
        wallstr[(seps[7] + 2):(seps[11] - 2)],
        wallstr[(seps[11] + 2):(len(wallstr) - 2)]
    ]
    subseps = [[0] + find_seps(col) + [len(col) + 1] for col in columns]
    
    wdict = {
        '["Top"]': 3, 
        '["Bottom"]': 4, 
        '["Right"]': 1, 
        '["Left"]': 2
    }

    new_walls = np.zeros((16, 4))
    for i, col in enumerate(columns):
        for j in range(4):
            ind = state_ind_from_state(arena, [i, j])[0]
            s1, s2 = subseps[i][j], subseps[i][j + 1]
            entries = col[(s1 + 2):(s2 - 2)].split(",")
            for entry in entries:
                if len(entry) > 0.5:
                    new_walls[ind, wdict[entry]] = 1
    return new_walls

import numpy as np
import pandas as pd

def extract_maze_data(db, user_id, Larena, T=100, max_RT=5000, game_type="play", 
                      skip_init=1, skip_finit=0):
    Nstates = Larena**2

    # Query episodes for the given user and game type
    epis = pd.DataFrame(db.execute(
        f"SELECT * FROM episodes WHERE user_id = {user_id} AND game_type = '{game_type}'"
    ))

    # Discard episodes with a failed attention check
    if "attention_problem" in epis.columns:
        atts = epis["attention_problem"]
        keep = np.where(atts == "null")[0]
        epis = epis.iloc[keep]

    ids = epis["id"].values  # Episode IDs

    # Ensure each episode has at least 2 steps
    stepnums = [
        len(pd.DataFrame(db.execute(f"SELECT * FROM steps WHERE episode_id = {id}")))
        for id in ids
    ]
    ids = ids[np.array(stepnums) > 1.5]

    # Allow for discarding the first/last few episodes
    inds = np.arange(1 + skip_init, len(ids) - skip_finit)
    ids = ids[inds]

    batch_size = len(ids)

    # Initialize arrays
    rews, actions, times = np.zeros((batch_size, T)), np.zeros((batch_size, T)), np.zeros((batch_size, T))
    states = np.ones((2, batch_size, T))
    trial_nums, trial_time = np.zeros((batch_size, T)), np.zeros((batch_size, T))
    wall_loc, ps = np.zeros((16, 4, batch_size)), np.zeros((16, batch_size))
    for b in range(batch_size):
        steps = pd.DataFrame(db.execute(f"SELECT * FROM steps WHERE episode_id = {ids[b]}"))
        trial_num = 1
        t0 = 0

        wall_loc[:, :, b] = get_wall_rep(epis.iloc[inds[b]]["walls"], Larena)
        ps[:, b] = onehot_from_state(
            Larena, [int(epis.iloc[inds[b]]["reward"][i]) + 1 for i in [2, 4]]
        )
        Tb = len(steps)  # Steps in this trial

        for i in reversed(range(Tb)):  # Steps are stored in reverse order
            t = steps.iloc[i]["step"]
            if (t > 0.5) and (i < Tb - 0.5 or steps.iloc[i]["action_time"] < 20000):  # Last action can carry over
                times[b, t - 1] = steps.iloc[i]["action_time"]
                rews[b, t - 1] = int(steps.iloc[i]["outcome"] == '["Hit_reward"]')
                actions[b, t - 1] = adict[steps.iloc[i]["action_type"]]
                states[:, b, t - 1] = [int(steps.iloc[i]["agent"][j]) + 1 for j in [2, 4]]

                trial_nums[b, t - 1] = trial_num
                trial_time[b, t - 1] = t - t0
                if rews[b, t - 1] > 0.5:  # Found reward
                    trial_num += 1  # Next trial
                    t0 = t  # Reset trial time

    # Compute reaction times
    RTs = np.hstack((times[:, :1], times[:, 1:T] - times[:, :T-1]))
    RTs[RTs < 0.5] = np.nan  # End of trial
    for b in range(batch_size):
        rewtimes = np.where(rews[b, :T] > 0.5)[0]
        RTs[b, rewtimes + 1] -= (8 * 50)  # Adjust for reward display time

    # Create one-hot representations of states
    shot = np.full((Nstates, states.shape[1], states.shape[2]), np.nan)
    for b in range(states.shape[1]):
        Tb = int(np.sum(actions[b, :] > 0.5))
        shot[:, b, :Tb] = onehot_from_state(Larena, states[:, b, :Tb].astype(int))

    return rews, actions, states, wall_loc, ps, times, trial_nums, trial_time, RTs, shot
