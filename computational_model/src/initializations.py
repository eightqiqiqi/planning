import numpy as np
from scipy.stats import rv_discrete
from .walls import state_from_loc,WallState,gen_input
from .environment import WorldState
from .maze import maze

def reset_agent_state(Larena, reward_location, batch):
    Nstates = Larena**2
    agent_state = np.random.choice(Larena, (2, batch))  # random starting location (2 x batch)

    # Make sure we cannot start at reward
    for b in range(batch):
        tele_reward_location = np.ones(Nstates) / (Nstates - 1)
        tele_reward_location[reward_location[:, b].astype(bool)] = 0
        agent_state[:, b] = state_from_loc(
            Larena, np.random.choice(Nstates, p=tele_reward_location)
        ).reshape(-1)
    return agent_state

# Task-specific initialization
def gen_maze_walls(Larena, batch):
    wall_loc = np.zeros((Larena**2, 4, batch), dtype=np.float32)  # whether there is a wall between neighboring agent_states
    for b in range(batch):
        wall_loc[:, :, b] = maze(Larena)
    return wall_loc

def initialize_arena(reward_location, agent_state, batch, model_properties, environment_dimensions, initial_plan_state, initial_params=[]):
    # Ignore gradients in this context
    Larena = environment_dimensions.Larena
    Nstates = Larena**2
    rew_loc = np.random.choice(Nstates, batch, p=np.ones(Nstates) / Nstates)

    if np.max(reward_location) <= 0:
        reward_location = np.zeros((Nstates, batch), dtype=np.float32)  # Nstates x batch
        for b in range(batch):
            reward_location[rew_loc[b], b] = 1.0

    if np.max(agent_state) <= 0:
        agent_state = reset_agent_state(Larena, reward_location, batch)

    if len(initial_params) > 0:  # Load environment
        wall_loc = initial_params
    else:
        wall_loc = gen_maze_walls(Larena, batch)

    # Note: start at t=1 for backward compatibility
    world_state = WorldState(
        environment_state=WallState(
            wall_loc=wall_loc.astype(np.int32),
            reward_location=reward_location.astype(np.float32),
            time=np.ones(batch, dtype=np.float32),
        ),
        agent_state=agent_state.astype(np.int32),
        planning_state=initial_plan_state(batch),
    )

    ahot = np.zeros((environment_dimensions.Naction, batch), dtype=np.float32)  # No actions yet
    rew = np.zeros((1, batch), dtype=np.float32)  # No rewards yet
    x = gen_input(world_state, ahot, rew, environment_dimensions, model_properties)

    return world_state, x.astype(np.float32)
