import numpy as np

class PlanState:
    def __init__(self, plan_input, plan_cache=None):
        self.plan_input = plan_input
        self.plan_cache = plan_cache

class Planner:
    def __init__(self, Lplan, Nplan_in, Nplan_out, planning_time, planning_cost, planning_algorithm, constant_rollout_time):
        self.Lplan = Lplan
        self.Nplan_in = Nplan_in
        self.Nplan_out = Nplan_out
        self.planning_time = planning_time
        self.planning_cost = planning_cost
        self.planning_algorithm = planning_algorithm
        self.constant_rollout_time = constant_rollout_time

def none_planner(world_state, ahot, ep, agent_output, at_rew, planner, model, h_rnn, mp):
    batch = ahot.shape[1]
    xplan = np.zeros((0, batch), dtype=np.float32)  # No input
    plan_inds = []  # No indices
    plan_cache = None  # No cache
    planning_state = PlanState(xplan, plan_cache)

    return planning_state, plan_inds

def build_planner(Lplan, Larena, planning_time=1.0, planning_cost=0.0, constant_rollout_time=True):
    Nstates = Larena**2

    if Lplan <= 0.5:
        Nplan_in, Nplan_out = 0, 0
        planning_algorithm = none_planner
        initial_plan_state = lambda batch: PlanState([], [])
    else:
        Nplan_in = 4 * Lplan + 1  # Action sequence and whether we ended at the reward location
        Nplan_out = Nstates  # Reward location
        from .model_planner import model_planner
        planning_algorithm = model_planner  # This needs to be defined elsewhere
        planning_time = 0.3  # Planning needs to be fairly cheap here
        initial_plan_state = lambda batch: PlanState([], [])  # We don't use a cache

    planner = Planner(Lplan, Nplan_in, Nplan_out, planning_time, planning_cost, planning_algorithm, constant_rollout_time)
    return planner, initial_plan_state
