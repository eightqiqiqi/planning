# 本脚本用于分析回放对行为的因果影响
# 我们分别分析成功和失败的回放

# 导入必要的脚本
import numpy as np
import random
import pickle
import bson
from .anal_utils import dist_to_rew,planning_state,gen_input,loaddir,greedy_actions,datadir,seeds,N,Lplan,epoch
from ..src.io import recover_model
from ..src.walls_build import build_environment,act_and_receive_reward
from ..src.model import ModularModel,forward_modular
from ..src.a2c import GRUind
from ..src.planning import build_planner
from ..src.environment import WorldState

run_default_analyses =False
try:
    print(f"running default analyses: {run_default_analyses}")
except Exception as e:
    run_default_analyses = True

# 定义因果回放分析函数
def run_causal_rollouts(seeds, N, Lplan, epoch, prefix="", model_prefix=""):
    #print("分析成功和失败回放后的行为")
    print("analysing behaviour after successful and unsuccessful replays")

    for seed in seeds:  # iterate through independently trained models

        fname = f"{model_prefix}N{N}_T50_Lplan{Lplan}_seed{seed}_{epoch}"  # model name
        print(f"loading {fname}")
        network, opt, store, hps, policy, prediction = recover_model(f"{loaddir}{fname}")  # load parameters

        Larena = hps["Larena"]  # arena size
        # construct environment
        model_properties, wall_environment, model_eval = build_environment(
            Larena, hps["Nhidden"], hps["T"], Lplan=hps["Lplan"], greedy_actions=greedy_actions
        )
        m = ModularModel(model_properties, network, policy, prediction, forward_modular)  # construct model

        # set some basic parameters
        batch = 1
        ed = wall_environment.dimensions
        Nout, Nhidden = m.model_properties.Nout, m.model_properties.Nhidden
        Nstates, T = ed.Nstates, ed.T
        nreps = 1000  # number of environments to consider

        # initialize some containers for storing variables of interest
        p_continue_sim = np.full((3, nreps), np.nan)  # probability of doing another rollout
        p_initial_sim = np.full((3, nreps), np.nan)  # initial probability of rollout
        p_simulated_actions = np.full((3, nreps), np.nan)  # probability of taking a_hat
        p_simulated_actions_old = np.full((3, nreps), np.nan)  # initial probability of taking a_hat
        time_to_second_rew = np.full((3, nreps), np.nan)  # time to get to reward
        plan_dists = np.full((3, nreps), np.nan)  # distance to reward at rollout time
        sim_lengths = np.full((3, nreps), np.nan)  # number of rollout actions
        hidden_old = np.full((3, hps["Nhidden"], nreps), np.nan)  # hidden state before rollout
        hidden_new = np.full((3, hps["Nhidden"], nreps), np.nan)  # hidden state after rollout
        V_old, V_new = [np.full((3, nreps), np.nan) for _ in range(2)]  # value functions before and after rollout
        planning_is = np.full((3, nreps), np.nan)  # number of attempts to get this scenario

        for rep in range(1, nreps + 1):  # for each repetition (i.e. newly sampled environment)
            if rep % 200 == 0:
                print(f"environment {rep} of {nreps}")

            for irew, plan_to_rew in enumerate([True, False, None]):  # for successful/unsuccessful/no rollout
                np.random.seed(rep)  # set random seed for consistent environment

                # initialize environment
                world_state, agent_input = wall_environment.initialize(
                    np.zeros(2), np.zeros(2), batch, m.model_properties
                )
                agent_state = world_state.agent_state
                h_rnn = m.network[GRUind].cell.state0 + np.zeros((Nhidden, batch), dtype=np.float32)  # expand hidden state
                rew = np.zeros(batch)
                tot_rew = np.zeros(batch)
                t_first_rew = 0  # time of first reward

                # let the agent act in the world
                exploit = np.zeros(batch, dtype=bool)  # are we in the exploitation phase
                planner, initial_plan_state = build_planner(Lplan, Larena)  # initialize planning module
                all_rews = []
                # instantiate some variables
                sim_a, Vt_old, h_old, path, πt_old = [np.nan] * 5
                iplan = 0
                just_planned, stored_plan = False, False

                while np.any(world_state.environment_state.time < (40 - 1e-2)):  # iterate through some trials

                    h_rnn, agent_output, a = m.forward(m, ed, agent_input, h_rnn)  # RNN step
                    active = world_state.environment_state.time < (T + 1 - 1e-2)  # active episodes
                    πt = np.exp(agent_output[:5, :])  # current policy
                    Vt = agent_output[5, 0]  # current value function

                    if just_planned and not stored_plan:  # if we just performed a rollout and this is the first one
                        stored_plan = True  # we have now performed a rollout
                        p_continue_sim[irew, rep] = πt[4]  # store the new policy
                        p_initial_sim[irew, rep] = πt_old[4]  # store the old policy
                        # pre/post probabilities of taking the imagined action
                        if np.isnan(sim_a):
                            p_a, p_a_old = np.nan, np.nan
                        else:
                            p_a, p_a_old = πt[sim_a], πt_old[sim_a]
                        p_simulated_actions[irew, rep] = p_a
                        p_simulated_actions_old[irew, rep] = p_a_old
                        sim_lengths[irew, rep] = np.sum(path[:, :, 0])  # rollout length

                        ps = world_state.environment_state.reward_location
                        ws = world_state.environment_state.wall_loc
                        state = np.array(world_state.agent_state, dtype=int)
                        plan_dists[irew, rep] = dist_to_rew(ps, ws, Larena)[state[0], state[1]]  # distance to goal
                        
                        # store value functions, hidden states, and planning iteration
                        V_old[irew, rep] = Vt_old
                        V_new[irew, rep] = Vt
                        hidden_old[irew, :, rep] = h_old[:]
                        hidden_new[irew, :, rep] = h_rnn[:]
                        planning_is[irew, rep] = iplan

                    # store most recent variables for an additional iteration
                    πt_old = πt.copy()
                    Vt_old = Vt.copy()
                    h_old = h_rnn.copy()

                    if (tot_rew[0] == 1) and stored_plan and (a[0] > 4.5):
                        # if we're in trial two and have already planned, don't plan any more
                        if greedy_actions:
                            a[0] = np.argmax(πt[:4])  # only consider physical actions
                        else:
                            a[0] = np.random.choice(np.arange(4), p=πt[:4] / np.sum(πt[:4]))

                    just_planned = False  # have we just done a rollout
                    donot_plan = ~(active & (a[:] > 4.5) & (rew[:] < 0.5))  # whether to plan

                    if plan_to_rew is None and exploit[0] and not stored_plan and not donot_plan[0]:
                        # if we should have planned but are in the no-planning condition, store some data
                        if greedy_actions:
                            a[0] = np.argmax(πt[:4])  # skip planning
                        else:
                            a[0] = np.random.choice(np.arange(4), p=πt[:4] / np.sum(πt[:4]))
                        sim_a = a[0]
                        just_planned = True  # as if we have planned
                        donot_plan[0] = True  # now don't plan again

                    # explicitly run planning and return states
                    ahot = np.zeros((5, batch), dtype=np.float32)
                    for b in range(batch):
                        ahot[int(a[b]), b] = 1.0  # 1-hot action representation
                    iplan = 0  # number of resamples
                    cont = True  # whether we should continue resampling
                    planning_state = None  # result of rollout

                    while cont:  # while we haven't finished
                        iplan += 1  # iterate attempt number
                        # sample a rollout
                        planning_state, plan_inds, (path, all_Vs, found_rew, plan_states) = planner.planning_algorithm(
                            world_state,
                            ahot,
                            wall_environment.dimensions,
                            agent_output,
                            donot_plan,
                            planner,
                            m,
                            h_rnn,
                            m.model_properties,
                            returnall=True
                        )
                    
                        pinput = planning_state.plan_input  # feedback from rollout
                        found_rew = pinput[-1, :] > 0.5  # was it successful

                        if not exploit[0] or donot_plan[0] or stored_plan:
                            # if (i) in exploration, (ii) not planning, or (iii) already stored a plan, continue as normal
                            cont = False
                        elif plan_to_rew and found_rew[0]:  # in 'successful' condition and sampled successful rollout
                            cont = False  # don't sample anymore
                            sim_a = np.argmax(path[:, 0, 0])  # first sampled action
                            just_planned = True  # just did a rollout
                        elif not plan_to_rew and not found_rew[0]:  # in 'unsuccessful' condition and samples unsuccessful rollout
                            cont = False  # don't sample anymore
                            sim_a = np.argmax(path[:, 0, 0])  # first sampled action
                            just_planned = True  # just did a rollout
                        elif iplan > 100:  # if we've exceeded our limit
                            cont = False  # don't sample anymore
                            sim_a = np.nan  # no sampled action
                            just_planned = True  # as if we just did a rollout (give up)

                    if rew[0] > 0.5:  # if we found reward
                        if tot_rew[0] == 0:  # first reward
                            t_first_rew = world_state.environment_state.time[0]  # store time of first reward
                        elif (tot_rew[0] == 1) and stored_plan:  # second reward and we did a rollout to start the trial
                            time_to_second_rew[irew, rep] = (world_state.environment_state.time[0] - t_first_rew)  # store time to rew

                    tot_rew += (rew > 0.5).astype(np.float64)  # total reward accumulated
                    exploit[rew[:] > 0.5] = True  # exploitation phase if we've found reward

                    # now perform a step of the environment dynamics
                    rew, world_state, predictions, ahot, teleport = act_and_receive_reward(
                        a, world_state, planner, wall_environment.dimensions, agent_output, m, h_rnn, m.model_properties
                    )

                    # overwrite the rollout
                    world_state = WorldState(
                        agent_state=world_state.agent_state,
                        environment_state=world_state.environment_state,
                        planning_state=planning_state  # the one sampled above
                    )
                    # check that we successfully overwrote the rollout
                    assert np.all(planning_state.plan_input == world_state.planning_state.plan_input)

                    # generate input for the agent
                    agent_input = gen_input(world_state, ahot, rew, wall_environment.dimensions, m.model_properties)
                    rew[~active] = 0.0  # zero out for finished episodes

                if (tot_rew[0] == 1) and stored_plan:
                    # did a rollout during trial 2 but didn't find reward
                    time_to_second_rew[irew, rep] = (world_state.environment_state.time[0] - t_first_rew)

    # evaluation and plotting

        no_nans = np.where(~np.isnan(plan_dists[2, :]))  # indices where we stored data
        assert np.all((plan_dists[0, :] == plan_dists[1, :])[no_nans])  # check that we used the same environments

        # collect data
        data = {
            "plan_dists": plan_dists,
            "p_simulated_actions": p_simulated_actions,
            "p_simulated_actions_old": p_simulated_actions_old,
            "p_continue_sim": p_continue_sim,
            "sim_lengths": sim_lengths,
            "time_to_second_rew": time_to_second_rew,
            "p_initial_sim": p_initial_sim,
            "V_old": V_old,
            "V_new": V_new,
            "hidden_old": hidden_old,
            "hidden_new": hidden_new,
            "planning_is": planning_is
        }

        # write data
        filename = f"{datadir}/{prefix}causal_N{N}_Lplan{Lplan}_{seed}_{epoch}.bson"
        with open(filename, 'wb') as f:
            f.write(bson.BSON.encode({'data': data}))

# run_default_analyses is a global parameter in anal_utils.jl
if run_default_analyses:
    run_causal_rollouts(seeds=seeds, N=N, Lplan=Lplan, epoch=epoch)



            





