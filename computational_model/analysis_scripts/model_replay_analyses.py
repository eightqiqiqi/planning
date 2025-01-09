# 导入必要的库
import numpy as np
import random
import time
import pickle
import bson
from .anal_utils import plan_epoch,seeds,loaddir,greedy_actions,datadir
from ..walls_train import recover_model,build_environment
from ..src.model import ModularModel
from ..src.a2c import forward_modular,GRUind
from ..src.planning import build_planner
from ..src.walls_baselines import dist_to_rew,optimal_policy
from ..src.walls import state_from_loc,state_ind_from_state

# 定义一些辅助函数，这些函数将由模型和环境提供
# 假设这里有一些类和函数已经定义好了
# 例如：recover_model, build_environment, ModularModel, wall_environment, etc.

# 模拟生物重放分析
#print("重复生物学重放分析")
print("repeating biological replay analyses in the RL agent")

epoch = plan_epoch  # 训练的epoch
res_dict = {}  # 存储结果的容器

for seed in seeds:  # 遍历独立训练的RL代理
    print(f"\nnew seed {seed}!")
    res_dict[seed] = {}  # 为每个代理创建结果字典

    filename = f"N100_T50_Lplan8_seed{seed}_{epoch}"  # 加载的模型文件
    network, opt, store, hps, policy, prediction = recover_model(loaddir + filename)  # 加载模型参数

    # 构建RL环境
    Larena = hps["Larena"]
    model_properties, wall_environment, model_eval = build_environment(
        Larena, hps["Nhidden"], hps["T"], Lplan=hps["Lplan"], greedy_actions=greedy_actions
    )

    # 构建RL代理
    m = ModularModel(model_properties, network, policy, prediction, forward_modular)

    # 提取一些有用的参数
    ed = wall_environment.dimensions
    Nout, Nhidden = m.model_properties.Nout, m.model_properties.Nhidden
    Nstates, Naction, T = ed.Nstates, ed.Naction, ed.T

    random.seed(2)  # 设置随机种子以确保可复现性
    batch = 25000  # 需要模拟的环境数目

    # 初始化环境
    #print("模拟代理")
    print("simulating agent")
    world_state, agent_input = wall_environment.initialize(
        np.zeros(2), np.zeros(2), batch, m.model_properties
    )
    agent_state = world_state.agent_state
    h_rnn = m.network[GRUind].cell.state0 + np.zeros((Nhidden, batch), dtype=np.float32)  # 扩展隐藏状态
    rew = np.zeros(batch)  # 存储奖励信息的容器

    # 初始化存储数据的容器
    tmax = 200  # 运行的最大迭代次数
    exploit = np.zeros(batch, dtype=bool)  # 是否在开发阶段
    Lplan = model_properties.Lplan  # 计划深度
    plans = np.full((batch, tmax, Lplan), np.nan)  # 存储滚动计划的容器
    plan_as = np.full((batch, tmax, Lplan), np.nan)  # 存储计划动作的容器
    planner, initial_plan_state = build_planner(Lplan, Larena)  # 初始化规划模块
    agent_states = np.zeros((batch, 2, tmax))  # 存储代理的状态
    actions = np.zeros((batch, tmax))  # 存储代理的动作
    all_rews = []  # 存储奖励数据

    rewlocs = [np.argmax(world_state.environment_state.reward_location[:, i]) for i in range(batch)]  # 奖励位置
    success, success_cv = [np.full((batch, tmax, 16), np.nan) for _ in range(2)]  # 存储滚动是否成功

    # 运行每个时间步的循环
    for t in range(1, tmax + 1):  # 每个迭代步骤
        agent_input = agent_input  # 复制到本地变量
        world_state = world_state
        rew = rew
        h_rnn, agent_output, a = m.forward(m, ed, agent_input, h_rnn)  # RNN步进
        active = (world_state.environment_state.time < (T + 1 - 1e-2))  # 激活的episodes
        agent_states[:, :, t] = world_state.agent_state.T  # 存储代理位置
        actions[:, t] = a[:]  # 存储动作
        actions[~active, t] = 0  # 如果episode结束，设置动作为0
        a[~active] = 1  # 如果episode结束，不再滚动

        at_rew = (~(active & exploit & (a[:] > 4.5) & (rew[:] < 0.5)))  # 如果奖励位置，不能进行滚动

        ahot = np.zeros((5, batch), dtype=np.float32)
        for b in range(batch):
            ahot[int(a[b]), b] = 1.0  # one-hot编码的动作表示
        # 运行一次滚动计划
        _, _, (path_cv, _, _, plan_states_cv) = planner.planning_algorithm(
            world_state, ahot, wall_environment.dimensions, agent_output, at_rew, planner, m, h_rnn, m.model_properties, returnall=True
        )

        exploit[rew[:] > 0.5] = True  # 进入开发阶段
        # 进行一次模型更新步骤，使用不同的滚动计划引导行为
        rew, agent_input, world_state, predictions = wall_environment.step(
            agent_output, a, world_state, wall_environment.dimensions, m.model_properties, m, h_rnn
        )
        path = np.reshape(world_state.planning_state.plan_input[:(4 * Lplan), :], (4, Lplan, batch))  # 一系列动作
        plan_states = world_state.planning_state.plan_cache  # 一系列状态

        # 存储滚动计划的结果
        for b in range(batch):  # 每个episode
            if world_state.planning_state.plan_cache[0, b] > 0.5 and not at_rew[b]:  # 如果执行了滚动计划
                plans[b, t, :] = plan_states[:, b]  # 存储状态序列
                plan_as[b, t, :] = [np.argmax(path[:, l, b]) for l in range(Lplan)]  # 存储动作序列
                for loc in range(16):  # 对每个位置
                    success[b, t, loc] = np.any(plan_states[:, b] == loc)  # 存储是否成功
                    success_cv[b, t, loc] = np.any(plan_states_cv[:, b] == loc)  # 交叉验证

        rew[~active] = 0.0  # 对结束的episode奖励设为0
        all_rews.append(rew)  # 存储奖励

    # 合并奖励数据
    wall_loc, ps = world_state.environment_state.wall_loc, world_state.environment_state.reward_location  # 墙和奖励位置
    rews = np.concatenate(all_rews).T  # 合并为一个单一数组

    ## 收集一些汇总数据
    #print("收集数据")
    print("collecting data")
    next_as = np.zeros((batch, tmax))  # 下一步动作
    opt_as = np.zeros((batch, tmax, 4))  # 每个时间点的最优动作
    agent_opt_as = np.zeros((batch, tmax))  # 是否采取最优动作
    goal_steps = np.zeros((batch, tmax))  # 到达目标的步数
    goal_dist = np.zeros((batch, tmax))  # 到目标的最优步数

    trial_ts = np.zeros((batch, tmax))  # 试验中的时间
    trial_ids = np.zeros((batch, tmax))  # 试验号
    trial_anums = np.zeros((batch, tmax))  # 每个试验中的动作号
    for b in range(batch):  # 对每个episode
        Nrew = np.sum(rews[b, :] > 0.5)  # 奖励的数量
        sortrew = np.argsort(-rews[b, :])  # 按照奖励大小排序
        rewts = sortrew[:Nrew]  # 奖励时间点
        diffs = np.concatenate(([rewts], [tmax + 1])) - np.concatenate(([0], rewts))  # 每个试验的持续时间
        trial_ids[b, :] = np.concatenate([np.ones(diffs[i]) * i for i in range(Nrew + 1)])[:tmax]  # 试验号
        trial_ts[b, :] = np.concatenate([np.arange(1, diffs[i] + 1) for i in range(Nrew + 1)])[:tmax]  # 每个试验中的时间

        # 对于结束的episode，时间点置为0
        finished = np.where(actions[b, :] == 0)[0]
        trial_ids[b, finished] = 0
        trial_ts[b, finished] = 0

        # 收集每个试验中的动作号
        ep_as = actions[b, :]  # 当前episode的动作
        for id in range(1, Nrew + 2):  # for each trial
            inds = np.where(trial_ids[b, :] == id)[0]  # timepoints corresponding to this trial
            trial_as = ep_as[inds]  # actions within this trial
            anums = np.zeros(len(inds), dtype=np.int64)  # container for action numbers
            anum = 1  # start at 1
            for a in range(1, len(inds)):  # iterate through the timesteps
                anums[a] = anum  # store action number
                if trial_as[a] <= 4.5:
                    anum += 1  # only increment if physical action
            trial_anums[b, inds] = anums  # store action numbers

        trial_anums[b, finished] = 0  # zero after episode is finished

        # get list of next actions
        for t in range(1, tmax + 1):  # for each iteration
            next = actions[b, t - 1:] [actions[b, t - 1:] < 4.5]  # future physical actions
            if len(next) == 0:  # last action
                next_as[b, t - 1] = 0  # no next action
            else:
                next_as[b, t - 1] = next[0]  # first future action

        # get distance to goal and optimal actions
        dists = dist_to_rew(ps[:, b:b+1], wall_loc[:, :, b:b+1], Larena)  # all distances to goal
        for t in range(tmax):  # Python indexing starts from 0
            state = agent_states[b, :, t].astype(int)  # current agent state
            goal_dist[b, t] = dists[state[0], state[1]]  # minimum distance to goal
            opt_pi = optimal_policy(state, wall_loc[:, :, b:b+1], dists, ed)  # optimal policy
            opt_as[b, t, :] = (opt_pi[:4] > 1e-2).astype(np.float64)  # set of optimal actions
            nexta = int(next_as[b, t])  # next action here
            if nexta > 0.5:
                agent_opt_as[b, t] = float(opt_as[b, t, nexta - 1] > 1e-2)  # was it optimal?
            goal_steps[b, t] = (
                trial_anums[b, np.where(trial_ids[b, :] == trial_ids[b, t])[0][-1]]
                - trial_anums[b, t] + 1
            )  # number of physical actions taken to goal
        
    # calculate wall conformity
    print("calculating wall conformity")

    def get_action(Larena, s1, s2):  # find the action that would take us from state 1 to state 2
        ss = state_from_loc(Larena, np.array([s1, s2]).T)  # convert to index
        s1, s2 = ss[:, 0], ss[:, 1]
        vecs = [np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1])]  # possible actions
        for a in range(4):
            if np.all(s2 == (s1 + vecs[a] + Larena - 1) % Larena + 1):  # find the one that takes us to s2
                return a + 1  # Python indices are 0-based
        return 0
    
    batch_wall_probs = []  # true probability of going through wall
    batch_rand_wall_probs = []  # control probability

    for b in range(batch):
        # plan_times: indices at which the agent performed a rollout
        plan_times = np.where(plans[b, :, 1] > 0.5)[0]
        wall_probs = []  # for this episode
        rand_wall_probs = []  # for this episode

        for t in plan_times:  # go through rollouts
            # concatenate current state and subsequent imagined states
            newplan = np.concatenate([[state_ind_from_state(Larena, agent_states[b, :, t])[0]], plans[b, t, :]])
            newplan_as = plan_as[b, t, :]  # actions that were taken
            nsteps = np.sum(newplan > 0.5) - 1  # number of steps that were taken

            for i in range(nsteps):  # for each step
                s1, s2 = newplan[i], newplan[i + 1]  # first and second state
                if s1 != s2:  # check that we moved
                    walls = world_state.environment_state.wall_loc[int(newplan[i]), :, b]  # wall locations in this episode
                    a = get_action(Larena, s1, s2)  # action that would've taken me there
                    if a > 0.5:  # ignore discontinuous jumps
                        wall_probs.append(walls[int(newplan_as[i])])  # did I move through a wall?
                        rand_wall_probs.append(np.mean(walls))  # baseline probability

        batch_wall_probs.extend(wall_probs)  # append result for this episode
        batch_rand_wall_probs.extend(rand_wall_probs)  # append baseline

    # store result
    res_dict[seed]["batch_wall_probs"] = batch_wall_probs
    res_dict[seed]["batch_rand_wall_probs"] = batch_rand_wall_probs

    # now look at fraction of successful replays
    print("calculating success frequency")
    true_succs, false_succs = [], []  # true and control success fractions
    for b in range(batch):  # for each episode
        inds = np.where(~np.isnan(success[b, :, rewlocs[b]]))[0]  # planning times
        true_succs.extend(success[b, inds, rewlocs[b]])  # success of plans w.r.t true reward locations
        new_false = [
            np.mean(success[b, ind, np.where(np.arange(1, 17) != rewlocs[b])[0]]) for ind in inds
        ]  # w.r.t control locations
        false_succs.extend(new_false)  # store data

    # save data
    res_dict[seed]["true_succs"] = true_succs
    res_dict[seed]["false_succs"] = false_succs

    ## now look at p(goal | plan number)
    print("calculating success by replay number")
    maxL = 5  # maximum number of plans to consider
    for minL in [2, 3]:  # consider a minimum number of rollouts
        succ_byp = np.full((batch, tmax, maxL + 2), np.nan)  # container for storing results
        succ_byp_ctrl = np.full((15, batch, tmax, maxL + 2), np.nan)  # container for storing control results
        minreps, minreps_ctrl = [], []

        for b in range(batch):  # for each episode
            nt, np = 0, 0  # number of sequences and individual rollouts within the sequence
            loc = rewlocs[b]  # reward location
            for t in range(int(np.sum(actions[b, :] > 0.5))):  # for each index before episode finished
                if np.isnan(success_cv[b, t, loc]):  # no rollout
                    np = 0  # reset
                else:
                    if np == 0:  # first plan at this step
                        nt += 1  # increment plan number
                    np += 1  # increment rollout number within sequence
                    if np <= (maxL + 2):
                        succ_byp[b, nt - 1, np - 1] = success_cv[b, t, loc]  # was this rollout successful
                        for i in range(15):  # ctrl locations
                            succ_byp_ctrl[i, b, nt - 1, np - 1] = success_cv[b, t, np.where(np.arange(16) != loc)[0][i]]  # was it successful to the control loc

            newreps = succ_byp[b, :nt, :]  # extract success data
            nplans = np.sum(~np.isnan(newreps), axis=1)  # number of rollouts within each sequence
            inds = np.where((nplans >= minL) & (nplans <= maxL))[0]  # subselect by number of rollouts in sequence
            minreps.append(newreps[inds, :minL])  # store success rates
            minreps_ctrl.append(succ_byp_ctrl[:, b, inds, :minL])  # store controls

        cat_succ = np.vstack(minreps)  # combine arrays across episodes
        cat_succ_ctrl = np.hstack(minreps_ctrl)  # combine array across episodes
        res_dict[seed][f"suc_by_rep_min{minL}"] = cat_succ  # store result
        res_dict[seed][f"suc_by_rep_min{minL}_ctrl"] = cat_succ_ctrl  # store ctrl result

    ## now look at p(follow | goal) and p(follow | no-goal)
    print("calculating behavior by success/non-success")
    succs, nons = [], []

    for b in range(batch):  # for each episode
        succ_inds = np.where(success[b, :, rewlocs[b]] == 1)[0]  # indices of successful rollouts
        non_inds = np.where(success[b, :, rewlocs[b]] == 0)[0]  # indices of unsuccessful rollouts
        succs.extend(plan_as[b, succ_inds, 0] == next_as[b, succ_inds])  # am I consistent with rollout actions after successful rollouts?
        nons.extend(plan_as[b, non_inds, 0] == next_as[b, non_inds])  # am I consistent after unsuccessful rollouts?

    # write the data
    res_dict[seed]["follow_succs"] = succs
    res_dict[seed]["follow_non"] = nons

with open(datadir + "model_replay_analyses.bson", 'wb') as f:
    f.write(bson.BSON.encode({'res_dict': res_dict}))





    


