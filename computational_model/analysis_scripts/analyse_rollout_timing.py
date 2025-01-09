# 本脚本分析了强化学习（RL）代理的回合时间
# 通过分析这些“反应时间”，可以将其与人类行为数据进行比较

# 导入必要的库
import numpy as np
import time
import random
import bson
from .anal_utils import plan_epoch,seeds,greedy_actions,datadir
from ..walls_train import recover_model,build_environment
from ..src.model import ModularModel,forward_modular
from ..src.a2c import run_episode
from ..src.walls_baselines import dist_to_rew
from ..src.walls import state_ind_from_state
from ..src.loss_hyperparameters import LossHyperparameters
# 打印分析开始信息
print("analysing the timings of rollouts in the RL agent")

# 设置超参数（不计算损失）
loss_hp = LossHyperparameters(0, 0, 0, 0) 
epoch = plan_epoch  # 测试的epoch

# 遍历每个随机种子，加载独立训练的模型
for seed in seeds:
    # 加载模型参数
    fname = f"N100_T50_Lplan8_seed{seed}_{epoch}"
    print(f"loading {fname}")
    network, opt, store, hps, policy, prediction = recover_model(f"../models/{fname}")

    # 初始化模型和环境
    Larena = hps["Larena"]
    model_properties, wall_environment, model_eval = build_environment(
        Larena, hps["Nhidden"], hps["T"], Lplan=hps["Lplan"], greedy_actions=greedy_actions
    )
    m = ModularModel(model_properties, network, policy, prediction, forward_modular)

    # 运行多个回合（episodes）
    random.seed(1)  # 设置随机种子，保证可复现
    batch_size = 50000  # 每个batch的环境数
    tic = time.time()  # 记录开始时间
    L, ys, rews, as_, world_states, hs = run_episode(
        m, wall_environment, loss_hp, batch=batch_size, calc_loss=False
    )

    # 提取需要的数据
    states = np.concatenate([ws.agent_state for ws in world_states], axis=2)  # 按时间维度拼接状态
    wall_loc = world_states[0].environment_state.wall_loc  # 墙壁位置
    ps = world_states[0].environment_state.reward_location  # 奖励位置
    Tmax, Nstates = as_.shape[1], Larena ** 2  # 提取一些维度信息
    rew_locs = np.reshape(ps, (Nstates, batch_size, 1)) * np.ones((1, 1, Tmax))  # 每个时间点的奖励位置
    # calculate and print average reward and time
    average_reward = np.sum(rews > 0.5) / batch_size
    print(f"average reward: {average_reward}  time: {time.time() - tic}")

    # 计算规划的步数
    plan_steps = np.zeros((batch_size, Tmax))
    for t in range(Tmax - 1):
        plan_steps[:, t] = np.sum(world_states[t + 1].planning_state.plan_cache > 0.5, axis=1)

    # 提取一些实验信息
    trial_ts = np.zeros((batch_size, Tmax))  # 网络迭代时间
    trial_ids = np.zeros((batch_size, Tmax))  # 实验编号
    trial_anums = np.zeros((batch_size, Tmax))  # 动作编号（不计算回合）
    for b in range(batch_size):  # 遍历每个回合
        Nrew = np.sum(rews[b, :] > 0.5)  # 获得的奖励数量
        sortrew = np.argsort(-rews[b, :])  # 排序后的奖励索引
        rewts = sortrew[:Nrew]  # 获得奖励的时间点
        diffs = np.concatenate([rewts, [Tmax + 1]]) - np.concatenate([[0], rewts])  # 每个实验的持续时间
        trial_ids[b, :] = np.concatenate([np.ones(diffs[i]) * i for i in range(Nrew + 1)])[:Tmax]  # 实验编号
        trial_ts[b, :] = np.concatenate([np.arange(1, diffs[i] + 1) for i in range(Nrew + 1)])[:Tmax]  # 每个实验内的时间

        finished = np.where(as_[b, :] == 0)[0]  # 回合结束的时间点
        # 将结束的步骤归零
        trial_ids[b, finished] = 0
        trial_ts[b, finished] = 0
        plan_steps[b, finished] = 0

        # 提取每次迭代的动作编号
        ep_as = as_[b, :]
        for id in range(Nrew + 1):  # 遍历每个实验
            inds = np.where(trial_ids[b, :] == id)[0]  # 当前实验的索引
            trial_as = ep_as[inds]  # 当前实验的动作
            anums = np.zeros(len(inds), dtype=int)  # 动作编号列表
            anum = 1  # 从第一个动作开始
            for a in range(1, len(inds)):  # 遍历所有网络迭代
                anums[a] = anum  # 存储当前动作编号
                if trial_as[a] <= 4.5:  # 如果不是回合动作
                    anum += 1  # 增加动作编号
            trial_anums[b, inds] = anums  # 存储所有动作编号

    ## 按实验编号分析性能

    Rmin = 4  # 只考虑奖励大于等于 Rmin 的实验（控制奖励与每个实验的步骤数之间的相关性）
    inds = np.where(np.sum(rews, axis=1) >= Rmin)[0]  # 奖励大于等于 Rmin 的回合
    perfs = np.concatenate([[trial_anums[b, trial_ids[b, :] == t][-1] for t in range(1, Rmin + 1)] for b in inds]).T  # 每个实验编号的性能

    # 计算最优基准
    mean_dists = np.zeros(batch_size)  # 平均目标距离
    for b in range(batch_size):
        dists = dist_to_rew(ps[:, b:b], wall_loc[:, :, b:b], Larena)  # 每个状态的目标距离
        mean_dists[b] = np.sum(dists) / (Nstates - 1)  # 非目标状态的平均距离
    μ, s = np.mean(perfs, axis=0), np.std(perfs, axis=0) / np.sqrt(batch_size)  # 计算总结统计数据
    data = [Rmin, μ, s, np.mean(mean_dists)]
    with open(f"{datadir}/model_by_trial{seed}.bson", "wb") as f:
        bson.dump(data, f)  # 存储数据

    ## 按难度分析规划

    trials = 15
    new_RTs = np.full((trials, batch_size, hps["T"]), np.nan)
    new_alt_RTs = np.full((trials, batch_size, hps["T"]), np.nan)
    new_dists = np.full((trials, batch_size), np.nan)
    for b in range(batch_size):
        rew = rews[b, :]  # 当前回合的奖励
        min_dists = dist_to_rew(ps[:, b:b], wall_loc[:, :, b:b], Larena)  # 每个状态到目标的最小距离
        for trial in range(1, trials):  # 遍历每个实验
            if np.sum(rew > 0.5) > (trial - 0.5):  # 实验完成
                inds = np.where((trial_ids[b, :] == trial) & (trial_ts[b, :] > 1.5))[0]  # 当前实验的时间点

                anums = trial_anums[b, inds]
                RTs = [np.sum(anums == anum) for anum in range(1, anums[-1] + 1)]

                plan_nums = plan_steps[b, inds]
                alt_RTs = [np.sum(plan_nums[anums == anum]) for anum in range(1, anums[-1] + 1)]  # 模拟步骤数
                new_alt_RTs[trial, b, :len(alt_RTs)] = alt_RTs  # 存储模拟步骤数
                
                for anum in range(1, anums[-1] + 1):
                    ainds = [i for i, x in enumerate(anums) if x == anum]
                    if len(ainds) > 1.5:
                        assert all(plan_nums[ainds[i]] > 0.5 for i in range(len(ainds) - 1))  # should all have non-zero plans

                new_RTs[trial, b, :len(RTs)] = RTs  # 存储反应时间
                state = states[:, b, inds[0]]  # 初始状态
                new_dists[trial, b] = min_dists[int(state[0]), int(state[1])]  # 存储目标距离

    dists = np.arange(1, 9)
    dats = [new_RTs[new_dists == dist, :] for dist in dists]
    data = [dists, dats]
    with open(f"{datadir}model_RT_by_complexity{seed}_{epoch}.bson", "wb") as f:
        bson.dump(data, f)
    
    alt_dats = [new_alt_RTs[new_dists == dist, :] for dist in dists]
    data = [dists, alt_dats]
    with open(f"{datadir}model_RT_by_complexity_bystep{seed}_{epoch}.bson", "wb") as f:
        bson.dump(data, f)

    ## 探索分析

    RTs = np.full_like(rews, np.nan)
    unique_states = np.full_like(rews, np.nan)  # 每次动作时已经访问过的状态数
    for b in range(batch_size):
        inds = np.where(trial_ids[b, :] == 1)[0]
        anums = trial_anums[b, inds].astype(int)
        if np.sum(rews[b, :]) == 0:
            tmax = np.sum(as_[b, :] > 0.5)
        else:
            tmax = np.where(rews[b, :] == 1)[0][0]  # 找到第一个奖励的时间点
        visited = np.zeros(16, dtype=bool)  # 记录哪些状态已经访问过
        for anum in np.unique(anums):
            state = states[:, b, np.where(anums == anum)[0][0]]
            visited[int(state_ind_from_state(Larena, state)[0])] = True
            unique_states[b, anum + 1] = np.sum(visited)
            RTs[b, anum + 1] = np.sum(anums == anum)

    data = [RTs, unique_states]
    with open(f"{datadir}model_unique_states_{seed}_{epoch}.bson", "wb") as f:
        bson.dump(data, f)

    ## 按独特状态进行奖励位置的解码

    unums = np.arange(1, 16)
    dec_perfs = np.zeros(len(unums))
    for unum in unums:
        inds = np.where(unique_states == unum)[0]
        ahot = np.zeros((5, len(inds)), dtype=np.float32)
        for i, ind in enumerate(inds):
            ahot[int(as_[ind]), i] = 1.0
        X = np.concatenate([hs[:, inds], ahot], axis=0)  # Nhidden x batch x T -> Nhidden x iters
        Y = rew_locs[:, inds]
        Yhat = m.prediction(X)[16:32, :]
        Yhat = np.exp(Yhat - np.log(np.sum(np.exp(Yhat), axis=0, keepdims=True)))  # softmax over states
        perf = np.sum(Yhat * Y) / Y.shape[1]
        dec_perfs[unum - 1] = perf

    data = [unums, dec_perfs]
    # save results
    filename = f"{datadir}model_exploration_predictions_{seed}_{epoch}.bson"
    with open(filename, 'wb') as f:
        f.write(bson.BSON.encode({'data': data}))
