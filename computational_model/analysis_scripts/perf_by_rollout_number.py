# 该脚本分析模型性能如何依赖于回滚次数
# 这帮助我们调查回滚是否能够改善策略

# 导入必要的库
import numpy as np
import random
import os
import bson
from .anal_utils import datadir,seeds,N,Lplan,epoch
from ..walls_train import recover_model,build_environment
from ..src.model import ModularModel
from ..src.walls_baselines import dist_to_rew
# 假设 recover_model 和其他相关函数已经被定义和导入

run_default_analyses = False
try:
    print("running default analyses: ", run_default_analyses)
except Exception as e:
    run_default_analyses = True


def run_perf_by_rollout_number(seeds, N, Lplan, epoch, prefix="", model_prefix=""):
    """
    根据回滚次数分析试验2的性能（以达到目标的步数为指标）
    """
    print("量化试验2性能，依据回滚次数")

    res_dict = {}  # 用于存储结果的字典

    for seed in seeds:  # 遍历随机种子
        res_dict[seed] = {}  # 当前种子的结果
        filename = f"{model_prefix}N{N}_T50_Lplan{Lplan}_seed{seed}_{epoch}"  # 加载模型的文件名
        print(f"\nloading {filename}")                                                                                                           
        # 加载模型参数
        network, opt, store, hps, policy, prediction = recover_model(filename) 
        
        Larena = hps["Larena"]  # Arena的大小
        # 构建环境
        model_properties, wall_environment, model_eval = build_environment(
            Larena, hps["Nhidden"], hps["T"], Lplan=hps["Lplan"], greedy_actions=True
        )
        m = ModularModel(model_properties, network, policy, prediction)  # 构建模型

        # 设置一些参数
        batch = 1  # 一次处理一个batch
        ed = wall_environment.dimensions
        Nstates, Naction = ed.Nstates, ed.Naction
        Nin_base = Naction + 1 + 1 + Nstates + 2 * Nstates  # '物理'输入维度
        Nhidden = m.model_properties.Nhidden
        tmax = 50  # 最大时间步数
        Lplan = model_properties.Lplan
        nreps = 5000  # 随机环境的数量
        nplans = np.arange(0, 16)  # 强制的计划数量
        dts = np.full((2, nreps, len(nplans)), np.nan)  # 存储达到目标的步数
        policies = np.full((2, nreps, len(nplans), 10, 5), np.nan)  # 存储策略
        mindists = np.zeros((nreps, len(nplans)))  # 存储最小距离
        
        for ictrl in [1, 2]:  # 是否使用规划输入（0 或 1）
            for nrep in range(nreps):  # 遍历每次重复实验
                if nrep % 1000 == 0:
                    print(nrep)                
                for iplan, nplan in enumerate(nplans):  # 遍历每个回滚数
                    random.seed(nrep)  # 设置随机种子确保环境一致
                    world_state, agent_input = wall_environment.initialize(
                        np.zeros(2), np.zeros(2), batch, m.model_properties
                    )  # 初始化环境
                    agent_state = world_state.agent_state  # 初始化代理位置
                    h_rnn = np.zeros((Nhidden, batch), dtype=np.float32)  # 扩展隐藏状态
                    exploit = np.zeros(batch, dtype=bool)  # 跟踪探索与利用
                    rew = np.zeros(batch)  # 跟踪奖励
                    if iplan == 1:
                        ps, ws = world_state.environment_state.reward_location, world_state.environment_state.wall_loc
                        dists = dist_to_rew(ps, ws, Larena)  # 计算距离目标的距离

                    tot_n = nplan  # 回滚次数
                    t = 0  # 时间步
                    finished = False  # 是否结束
                    nact = 0  # 物理动作次数                    
                    while not finished:  # 直到结束
                        t += 1  # 更新步数
                        agent_input, world_state, rew = agent_input, world_state, rew
                        if ictrl == 2 and exploit[0]:  # 如果是利用阶段且控制
                            agent_input[Nin_base+1:] = 0  # 如果是控制阶段，清空输入
                        h_rnn, agent_output, a = m.forward(m, ed, agent_input, h_rnn)  # RNN步骤

                        plan = False  # 是否进行回滚
                        if exploit[0] and tot_n > 0.5:  # 如果是利用阶段且回滚次数大于0
                            plan = True
                            tot_n -= 1  # 执行回滚，减少剩余回滚次数
                            if tot_n == 0:  # 如果回滚结束
                                state = world_state.agent_state[:]  # 当前状态
                                mindists[nrep, iplan] = dists[state[0], state[1]]  # 存储距离目标的距离

                        if plan:  # 执行回滚
                            a[0] = 5  # 执行回滚
                        elif exploit[0]:  # 利用阶段
                            nact += 1  # 增加动作次数
                            a[0] = np.argmax(agent_output[:4, 0])  # 贪心选择动作
                            if nact <= 10:
                                policies[ictrl-1, nrep, iplan, nact-1, :] = agent_output[:5, 0]  # 存储策略
                        else:  # 探索阶段
                            a[0] = np.argmax(agent_output[:5, 0])  # 贪心选择动作

                        exploit[rew[:] > 0.5] = True  # 如果奖励大于0.5，则进入利用阶段
                        # 传递动作到环境
                        rew, agent_input, world_state, predictions = wall_environment.step(
                            agent_output, a, world_state, wall_environment.dimensions, m.model_properties, m, h_rnn
                        )

                        if rew[0] > 0.5:  # 如果找到了奖励
                            if exploit[0]:  # 如果之前已找到奖励
                                finished = True  # 结束实验
                                dts[ictrl-1, nrep, iplan] = t - t1 - 1 - nplan  # 存储达到目标所需步数
                                assert nact == (t - t1 - 1 - nplan)
                                assert nact >= mindists[nrep, iplan]
                            else:  # 第一次找到奖励
                                t1 = t  # 重置时间
                        if t > tmax:  # 达到最大步数限制
                            finished = True

        # 存储当前种子的结果
        res_dict[seed]["dts"] = dts
        res_dict[seed]["mindists"] = mindists
        res_dict[seed]["nplans"] = nplans
        res_dict[seed]["policies"] = policies

    # save our data
    savename = f"{prefix}N{N}_Lplan{Lplan}"
    with open(datadir + f"perf_by_n_{savename}.bson", 'wb') as f:
        f.write(bson.BSON.encode({'res_dict': res_dict}))

# run_default_analyses is a global parameter in anal_utils.jl
if run_default_analyses:
    run_perf_by_rollout_number(seeds=seeds, N=N, Lplan=Lplan, epoch=epoch)

