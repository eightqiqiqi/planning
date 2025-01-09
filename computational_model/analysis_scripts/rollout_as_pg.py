import numpy as np
import os
import pickle
import torch
import torch.autograd as autograd
import scipy
import bson
import jax.numpy as jnp
from torch import nn, optim
from scipy.misc import derivative
from .anal_utils import seeds,epoch,loaddir,greedy_actions,build_planner,datadir
from ..walls_train import recover_model,build_environment
from ..src.model import ModularModel
from ..src.a2c import forward_modular,GRUind
from ..src.planning import build_planner
from ..src.walls_build import update_agent_state
from ..src.walls import onehot_from_state

print("Comparing hidden state updates to policy gradients")

# Load some data
res_dict = {}  # Dictionary for storing results
# Iterate through RL agent seeds
for seed in seeds:
    res_dict[seed] = {}

    filename = f"N100_T50_Lplan8_seed{seed}_{epoch}"  # Model to load
    print(f"Loading model: {filename}")
    # Load model parameters
    network, opt, store, hps, policy, prediction = recover_model(os.path.join(loaddir, filename))

    # Construct environment and RL agent
    Larena = hps["Larena"]
    model_properties, wall_environment, model_eval = build_environment(
        Larena, hps["Nhidden"], hps["T"], Lplan=hps["Lplan"], greedy_actions=greedy_actions
    )
    m = ModularModel(model_properties, network, policy, prediction, forward_modular)

    # Define some useful parameters
    Nstates = Larena**2
    Naction = wall_environment.dimensions.Naction
    Nin_base = Naction + 1 + 1 + Nstates + 2 * Nstates
    Nout, Nhidden, Nin = m.model_properties.Nout, m.model_properties.Nhidden, m.model_properties.Nin
    Lplan = model_properties.Lplan
    ed = wall_environment.dimensions
    Nstates, Naction, T = ed.Nstates, ed.Naction, ed.T

    # Instantiate arrays for storing results
    all_sim_as, all_sim_a2s = [], []
    all_jacs, all_jacs_shift, all_jacs_shift2 = [], [], []
    all_gs, all_gs2 = [], []
    all_pis = []
    full_inps = []
    meangv = []

    np.random.seed(2)  # Set random seed for reproducibility
    for i_mode, mode in enumerate(["R_tau", "test"]): 
        if mode == "R_tau":
            print("Estimating R_tau")
        else:
            print("Computing policy gradients")
        all_rews = []
        agent_output = None

        # run a handful of steps
        batch = 1002 #number of episodes to consider

        # Initialize environment
        world_state, agent_input = wall_environment.initialize(
            np.zeros(2), np.zeros(2), batch, m.model_properties
        )
        agent_state = world_state.agent_state
        h_rnn = m.network[GRUind].cell.state0 + np.zeros((Nhidden, batch), dtype=np.float32)  # expand hidden state
        rew = np.zeros(batch)

        exploit, just_planned = np.zeros(batch, dtype=bool), np.zeros(batch, dtype=bool)
        tmax = 50
        planner, initial_plan_state = build_planner(Lplan, Larena)
        all_ts, all_as = np.zeros((batch, tmax)), np.zeros((batch, tmax))

        for t in range(1, tmax + 1):
            if t % 20 == 0:
                print(f"iteration {t} of {tmax}")

            # copy over local variables
            agent_input = agent_input
            world_state = world_state
            rew = rew
            all_ts[:, t] = world_state.environment_state.time[:]  # store environment time

            # Identify planning episodes during exploitation
            plan_bs = np.where(exploit & just_planned)[0]
            Nps = len(plan_bs)
            if mode == "test" and Nps > 0:
                logπ = agent_output[:4, plan_bs] 
                all_pis.append(np.exp((logπ - scipy.special.logsumexp(logπ, axis=0)).astype(np.float64)).T)

                newgs, newgs2 = [np.zeros((Nps, Nhidden, 4)) for _ in range(2)]
                sim_as, sim_a2s = [np.full(Nps, np.nan) for _ in range(2)]
                jacs, jacs_shift, jacs_shift2 = [np.zeros((Nps, Nhidden)) for _ in range(3)]

                for ib, b in enumerate(plan_bs):
                    # Simulated actions
                    sim_a = np.argmax(agent_input[Nin_base + 1:Nin_base + 4, b])
                    sim_as[ib] = sim_a
                    sim_a2 = np.argmax(agent_input[Nin_base + 5:Nin_base + 8, b])
                    shift_a = (sim_a % 4) + 1
                    shift_a2 = (sim_a2 % 4) + 1
                    assert agent_input[Nin_base + sim_a, b] == 1

                    pert = np.zeros(Nin, dtype=np.float32)
                    pert[Nin_base + 1:] = meangv

                    # Delta vectors for action changes
                    shifta = np.zeros(Nin, dtype=np.float32)
                    shifta[Nin_base + sim_a] = -1.0
                    shifta[Nin_base + shift_a] = 1.0

                    shifta2 = np.zeros(Nin, dtype=np.float32)
                    shifta2[Nin_base + 4 + sim_a2] = -1.0
                    shifta2[Nin_base + 4 + shift_a2] = 1.0

                    # Function for RNN update step
                    def fh(x, shift=np.zeros(Nin, dtype=np.float32)):
                        return m.network[GRUind].cell(h_rnn[:, b:b + 1], agent_input[:, b:b + 1] + shift + pert * x)[0]

                    jac = autograd.functional.jacobian(fh, 0.0)[0]
                    jacs[ib, :] = jac

                    fh_shift = lambda x: fh(x, shift=shifta)
                    jacs_shift[ib, :] = autograd.functional.jacobian(fh_shift, 0.0)[0]
                    fh_shift2 = lambda x: fh(x, shift=shifta2)
                    jacs_shift2[ib, :] = autograd.functional.jacobian(fh_shift2, 0.0)[0]

                    # Function mapping hidden state to policy
                    def fp(x, a):
                        logπ = m.policy(x)[:4]
                        return logπ[a] - np.logaddexp.reduce(logπ)

                    for ia in range(4):
                        fa = lambda x: fp(x, ia)
                        gs = autograd.grad(fa, h_rnn[:, b:b+1])
                        newgs[ib, :, ia] = gs

                    if agent_input[Nin_base + 4 + sim_a2, b] == 1:  # 滚动包含至少两个动作
                        sim_a2s[ib] = sim_a2  # 存储模拟的动作
                        # 下一次迭代的输入（当前是外部循环的迭代）
                        nextinp = np.zeros((Nin,), dtype=np.float32)
                        nextinp[:, None] = agent_input[:, b:b + 1]
                        nextinp[Nin_base + 1:] = 0.0  # 假设我们采取了一个动作，而不是滚动
                        nextinp[Naction + 2, None] += (1.0 - 0.3)  # 增加时间（假设我们执行了一个动作）
                        nextinp[0:5] = 0.0
                        nextinp[sim_a] = 1.0  # 覆盖为想象的动作 (ahat_1)
                        newstate = update_agent_state(world_state.agent_state[:, b:b + 1], nextinp[0:5, None], Larena)  # 计算所处的状态
                        # 想象如果执行了指定动作并移动到对应状态后的输入
                        nextinp[(Naction + 3):(Naction + 2 + Nstates)] = onehot_from_state(Larena, newstate).astype(np.float32)

                        def f2(x, a):
                            newh = m.network[GRUind].cell(x, nextinp)[0]  # 假设我们采取模拟动作而不是规划
                            logπ2 = m.policy(newh)[0:4]  # 从此状态的对数策略
                            logπa2 = logπ2[a] - np.log(np.sum(np.exp(logπ2)))  # 计算从此状态选择 a2 的概率
                            return logπa2

                        for ia in range(1, 5):  # 对每个动作进行处理
                            fa = lambda x: f2(x, ia)  # 构造函数以输出该动作的策略
                            gs = autograd.grad(fa, h_rnn[:, b:b + 1])  # 计算 log π(a) 对当前隐藏状态的梯度
                            newgs2[ib, :, ia - 1] = gs  # 存储梯度
            
            # Store results
            all_sim_as.append(sim_as)
            all_sim_a2s.append(sim_a2s)
            all_jacs.append(jacs)
            all_jacs_shift.append(jacs_shift)
            all_jacs_shift2.append(jacs_shift2)

        # 存储代理输入
        full_inps.append(agent_input[Nin_base + 1:])
        # 执行 RNN 步骤
        h_rnn, agent_output, a = m.forward(m, ed, agent_input, h_rnn)
        # 活跃的回合
        active = world_state.environment_state.time < (T + 1 - 1e-2)
        # 如果 teleport 条件满足，则不进行滚动
        teleport = np.logical_not(np.logical_and.reduce([active, exploit, a > 4.5, rew < 0.5]))
        # 存储实际的物理动作
        all_as[:, t - 1] = a

        # 执行滚动并返回状态
        exploit[rew > 0.5] = True  # 如果找到奖励，则进入利用阶段
        just_planned = np.zeros(batch, dtype=bool)  # 记录是否刚刚执行了滚动
        just_planned[np.logical_not(teleport)] = True  # 记录刚刚执行滚动的状态

        # 在环境中迈出一步
        old_world_state = world_state  # 保存当前状态
        rew, agent_input, world_state, predictions = wall_environment.step(
            agent_output, a, world_state, wall_environment.dimensions, m.model_properties, m, h_rnn
        )
        # 假象的滚动状态
        plan_states = world_state.planning_state.plan_cache
        # 如果回合完成，则将奖励置为 0
        rew[np.logical_not(active)] = 0.0
        # 存储奖励
        all_rews.append(rew)

    if mode == "R_tau":  # 如果我们处于估计 R_tau 的第一阶段
        # 合并所有奖励
        rews = np.concatenate(all_rews, axis=0)
        # 合并所有滚动反馈
        inps = np.concatenate(full_inps, axis=2)
        # 计算滚动反馈的总和（用于检查是否存在）
        inp_sums = np.sum(inps, axis=0)[0, :, :]
        # 初始化到下一个奖励的时间矩阵
        t_to_rew = np.full_like(rews, np.nan, dtype=np.float32)
        for b in range(batch):  # 对每个回合进行处理
            if np.sum(rews[:, b]) > 1.5:  # 如果代理完成了至少两个试验
                ks = np.where(rews[:, b] == 1)[0]  # 找到获得奖励的迭代次数
                k1, k2 = ks[0] + 2, ks[-1]
                ts = all_ts[b, ks]  # 在回合中经过的时间
                for k in range(k1, k2 + 1):  # 遍历从第二次试验开始到最后一次奖励的每次迭代
                    dts = ts - all_ts[b, k]  # 当前时间到所有奖励时间的差
                    dts_valid = dts[dts >= 0]  # 筛选有效时间差
                    if len(dts_valid) > 0:
                        t_to_rew[k, b] = dts_valid[0]  # 记录到下一个奖励的时间

        # 找到执行了滚动且后来获得奖励的迭代
        inds = np.where((~np.isnan(t_to_rew.T)) & (inp_sums > 0.5))
        # 滚动反馈在这些迭代中的值
        X = inps[:, inds[0]]
        # 到奖励的时间
        y = t_to_rew.T[inds]
        y = -y  # 取负值（希望时间越短越好）
        # 计算回归系数
        beta = np.linalg.inv(X @ X.T) @ X @ y
        # 将第一个动作的偏移归零（这是截距）
        beta[:4] -= np.mean(beta[:4])
        # 归一化
        meangv = beta / np.sqrt(np.sum(beta**2))

    else:  # 如果我们处于测试阶段
        # 合并结果数组并存储结果
        arrs = [all_sim_as, all_sim_a2s, all_jacs, all_jacs_shift, all_jacs_shift2, all_gs, all_gs2, all_pis]
        labels = ["sim_as", "sim_a2s", "jacs", "jacs_shift", "jacs_shift2", "sim_gs", "sim_gs2", "all_pis"]
        cat_arrs = [np.concatenate(arr, axis=0) for arr in arrs]
        for arr_ind, label in enumerate(labels):
            res_dict[seed][label] = cat_arrs[arr_ind]

# write our results to a file
with open(datadir + "planning_as_pg.bson", 'wb') as f:
    f.write(bson.BSON.encode({'res_dict': res_dict}))

