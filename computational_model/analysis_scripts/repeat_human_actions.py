import numpy as np
import pickle
import bson
from scipy.stats import sem
from .anal_utils import datadir,calc_post_mean,loaddir,greedy_actions,seeds,N,Lplan,epoch
from ..walls_train import recover_model,build_environment
from ..src.model import ModularModel
from ..src.a2c import forward_modular,GRUind,zeropad_data
from ..src.walls_baselines import dist_to_rew

run_default_analyses = False
try:
    print("running default analyses: ", run_default_analyses)
except Exception as e:
    run_default_analyses = True


def repeat_human_actions(seeds, N, Lplan, epoch, prefix="", model_prefix=""):
    #print("分析重复人类动作时的策略概率")
    print("analysing rollout probabilities when repeating human actions")

    # 载入人类数据
    with open(f"{datadir}/human_RT_and_rews_play.bson", "rb") as file:
        data_play = pickle.load(file)
    with open(f"{datadir}/human_RT_and_rews_follow.bson", "rb") as file:
        data_follow = pickle.load(file)
    # 计算每个参与者的平均RT
    mean_RTs = [np.nanmean(data["all_RTs"]) for data in [data_play, data_follow]]
    keep = np.where(mean_RTs[1] < 690)[0]  # 仅保留RT小于690ms的参与者
    Nkeep = len(keep)  # 保留的参与者数量
    
    # load all data from these participants
    with open(f"{datadir}/human_all_data_play.bson", 'rb') as f:
        data = bson.BSON.decode(f.read())

    all_states, all_ps, all_as, all_wall_loc, all_rews, all_RTs, all_trial_nums, all_trial_time = [
        dat[keep] for dat in data
    ]

    # load response time prior parameters
    with open(f"{datadir}/guided_lognormal_params_delta.bson", 'rb') as f:
        params = bson.BSON.decode(f.read())

    lognormal_params = {key: params[key][keep, :] for key in ["initial", "later"]}

    # 初始化结果数组
    all_as_p, all_ys_p, all_pplans_p, all_Nplans_p = [], [], [], []
    all_dists_to_rew_p, all_new_states_p = [], []
    all_RTs_p = []

    # 对每个参与者进行循环
    for i in range(len(keep)):  # 对每个参与者进行分析
        print(f"user: {i+1}")
        
        # 载入当前参与者的数据
        states = all_states[i]  # 位置数据
        ps = all_ps[i]  # 奖励位置
        as_ = all_as[i]  # 采取的动作
        wall_loc = all_wall_loc[i]  # 墙的位置
        rews = all_rews[i]  # 奖励
        RTs = np.copy(all_RTs[i])  # 响应时间数据
        trial_ts = all_trial_time[i]  # 试验中的迭代次数
        batch_size = as_.shape[0]  # 每个批次的大小

        ### compute thinking times from prior ###
        initial, later = [lognormal_params[key][i, :] for key in ["initial", "later"]]

        # posterior mean for initial action
        def initial_post_mean(r):
            return calc_post_mean(r, muhat=initial[0], sighat=initial[1], deltahat=initial[2])
        # posterior mean for later actions
        def later_post_mean(r):
            return calc_post_mean(r, muhat=later[0], sighat=later[1], deltahat=later[2])
        RTs[trial_ts == 1] = np.array([initial_post_mean(r) for r in RTs[trial_ts == 1]])  # use different parameters for first action
        RTs[trial_ts != 1] = np.array([later_post_mean(r) for r in RTs[trial_ts != 1]])  # posterior mean

        all_RTs_p.append(RTs)  # store RTs
        
        # now run all RL agents through this participant
        as_p, ys_p, pplans_p, Nplans_p = [], [], [], [] #arrays for storing data
        dists_to_rew_p, new_states_p = [], []
        for seed in seeds:  # 对每个模型进行循环
            fname = f"{model_prefix}N{N}_T50_Lplan{Lplan}_seed{seed}_{epoch}"  # 模型文件名
            print(f"loading {fname}")
            network, opt, store, hps, policy, prediction = recover_model(f"{loaddir}/{fname}")  # 载入模型参数

            # 构建RL环境并实例化代理
            model_properties, wall_environment, model_eval = build_environment(
                hps["Larena"], hps["Nhidden"], hps["T"], Lplan=hps["Lplan"], greedy_actions=greedy_actions
            )
            m = ModularModel(model_properties, network, policy, prediction, forward_modular)

            # 提取一些有用的参数
            environment = wall_environment
            ed = environment.dimensions
            Nout, Nhidden, Nstates, T = m.model_properties.Nout, m.model_properties.Nhidden, ed.Nstates, ed.T

            for rep in range(21):  # 重复多次，因为rollout是随机的
                np.random.seed(rep)

                # 初始化环境
                world_state, agent_input = environment.initialize(ps, np.int32(states[:, :, 1]), batch_size, m.model_properties)
                agent_state = world_state.agent_state  # 代理位置
                world_state.environment_state.wall_loc = np.copy(wall_loc)  # 墙的位置

                h_rnn = m.network[GRUind].cell.state0 + np.float32(np.zeros((Nhidden, batch_size)))  # 扩展隐藏状态

                new_ys, new_as = [], []
                ts = np.zeros(batch_size, dtype=int)  # 当前时间步
                rew, rew_prev = np.zeros(batch_size), np.zeros(batch_size, dtype=bool)  # 当前奖励
                p_plans = np.full((batch_size, 51), np.nan)  # 执行rollout的概率
                N_plans = np.zeros((batch_size, 51))  # 执行的rollout数量
                dists_to_rew = np.zeros((batch_size, 51))  # 到目标的距离
                new_states = np.ones((2, batch_size, 100))  # 代理的状态
                new_rews = np.zeros((batch_size, 51))  # 每次迭代的奖励
                a5s = np.zeros(batch_size, dtype=bool)  # 是否执行了rollout
                active = np.ones(batch_size, dtype=bool)  # 该episode是否完成
                nplan = 0
                while np.any(world_state.environment_state.time < (T + 1 - 1e-2)):  # 直到人类或代理的时间结束
                    h_rnn, agent_output, a = m.forward(m, ed, agent_input, h_rnn)  # RNN步骤

                    for b in range(batch_size):  # 对每个episode
                        if rew[b] < 0.5:  # 如果没有得到奖励
                            if not a5s[b] and active[b]: 
                                ts[b] += 1  # 增加时间步
                            a[0, b] = max(1, as_[b, ts[b]])  # 用人类动作替换

                            if as_[b, ts[b]] > 0.5 and not a5s[b] and active[b]:  # 如果执行了动作
                                p_plans[b, ts[b]] = np.exp(agent_output[5, b])  # 计算rollout的概率
                                new_states[:, b, ts[b]] = world_state.agent_state[:, b]  # 当前状态
                                dists = dist_to_rew(ps[:, b], wall_loc[:, :, b], hps["Larena"])  # 到目标的距离
                                dists_to_rew[b, ts[b]] = dists[int(new_states[0, b, ts[b]]), int(new_states[1, b, ts[b]])]
                                
                            # 如果满足条件则执行rollout
                            if ts[b] > 1.5 and np.random.rand() < np.exp(agent_output[5, b]):
                                nplan = 1  # 执行rollout
                            else:
                                nplan = 0  # 否则执行动作
                        else:  # 如果已经得到奖励
                            nplan = 0  # 不再进行计划

                        if nplan > 0.5:  # 如果执行了rollout
                            a[0, b] = 5  # 设置动作为'rollout'
                            a5s[b] = True  # 标记为执行了rollout
                            if active[b]:
                                N_plans[b, ts[b]] += 1  # 增加rollout计数
                        else:
                            a5s[b] = False  # 执行了物理动作

                    active = np.array([world_state.environment_state.time[b] < (T + 1 - 1e-2) and as_[b, ts[b]] > 0.5 for b in range(batch_size)])  # 激活的episode
                    rew_prev = (rew > 0.5)  # 是否在之前的步骤中得到了奖励

                    # 更新环境状态
                    rew, agent_input, world_state, predictions = environment.step(
                        agent_output, a, world_state, environment.dimensions, m.model_properties, m, h_rnn
                    )

                    for b in np.where(rew_prev)[0]:  # 对奖励找到的episode
                        world_state.agent_state[:, b] = states[:, b, ts[b] + 1]  # 将代理传送到正确的位置

                    # 生成输入
                    ahot = np.zeros((5, batch_size), dtype=np.float32)
                    for b in range(batch_size):
                        ahot[int(a[0, b]), b] = 1.0
                    agent_input = np.concatenate([np.concatenate([ahot, predictions], axis=0), np.zeros((Nhidden, batch_size))], axis=0)
                    
                    rew, agent_input, a = zeropad_data(rew, agent_input, a, active)  # remove data from finished episodes
                    new_ys.append(agent_output)  # store new outputs
                    new_as.append(a)  # store new actions

                # store various pieces of data from this repetition
                ys_p.append(np.concatenate(new_ys, axis=2))
                as_p.append(np.vstack(new_as).T)
                dists_to_rew_p.append(dists_to_rew)
                pplans_p.append(p_plans)
                new_states_p.append(new_states)
                Nplans_p.append(N_plans)

        # store data from this participant
        all_ys_p.append(ys_p)
        all_as_p.append(as_p)
        all_pplans_p.append(pplans_p)
        all_Nplans_p.append(Nplans_p)
        all_new_states_p.append(new_states_p)
        all_dists_to_rew_p.append(dists_to_rew_p)

    # perform some analyses with the data collected above
    for trial_type in ["explore", "exploit"]:  # consider exploration and exploitation trials separately
        trialstr = "_explore" if trial_type == "explore" else ""

        #### there is an issue with user i = 21, batch = 12 -- it seems like the walls have been loaded wrong.

        # initialize some data arrays
        alldRTs, alldplans, allddist, alldts = [], [], [], []
        allres, allsims, allsims_s = [], np.zeros((len(all_pplans_p), 3)), np.zeros((len(all_pplans_p), 3))
        p_plans_by_u, RTs_by_u, dists_by_u, steps_by_u, N_plans_by_u, anums_by_u, trialnums_by_u = [], [], [], [], [], [], []

    for i in range(len(all_pplans_p)):  # iterate through users
        # print(i)
        store = True

        p_plans = np.mean(np.concatenate(all_pplans_p[i], axis=2), axis=2)[:, :, 0]  # rollout probabilities
        N_plans = np.mean(np.concatenate(all_Nplans_p[i], axis=2), axis=2)[:, :, 0]  # number of rollouts

        # extract data from this user
        as_, trial_ts, states, rews, trial_nums = all_as[i], all_trial_time[i], all_states[i], all_rews[i], all_trial_nums[i]
        dists_to_rew, new_states, RTs = all_dists_to_rew_p[i][0], all_new_states_p[i][0], all_RTs_p[i]

        dRTs, dplans, ddist, dts, new_trial_nums, new_anums, Nplan_dat = [], [], [], [], [], [], []
        for b in range(as_.shape[0]):
            # print(b)
            tmin = 2  # ignore very first action
            tmax = min(np.sum(as_[b, :] > 0.5), np.sum(p_plans[b, :] > 0.0))  # last action in episode
            if (tmax > tmin + 5) and (np.sum(rews[b, :]) > 0.5):
                if not np.all(new_states[:, b, tmin:tmax] == states[:, b, tmin:tmax]):  # check that we followed correct states
                    store = False
                elif not np.all(all_new_states_p[i][0][:, b, tmin:tmax] == all_new_states_p[i][-1][:, b, tmin:tmax]):  # across trials
                    store = False
                # store various pieces of data from this participant
                dRTs.extend(RTs[b, tmin:tmax])
                dplans.extend(p_plans[b, tmin:tmax])
                ddist.extend(dists_to_rew[b, tmin:tmax])
                dts.extend(-trial_ts[b, tmin:tmax])
                Nplan_dat.extend(N_plans[b, tmin:tmax])
                new_trial_nums.extend(trial_nums[b, tmin:tmax])  # trial numbers
                new_anums.extend(range(tmin, tmax))  # global action number

        if trial_type == "exploit":
            inds = np.where(np.array(new_trial_nums) > 1.5)[0]  # exploitation trials
        else:
            inds = np.where(np.array(new_trial_nums) < 1.5)[0]  # exploration trials

        dRTs, dplans, ddist, dts = [np.array(dat)[inds] for dat in [dRTs, dplans, ddist, dts]]  # subselection

        # append data to global data structures
        if store:
            RTs_by_u.append(dRTs)
            p_plans_by_u.append(dplans)
            dists_by_u.append(ddist)
            steps_by_u.append(dts)
            N_plans_by_u.append(np.array(Nplan_dat)[inds])
            anums_by_u.append(np.array(new_anums)[inds])
            trialnums_by_u.append(np.array(new_trial_nums)[inds])
            # compute some correlations
            s_pplan, s_dist, s_ts = np.corrcoef(dRTs, dplans)[0, 1], np.corrcoef(dRTs, ddist)[0, 1], np.corrcoef(dRTs, dts)[0, 1]
            allsims[i, :] = [s_pplan, s_dist, s_ts]
        else:
            print(f"skipping {i}!!!")

    # calculate mean and standard error of correlations by user
    mean_corr = np.mean(allsims, axis=0)
    std_err_corr = sem(allsims, axis=0)
    print("by user: ", mean_corr, " ", std_err_corr)

    # let's just save a lot of data and we can select what to plot later
    data = {
        "correlations": allsims,
        "RTs_by_u": RTs_by_u,
        "pplans_by_u": p_plans_by_u,
        "dists_by_u": dists_by_u,
        "steps_by_u": steps_by_u,
        "N_plans_by_u": N_plans_by_u,
        "N_plans": np.vstack(N_plans_by_u),
        "trial_nums_by_u": trialnums_by_u,
        "anums_by_u": anums_by_u,
    }

    # save data for later loading
    savename = f"{prefix}N{N}_Lplan{Lplan}{trialstr}_{epoch}"
    with open(f"{datadir}/RT_predictions_{savename}.bson", 'wb') as f:
        f.write(bson.BSON.encode({'data': data}))

# run_default_analyses is a global parameter in anal_utils.jl
if run_default_analyses:
    repeat_human_actions(seeds=seeds, N=N, Lplan=Lplan, epoch=epoch)  # call analysis function