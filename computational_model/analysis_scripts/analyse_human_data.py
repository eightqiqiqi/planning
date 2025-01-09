# 本脚本加载人类行为数据并保存一些有用的统计数据

# 导入必要的库
import sqlite3
import pandas as pd
import numpy as np
import time
import bson
from collections import defaultdict
import sys
import os
from ..src.human_utils_maze import extract_maze_data
from ..src.walls_baselines import dist_to_rew
from ..src.walls import state_ind_from_state
from ..src.walls_baselines import dist_to_rew
from .euclidean_prolific_ids import euclidean_ids
from .anal_utils import datadir


# 打印加载数据的开始信息
#print("加载并处理人类行为数据")
print("loading and processing human behavioural data") 
# 设置wraparound标志
wraparound = True

# 针对非引导（"play"）和引导（"follow"）两种情境执行分析
for game_type in ["play", "follow"]:
    
    # 构建强化学习环境
    T = 100  # 最大步数
    Larena = 4  # Arena的大小
    environment_dimensions = (Larena ** 2, 2, 5, T, Larena)  # 环境维度

    # 初始化存储数据的数组
    all_RTs, all_trial_nums, all_trial_time, all_rews, all_states = [], [], [], [], []
    all_wall_loc, all_ps, all_as = [], [], []
    Nepisodes, tokens = [], []

    # 根据wraparound标志选择数据库路径
    if wraparound:
        db = "../../human_data/prolific_data.sqlite"
        wrapstr = ""
    else:
        db = "../../human_data/Euclidean_prolific_data.sqlite"
        wrapstr = "_euclidean"
    
    # 连接数据库
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    # 获取所有用户id
    cursor.execute("SELECT id FROM users")
    users = cursor.fetchall()

    # 根据游戏类型设置初始丢弃的回合数
    nskip = 2 if game_type == "play" else 8  # number of initial episodes to discard

    print(f"loading users for game type: {game_type}")

    i_user = 0
    for user_id in users:
        user_eps = pd.read_sql(f"SELECT * FROM episodes WHERE user_id = {str(user_id[0])}", db)  # episode data
        usize = user_eps.shape[0]  # total number of episodes for this user
        info = pd.read_sql(f"SELECT * FROM users WHERE id = {str(user_id)}", db)
        token = info.iloc[0]["token"]
        
        if usize >= 58 and len(token) == 24:  # 完成任务且token长度符合要求
            if wraparound or token in euclidean_ids:
                i_user += 1
                if i_user % 10 == 0:
                    print(i_user)
                
                # 提取当前用户的迷宫数据
                rews, as_, states, wall_loc, ps, times, trial_nums, trial_time, RTs, shot = extract_maze_data(
                    conn, user_id, Larena, game_type=game_type, skip_init=nskip
                )
                # 将数据加入相应的容器中
                all_RTs.append(RTs)  # 反应时间
                all_rews.append(rews)  # 奖励
                all_trial_nums.append(trial_nums)  # 回合数
                all_trial_time.append(trial_time)  # 每回合时间
                all_states.append(states)  # 位置数据
                all_wall_loc.append(wall_loc)  # 墙的位置
                all_ps.append(ps)  # 奖励位置
                all_as.append(as_)  # 执行动作
                Nepisodes.append(len(ps[1]))  # 当前用户的回合数
                tokens.append(info[1])  # 保存token
        
    valid_users = range(len(all_rews))

    #print(f"正在处理 {len(valid_users)} 个有效用户的数据")
    print(f"processing data for {len(valid_users)} users")

    # 存储所有数据
    data = [all_states, all_ps, all_as, all_wall_loc, all_rews, all_RTs, all_trial_nums, all_trial_time]
    with open(f"{datadir}/human_all_data_{game_type}{wrapstr}.bson", "wb") as f:
        bson.dump(data, f)

    # 存储一些常用数据
    data = {"all_rews": all_rews, "all_RTs": all_RTs}
    with open(f"{datadir}/human_RT_and_rews_{game_type}{wrapstr}.bson", "wb") as f:
        bson.dump(data, f)

    # 计算每步奖励
    def comp_rew_by_step(rews, Rmin=4):
        keep_inds = np.where(np.sum(rews > 0.5, axis=1) >= Rmin)[0]
        # 容器，用于存储每次试验的持续时间（以步数为单位）
        all_durs = np.zeros((len(keep_inds), Rmin))
        # 遍历回合
        for ib, b in enumerate(keep_inds):
            sortrew = np.argsort(-rews[b, :])  # 按奖励时间降序排序，获取索引
            rewtimes = np.concatenate(([0], sortrew[:Rmin]))  # 添加初始时间点
            durs = rewtimes[1:Rmin+1] - rewtimes[:Rmin]  # 奖励时间之间的差值
            all_durs[ib, :] = durs  # 存储持续时间
        # 平均值
        μ = np.mean(all_durs, axis=0)
        # 标准误差
        s = np.std(all_durs, axis=0) / np.sqrt(len(keep_inds))
        return μ, s

    μs, ss = [], []
    Rmin = 4
    for i in valid_users:  # 对每个用户计算
        μ, s = comp_rew_by_step(all_rews[i], Rmin=Rmin)
        μs.append(μ)
        ss.append(s)
    μs = np.column_stack(μs)  # 合并数据
    ss = np.column_stack(ss)

    # 保存数据
    data = [Rmin, μs, ss]
    with open(f"{datadir}/human_by_trial_{game_type}{wrapstr}.bson", "wb") as f:
        bson.dump(data, f)

    # 按难度计算RT
    def human_RT_by_difficulty(T, rews, ps, wall_loc, Larena, trial_nums, trial_time, RTs, states):
        trials = 20  # maximum number of trials
        new_RTs = np.full((trials, rews.shape[0], T), np.nan)  # RTs
        new_dists = np.full((trials, rews.shape[0]), np.nan)  # distances to goal
        
        for b in range(rews.shape[0]):  # for each episode
            rew = rews[b, :]  # rewards in this episode
            min_dists = dist_to_rew(ps[:, b:b+1], wall_loc[:, :, b:b+1], Larena)  # minimum distances to goal for each state
            
            for trial in range(1, trials):  # consider only exploitation
                if np.sum(rew > 0.5) > (trial - 0.5):  # finished trial
                    inds = np.where((trial_nums[b, :] == trial) & (trial_time[b, :] > 0.5))[0]  # all timepoints within trial
                    new_RTs[trial, b, :len(inds)] = RTs[b, inds]  # reaction times
                    state = states[:, b, inds[0]]  # initial state
                    new_dists[trial, b] = min_dists[int(state[0]), int(state[1])]  # distance to goal from initial state        
        return new_RTs, new_dists  # return RTs and distances

    # 对每个用户进行计算
    RTs, dists = [], []
    for u in valid_users:
        new_RTs, new_dists = human_RT_by_difficulty(T, all_rews[u], all_ps[u], all_wall_loc[u], Larena, all_trial_nums[u], all_trial_time[u], all_RTs[u], all_states[u])
        RTs.append(new_RTs)
        dists.append(new_dists)

    # 保存计算结果
    data = [RTs, dists, all_trial_nums, all_trial_time]
    with open(f"{datadir}RT_by_complexity_by_user_{game_type}{wrapstr}.bson", "wb") as f:
        bson.dump(data, f)

    # 计算探索过程中每个唯一状态的RT
    all_unique_states = []
    for i in range(len(all_RTs)):  # for each user
        states, rews, as_ = all_states[i], all_rews[i], all_as[i]  # extract states, rewards, and actions
        unique_states = np.full(all_RTs[i].shape, np.nan)  # how many states had been seen when the action was taken
        
        for b in range(rews.shape[0]):  # for each episode
            if np.sum(rews[b, :]) == 0:  # if there are no finished trials
                tmax = np.sum(as_[b, :] > 0.5)  # iterate until end
            else:
                tmax = np.where(rews[b, :] == 1)[0][0]  # iterate until first reward
            
            visited = np.zeros(16, dtype=bool)  # which states have been visited
            for t in range(tmax):  # for each action in trial 1
                visited[int(state_ind_from_state(Larena, states[:, b, t])[0])] = True  # visited corresponding state
                unique_states[b, t] = np.sum(visited)  # number of unique states
        
        all_unique_states.append(unique_states)  # add to container

    # 保存计算结果
    data = [all_RTs, all_unique_states]
    with open(f"{datadir}unique_states_{game_type}{wrapstr}.bson", "wb") as f:
        bson.dump(data, f)
