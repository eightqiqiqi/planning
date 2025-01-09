import numpy as np
import os
import pickle
import bson
from .anal_utils import datadir,lognorm

# 设置wrapstr标记，用于运行欧几里得分析
wrapstr = ""
# wrapstr = "_euclidean"  # 如果需要运行欧几里得分析，取消注释这一行

#print("计算人类反应时间的先验参数")
print("computing prior parameters for human response times")

# 先加载我们处理过的引导（'follow'）试验的数据
data_path = os.path.join(datadir, f"human_all_data_follow{wrapstr}.bson")
with open(data_path, 'rb') as f:
    data = pickle.load(f)

# 解包数据
_, _, _, _, _, all_RTs_f, all_trial_nums_f, all_trial_time_f = data
Nuser = len(all_RTs_f)  # 用户数量

# 创建存储每个参与者初始和后续行动的字典，存储3个参数（初始、后续）
params = {key: np.zeros((Nuser, 3)) for key in ["initial", "later"]}

# 对每个行动类型进行迭代：初始行动和后续行动
for i_init, initial in enumerate([True, False]):
    key = "initial" if initial else "later"  # 设置key为'initial'或'later'
    # 遍历每个参与者
    for u in range(Nuser):
        # 获取该用户的引导试验数据
        if initial:
            # 选择在试验中的初始行动
            inds = (all_trial_nums_f[u] > 1.5) & (all_trial_time_f[u] == 1)
        else:
            # 选择后续行动
            inds = (all_trial_nums_f[u] > 1.5) & (all_trial_time_f[u] > 1.5)
        RTs_f = all_RTs_f[u][inds]  # 获取反应时间数据
        RTs_f = RTs_f[~np.isnan(RTs_f)]  # 移除缺失数据
        if u % 10 == 0:
            print(f"user {u+1}, {key} actions, {len(RTs_f)} datapoints")
        
        # try different deltas in our shifted lognormal prior
        deltas = np.arange(0, np.min(RTs_f), 1)  # list of deltas to try (the ones with appropriate support)
        Ls, mus, sigs = [np.zeros(len(deltas)) for _ in range(3)]  # corresponding log liks and optimal params

        for i, delta in enumerate(deltas):  # compute likelihood with each delta
            mus[i] = np.mean(np.log(RTs_f - delta))  # mean of the shifted lognormal
            sigs[i] = np.std(np.log(RTs_f - delta))  # standard deviation
            Ls[i] = np.sum(np.log(lognorm(RTs_f, mu=mus[i], sig=sigs[i], delta=delta)))  # log likelihood of the data

        # extract maximum likelihood parameters
        muhat, sighat, deltahat = [arr[np.argmax(Ls)] for arr in [mus, sigs, deltas]]
        # 存储估计的参数
        params[key][u, :] = [muhat, sighat, deltahat]

# write to file
filename = f"{datadir}/guided_lognormal_params_delta{wrapstr}.bson"
with open(filename, 'wb') as f:
    f.write(bson.BSON.encode({'params': params}))
