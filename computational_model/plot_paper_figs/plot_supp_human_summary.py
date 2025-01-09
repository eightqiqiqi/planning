import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import sem
import os

# 设置绘图参数
cm = 1 / 2.54  # 厘米转英寸的比例
fig = plt.figure(figsize=(15 * cm, 3 * cm))
bot, top = 0, 1.0  # 图形的上下边界
col_c = 'b'  # 条形图颜色
fsize_label = 10  # 标签字体大小
fsize_leg = 8  # 图例字体大小

# 初始化一些参数
wrap = False
wrapstr = "" if wrap else "_euclidean"
datadir = '数据目录'  # 需要设置数据目录路径

# 加载数据：非引导和引导情境的行为数据
with open(f"{datadir}/human_RT_and_rews_play{wrapstr}.bson", 'rb') as f:
    data_play = pickle.load(f)  # 非引导情境数据
with open(f"{datadir}/human_RT_and_rews_follow{wrapstr}.bson", 'rb') as f:
    data_follow = pickle.load(f)  # 引导情境数据

# 计算反应时间和奖励的均值
means1 = [np.nanmean(RTs) for RTs in data_follow["all_RTs"]]
means2 = [np.nanmean(RTs) for RTs in data_play["all_RTs"]]

keep = np.where(np.array(means1) < 690)[0]  # 非离群用户
Nkeep = len(keep)

# 计算所有用户的平均反应时间和奖励
mean_RTs = [
    [np.nanmean(RTs) for RTs in data["all_RTs"]] for data in [data_follow, data_play]
]
mean_rews = [
    [np.sum(rews) / rews.shape[0] for rews in data["all_rews"]] for data in [data_follow, data_play]
]

# 创建绘图
grids = fig.add_gridspec(nrows=1, ncols=3, left=0.00, right=0.78, bottom=0, top=1.0, wspace=0.5)

# 绘制平均反应时间与平均奖励的关系
for i in range(2):  # 分别绘制引导和非引导情境的数据
    ax = fig.add_subplot(grids[0, i])
    ax.scatter(mean_RTs[i][keep], mean_rews[i][keep], color="k", marker=".", s=60)
    ax.set_xlabel("mean RT (ms)")
    ax.set_ylabel("mean reward")
    ax.set_title(["guided", "non-guided"][i], fontsize=fsize_label)
    ax.set_yticks(np.arange(4, 13, 2))

# 加载所有必要的行为数据
with open(f"{datadir}/human_all_data_follow{wrapstr}.bson", 'rb') as f:
    data_follow = pickle.load(f)
with open(f"{datadir}/human_all_data_play{wrapstr}.bson", 'rb') as f:
    data_play = pickle.load(f)

# 提取需要的数据
all_states, all_ps, all_as, all_wall_loc, all_rews, all_RTs, _, _ = data_play
Larena = 4
# 定义环境参数
ed = EnvironmentDimensions(4 ** 2, 2, 5, 50, Larena)

# 计算每个用户的动作最优性
all_opts = []
for i in keep:
    opts = []  # 每个用户的最优性列表
    for b in range(all_as[i].shape[0]):
        dists = dist_to_rew(all_ps[i][:, b:b], all_wall_loc[i][:, :, b:b], Larena)
        if np.sum(all_rews[i][b, :]) > 0.5:  # 至少完成了一个试验
            for t in np.where(all_rews[i][b, :] > 0.5)[0][1:]:
                pi_opt = optimal_policy(np.array(all_states[i][:, b, t], dtype=int), all_wall_loc[i][:, :, b], dists, ed)
                opts.append(float(pi_opt[int(all_as[i][b, t])] > 1e-2))  # 判断动作是否最优
    all_opts.append(np.mean(opts))  # 存储该用户的最优性

RTs = [np.nanmean(RTs) for RTs in all_RTs[keep]]  # 对应的反应时间
inds = np.where(np.array(all_opts) > 0.5)[0]
RTs, all_opts = np.array(RTs)[inds], np.array(all_opts)[inds].astype(float)

# 计算反应时间和最优性之间的相关性
rcor = np.corrcoef(RTs, all_opts)[0, 1]
ctrls = np.zeros(10000)
for i in range(10000):
    ctrls[i] = np.corrcoef(RTs, np.random.permutation(all_opts))[0, 1]
print(f"correlation: {rcor}, p-value: {np.mean(ctrls > rcor)}")

# 绘制反应时间与最优性之间的关系
ax = fig.add_subplot(grids[0, 2])
ax.scatter(RTs, all_opts, color="k", marker=".", s=60)
ax.set_xlabel("mean RT (ms)")
ax.set_ylabel("$p$ (optimal)")

# 加载先验分布的参数
with open(f"{datadir}/guided_lognormal_params_delta{wrapstr}.bson", 'rb') as f:
    params = pickle.load(f)

# 计算初始动作和后续动作的时间延迟
initial_delays = params["initial"][:, 2] + np.exp(params["initial"][:, 0] + params["initial"][:, 1] ** 2 / 2)
later_delays = params["later"][:, 2] + np.exp(params["later"][:, 0] + params["later"][:, 1] ** 2 / 2)

# 绘制每个用户的先验分布均值
grids = fig.add_gridspec(nrows=1, ncols=1, left=0.90, right=1.0, bottom=0, top=1.0, wspace=0.4)
ax = fig.add_subplot(grids[0, 0])
mus = [np.mean(initial_delays[keep]), np.mean(later_delays[keep])]  # 对所有用户的均值进行计算
ax.bar([1, 2], mus, color=col_c)  # 绘制条形图
# 绘制每个用户的数据点
ax.scatter(np.ones(Nkeep) + np.random.randn(Nkeep) * 0.1, initial_delays[keep], marker=".", s=6, color="k", zorder=1000)
ax.scatter(np.ones(Nkeep) * 2 + np.random.randn(Nkeep) * 0.1, later_delays[keep], marker=".", s=6, color="k", zorder=100)
ax.set_xticks([1, 2])
ax.set_xticklabels(["initial", "later"], rotation=45, ha="right")
ax.set_ylabel("time (ms)")

# 输出一些结果
print(f"correlation between thinking time and optimality: {rcor}, p = {np.mean(ctrls > rcor)}")
print(f"mean optimality: {np.mean(all_opts)}")

# 添加标签并保存
y1 = 1.16
x1, x2, x3, x4 = -0.09, 0.21, 0.49, 0.80
plt.text(x1, y1, "A", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
plt.text(x2, y1, "B", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
plt.text(x3, y1, "C", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
plt.text(x4, y1, "D", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)

# 保存图形
plt.savefig(f"./figs/supp_human_data{wrapstr}.pdf", bbox_inches="tight")
plt.savefig(f"./figs/supp_human_data{wrapstr}.png", bbox_inches="tight")
plt.close()
