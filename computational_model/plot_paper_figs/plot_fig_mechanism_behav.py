# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os

# 设置全局变量
bot, top = 0.0, 1.0
fig = plt.figure(figsize=(17 * 0.1, 3.0 * 0.1))  # 设置图形尺寸

# 载入数据并提取性能和熵信息
datadir = "数据目录路径"  # 替换为实际数据路径
seeds = sorted([k for k in res_dict.keys()])
Nseed = len(seeds)
ms1, ms2, bs, es1, es2 = [], [], [], [], []
dists = list(range(1, 7))  # 距离目标的范围
bydist = []

# 遍历每个模型
for seed in seeds:
    dts, mindists, policies = [res_dict[seed][k] for k in ["dts", "mindists", "policies"]]
    
    # 选择那些回合结束的实验
    keepinds = np.where(np.sum(dts, axis=(1, 2)) != np.nan)  # 排除nan值
    new_dts = dts[:, keepinds, :]
    new_mindists = mindists[keepinds, 2]
    policies = policies[:, keepinds, :, :, :]
    
    # 计算每个回合的性能
    m1, m2 = np.mean(new_dts[1, :, :], axis=1), np.mean(new_dts[2, :, :], axis=1)
    bydist.append([np.mean(new_dts[1, new_mindists == dist, :], axis=1) for dist in dists])
    ms1.append(m1)
    ms2.append(m2)
    bs.append(np.mean(new_mindists))  # 存储最佳性能

    # 计算熵
    p1, p2 = policies[1, :, :, :, :], policies[2, :, :, :, :]
    p1, p2 = [p - np.log(np.sum(np.exp(p), axis=4)) for p in [p1, p2]]
    e1, e2 = [-np.sum(np.exp(p) * p, axis=4)[:, :, :, 1] for p in [p1, p2]]
    m1, m2 = [np.mean(e[:, :, 1], axis=1) for e in [e1, e2]]
    es1.append(m1)
    es2.append(m2)

# 汇总按距离的结果
bydist = np.mean(np.concatenate(bydist, axis=2), axis=2)[:, :, 0]

# 连接所有种子的结果
ms1, ms2, es1, es2 = [np.concatenate(arr, axis=1) for arr in [ms1, ms2, es1, es2]]

# 计算种子的均值和标准误差
m1, s1 = np.mean(ms1, axis=1), np.std(ms1, axis=1) / np.sqrt(Nseed)
m2, s2 = np.mean(ms2, axis=1), np.std(ms2, axis=1) / np.sqrt(Nseed)
me1, se1 = np.mean(es1, axis=1), np.std(es1, axis=1) / np.sqrt(Nseed)
me2, se2 = np.mean(es2, axis=1), np.std(es2, axis=1) / np.sqrt(Nseed)
nplans = np.arange(len(m1))  # 计划数量

# 绘制性能与回合次数的关系
grids = fig.add_gridspec(nrows=1, ncols=2, left=0.00, right=0.33, bottom=bot, top=1.0, wspace=0.50)
ax = fig.add_subplot(grids[0, 0])
ax.plot(nplans, m1, ls="-", color="blue", label="agent")  # 平均性能
ax.fill_between(nplans, m1 - s1, m1 + s1, color="blue", alpha=0.2)  # 标准误差

# 绘制最优基准线
ax.plot([nplans[0], nplans[-1]], [np.mean(bs)] * 2, color="red", ls="-", label="optimal")

# 绘制对照组性能
ax.plot(nplans, m2, ls=":", color="red", label="ctrl")  # 平均性能
ax.fill_between(nplans, m2 - s2, m2 + s2, color="red", alpha=0.2)  # 标准误差

# 设置标签和范围
ax.set_xlabel("# rollouts")
ax.set_ylabel("steps to goal")
ax.set_ylim(0.9 * np.mean(bs), np.max(m1 + s1) + 0.1 * np.mean(bs))
ax.set_xticks([0, 5, 10, 15])

# 绘制熵与回合次数的关系
ax = fig.add_subplot(grids[0, 1])
ax.plot(nplans, me1, ls="-", color="blue", label="agent")  # 平均熵
ax.fill_between(nplans, me1 - se1, me1 + se1, color="blue", alpha=0.2)  # 标准误差

# 绘制均匀策略的熵
ax.plot([nplans[0], nplans[-1]], [np.log(4)] * 2, color="red", ls="-", label="uniform")

# 设置标签和范围
ax.set_xlabel("# rollouts")
ax.set_ylabel("entropy (nats)", labelpad=1)
ax.set_ylim(0, 1.1 * np.log(4))
ax.set_xticks([0, 5, 10, 15])
ax.set_yticks([0, 1])

# 绘制有无回合的性能比较
@loadmat(f"{datadir}/performance_with_out_planning.mat")  # 加载数据
results = loadmat(f"{datadir}/performance_with_out_planning.mat")
ress = np.zeros((len(seeds), 2))

# 计算每个种子下有回合和无回合的平均奖励
for i, plan in enumerate([True, False]):
    for iseed, seed in enumerate(seeds):
        rews = results[seed][plan]
        ress[iseed, i] = np.sum(rews) / rews.shape[0]

# 计算平均值和标准误差
m, s = np.mean(ress, axis=0), np.std(ress, axis=0) / np.sqrt(len(seeds))

# 输出结果
print("Performance with and without rollouts:")
print(m, " ", s)

# 加载随机打乱回合的数据
@loadmat(f"{datadir}/performance_shuffled_planning.mat")  # 加载数据
results_shuff = loadmat(f"{datadir}/performance_shuffled_planning.mat")
ress_shuff = np.zeros((len(seeds), 2))

# 计算每个种子下随机打乱回合的平均奖励
for i, shuffle in enumerate([True, False]):
    for iseed, seed in enumerate(seeds):
        rews = results_shuff[seed][shuffle]
        ress_shuff[iseed, i] = np.sum(rews) / rews.shape[1]

# 计算平均值和标准误差
m_shuff, s_shuff = np.mean(ress_shuff, axis=0), np.std(ress_shuff, axis=0) / np.sqrt(len(seeds))

# 输出打乱回合的结果
print(f"Shuffled performance: {m_shuff} ({s_shuff})")

# 合并结果
ress = np.hstack([ress, ress_shuff[:, :1]])

# 绘制性能比较图
grids = fig.add_gridspec(nrows=1, ncols=1, left=0.405, right=0.505, bottom=0, top=1, wspace=0.15)
ax = fig.add_subplot(grids[0, 0])
ax.bar([0, 1, 2], m, yerr=s, color="blue", capsize=5)  # 绘制条形图
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["rollout", "no roll", "shuffled"])
ax.set_ylabel("avg. reward")
ax.set_yticks([6, 7, 8])
ax.set_rotation(45)

# 绘制目标导向和非目标导向回合示例
grids = fig.add_gridspec(nrows=1, ncols=1, left=0.53, right=0.69, bottom=-0.03, top=0.80, wspace=0.15)
ax = fig.add_subplot(grids[0, 0])

# 加载回合数据
@loadmat(f"{datadir}/example_rollout.mat")  # 加载示例回合数据
store = loadmat(f"{datadir}/example_rollout.mat")
plan_state, ps, ws, state = store[10]
rew_loc = state_from_onehot(4, ps)

# 绘制竞赛场地
arena_lines(ps, ws, 4, rew=False, col="k", rew_col="k", col_arena="k", lw_arena=2, lw_wall=2)

# 绘制成功和失败的回合路径
labels = ["successful", "unsuccessful"]
for i in range(2):  # 成功和失败
    col = ["blue", "red"][i]  # 使用不同的颜色

    # 提取回合路径
    plan_state, ps, ws, state = store[[4, 5][i]]
    plan_state = plan_state[:np.sum(plan_state > 0)]
    states = [state, state_from_loc(4, plan_state)]  # 添加初始状态
    N = states.shape[1]

    # 绘制每段路径
    for s in range(N - 1):  # 遍历路径段
        x1, y1, x2, y2 = states[:, s], states[:, s + 1]
        if s == 0:
            label = labels[i]
        else:
            label = None

        # 绘制路径
        ax.plot([x1, x2], [y1, y2], color=col, label=label, lw=2)

# 标记原始位置和目标位置
ax.scatter(state[0], state[1], color="blue", s=150, zorder=1000)  # 原始位置
ax.plot(rew_loc[0], rew_loc[1], color="black", marker="x", markersize=12, ls="", mew=3)  # 目标位置

# 添加图例
ax.legend(frameon=False, fontsize=12, loc="upper center", bbox_to_anchor=(0.5, 1.28))

# 绘制策略变化（成功和失败回合）
grids = fig.add_gridspec(nrows=1, ncols=2, left=0.78, right=1.00, bottom=0, top=top, wspace=0.10)
for i in range(2):  # 奖励和非奖励回合
    ms = []
    for seed in seeds:  # 遍历每个种子
        data = loadmat(f"{datadir}/causal_N100_Lplan8_{seed}_{plan_epoch}.mat")
        p_simulated_actions, p_simulated_actions_old = data["p_simulated_actions"], data["p_simulated_actions_old"]
        p_initial_sim, p_continue_sim = data["p_initial_sim"], data["p_continue_sim"]

        # 重新标准化行动概率
        p_simulated_actions /= (1 - p_continue_sim)
        p_simulated_actions_old /= (1 - p_initial_sim)
        inds = np.where(np.isnan(np.sum(p_simulated_actions, axis=0)) == False)  # 检查数据是否有效
        ms.append([np.mean(p_simulated_actions_old[i, inds]), np.mean(p_simulated_actions[i, inds])])

    ms = np.concatenate(ms, axis=1)  # 连接各个种子的结果
    m3, s3 = np.mean(ms, axis=1)[:2], np.std(ms, axis=1)[:2] / np.sqrt(len(seeds))

    # 绘制条形图
    ax = fig.add_subplot(grids[0, i])
    ax.bar([1, 2], m3, yerr=s3, color=["blue", "red"][i], capsize=5)
    ax.scatter([1 + np.random.randn(len(ms[0])) * 0.2, 2 + np.random.randn(len(ms[1])) * 0.2],
               [ms[0], ms[1]], color="black", marker=".", s=15, zorder=100)

    # 设置图表参数
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["pre", "post"])
    ax.set_ylim(0.0, 0.8)
    ax.axhline(0.25, color="gray", ls="-")

# 添加标签并保存
plt.text(0.05, 0.95, "A", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=14)
plt.text(0.15, 0.95, "B", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=14)
plt.text(0.34, 0.95, "C", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=14)
plt.text(0.51, 0.95, "D", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=14)
plt.text(0.71, 0.95, "E", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=14)

# 保存图像
plt.savefig("./figs/fig_mechanism_behav.pdf", bbox_inches="tight")
plt.savefig("./figs/fig_mechanism_behav.png", dpi=300, bbox_inches="tight")
plt.show()
