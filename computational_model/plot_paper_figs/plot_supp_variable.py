import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import random

# 设置图形大小
fig = plt.figure(figsize=(12, 3))

# 绘制思考时间与执行回合概率的关系图
grids = fig.add_gridspec(nrows=1, ncols=1, left=0.0, right=0.3, bottom=0.0, top=1.0, wspace=0.05)
ax = fig.add_subplot(grids[0, 0])

# 加载数据
data = loadmat(f"{datadir}RT_predictions_variable_N100_Lplan8_{plan_epoch}.mat")
allsims, RTs, pplans, dists = [data[k] for k in ["correlations", "RTs_by_u", "pplans_by_u", "dists_by_u"]]
RTs, pplans, dists = [np.concatenate(arr, axis=0) for arr in [RTs, pplans, dists]]

# 定义直方图的bin边缘
bins = np.arange(0.1, 0.8, 0.05)
xs = 0.5 * (bins[:-1] + bins[1:])  # 计算bin的中心点

# 打乱RTs数据
RTs_shuff = np.random.permutation(RTs)

# 根据pplans分组RTs数据
dat = [RTs[(pplans >= bins[i]) & (pplans < bins[i+1])] for i in range(len(bins)-1)]
dat_shuff = [RTs_shuff[(pplans >= bins[i]) & (pplans < bins[i+1])] for i in range(len(bins)-1)]

# 计算均值和标准误差
m, m_c = [np.mean(d) for d in [dat, dat_shuff]]
s, s_c = [np.std(d) / np.sqrt(len(d)) for d in [dat, dat_shuff]]

# 绘制条形图
ax.bar(xs, m, color=col_p, width=0.04, linewidth=0, label="data")
ax.errorbar(xs, m, yerr=s, fmt="none", color="k", capsize=2, lw=1.5)
ax.errorbar(xs, m_c, yerr=s_c, fmt="-", color=col_c, capsize=2, lw=1.5, label="shuffle")
ax.set_xlabel(r"$\pi$ (rollout)")
ax.set_ylabel("thinking time (ms)")
ax.set_yticks([0, 50, 100, 150, 200, 250])
ax.set_ylim(0, 250)
ax.legend(frameon=False, fontsize=fsize_leg)

# 打印相关系数的均值和标准误
m = np.mean(allsims, axis=0)[:2]
s = np.std(allsims, axis=0)[:2] / np.sqrt(len(allsims))
print(f"mean and sem of correlations: {m} {s}")

# ---------------------------
# 绘制表现与回合数的关系图
# ---------------------------

# 加载表现数据
res_dict = loadmat(f"{datadir}/perf_by_n_variable_N100_Lplan8.mat")
seeds = sorted(res_dict.keys())
Nseed = len(seeds)

# 初始化空数组
ms1, ms2, bs, es1, es2 = [], [], [], [], []

# 逐个遍历模型
for seed in seeds:
    dts, mindists, policies = [res_dict[seed][k] for k in ["dts", "mindists", "policies"]]
    
    # 选择试验完成的回合
    keepinds = np.where(np.isnan(np.sum(dts, axis=(1, 3))) == False)[0]
    new_dts = dts[:, keepinds, :]
    new_mindists = mindists[keepinds, 1]
    policies = policies[:, keepinds, :, :, :]
    
    # 计算回合的均值
    m1, m2 = np.mean(new_dts[0, :, :], axis=1), np.mean(new_dts[1, :, :], axis=1)
    ms1.append(m1)
    ms2.append(m2)
    bs.append(np.mean(new_mindists))
    
    # 提取log策略
    p1, p2 = policies[0, :, :, :, :], policies[1, :, :, :, :]
    p1, p2 = [p - np.log(np.sum(np.exp(p), axis=4, keepdims=True)) for p in [p1, p2]]
    
    # 计算熵
    e1, e2 = [-np.sum(np.exp(p) * p, axis=4)[:, :, :, 0] for p in [p1, p2]]
    m1, m2 = [np.mean(e[:, :, 0], axis=1) for e in [e1, e2]]
    es1.append(m1)
    es2.append(m2)

# 合并所有模型的结果
ms1, ms2, es1, es2 = [np.concatenate(arr, axis=1) for arr in [ms1, ms2, es1, es2]]

# 计算均值和标准误差
m1, s1 = np.mean(ms1, axis=1), np.std(ms1, axis=1) / np.sqrt(Nseed)
m2, s2 = np.mean(ms2, axis=1), np.std(ms2, axis=1) / np.sqrt(Nseed)
me1, se1 = np.mean(es1, axis=1), np.std(es1, axis=1) / np.sqrt(Nseed)
me2, se2 = np.mean(es2, axis=1), np.std(es2, axis=1) / np.sqrt(Nseed)
nplans = np.arange(len(m1))

# 绘制表现与回合数的关系图
grids = fig.add_gridspec(nrows=1, ncols=1, left=0.41, right=0.62, bottom=0.0, top=1.0, wspace=0.50)
ax = fig.add_subplot(grids[0, 0])
ax.plot(nplans, m1, ls="-", color=col_p, label="agent")
ax.fill_between(nplans, m1-s1, m1+s1, color=col_p, alpha=0.2)
ax.plot([nplans[0], nplans[-1]], [np.mean(bs)]*2, color=col_c, ls="-", label="optimal")
ax.legend(frameon=False, loc="upper right", fontsize=fsize_leg)
ax.set_xlabel("# rollouts")
ax.set_ylabel("steps to goal")
ax.set_ylim(0.9 * np.mean(bs), np.max(m1 + s1) + 0.1 * np.mean(bs))
ax.set_xticks([0, 5, 10, 15])

# 绘制策略变化图（成功与失败的回合）
grids = fig.add_gridspec(nrows=1, ncols=2, left=0.75, right=1.00, bottom=0, top=1.0, wspace=0.10)
for i in range(2):
    ms = []
    for seed in seeds:
        data = loadmat(f"{datadir}/variable_causal_N100_Lplan8_{seed}_{plan_epoch}.mat")
        p_simulated_actions, p_simulated_actions_old = data["p_simulated_actions"], data["p_simulated_actions_old"]
        p_initial_sim, p_continue_sim = data["p_initial_sim"], data["p_continue_sim"]
        
        # 重新归一化
        p_simulated_actions /= (1 - p_continue_sim)
        p_simulated_actions_old /= (1 - p_initial_sim)
        
        # 只选择有效数据
        inds = np.where(np.isnan(np.sum(p_simulated_actions, axis=0)) == False)[0]
        ms.append(np.mean(p_simulated_actions_old[0, inds]), np.mean(p_simulated_actions[0, inds]))
    
    ms = np.concatenate(ms, axis=1)
    m3, s3 = np.mean(ms, axis=1)[:2], np.std(ms, axis=1)[:2] / np.sqrt(len(seeds))

    # 绘制结果
    ax = fig.add_subplot(grids[0, i])
    ax.bar([1, 2], m3, yerr=s3, color=[col_p1, col_p2][i], capsize=capsize)
    ax.scatter(np.repeat([1, 2], len(ms)), ms.flatten(), color=col_point, marker=".", s=15, zorder=100)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["pre", "post"])
    if i == 0:
        ax.set_ylabel(r"$\pi(\hat{a}_1)$", labelpad=0)
        ax.set_title("succ.", fontsize=fsize)
        ax.set_yticks([0.1, 0.3, 0.5, 0.7])
    else:
        ax.set_title("unsucc.", fontsize=fsize)
        ax.set_yticks([])

    ax.set_ylim(0.0, 0.8)
    ax.set_xlim([0.4, 2.6])
    ax.axhline(0.25, color=col_c, ls="-")

# 添加标签并保存图形
plt.text(x1, y1, "A", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
plt.text(x2, y1, "B", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
plt.text(x3, y1, "C", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)

# 保存为PDF和PNG
plt.savefig("./figs/supp_variable_time.pdf", bbox_inches="tight")
plt.savefig("./figs/supp_variable_time.png", bbox_inches="tight")
plt.close()
