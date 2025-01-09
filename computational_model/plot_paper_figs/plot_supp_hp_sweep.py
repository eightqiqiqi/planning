import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from scipy.stats import sem

# 设置绘图参数
cm = 1 / 2.54  # 厘米转英寸的比例
fig = plt.figure(figsize=(17 * cm, 7.5 * cm))
bot, top = 0.72, 0.38  # 图形的上下边界
params = [(60, 4), (60, 8), (60, 12), (100, 4), (100, 8), (100, 12), (140, 4), (140, 8), (140, 12)]
prefix = "hp_sweep_"
Nseed = len(seeds)

# 设置颜色
col_p = 'b'  # 样本条形的颜色
col_point = 'r'  # 数据点颜色
capsize = 3  # 误差条的帽子大小
fsize = 10  # 字体大小
fsize_label = 12  # 标签字体大小

# 第一部分：绘制与思维时间的相关性
all_vals, all_errs, all_ticklabels, all_corrs = [], [], [], []
for ip, p in enumerate(params):
    N, Lplan = p
    savename = f"hp_sweep_N{N}_Lplan{Lplan}_1000_weiji"
    
    # 加载数据
    with open(f"{datadir}/RT_predictions_{savename}.bson", 'rb') as f:
        data = pickle.load(f)
    
    allsims = data["correlations"]
    m1, s1 = np.mean(allsims[:, 0]), np.std(allsims[:, 0]) / np.sqrt(allsims.shape[0])
    all_vals.append(m1)
    all_errs.append(s1)
    
    print(f"{p}: {m1} {s1}")
    corrs = allsims[:, 0]
    print(f"{np.min(corrs)} {np.max(corrs)}")
    all_corrs.append(corrs)
    all_ticklabels.append(str(p))

# 绘制相关性图
grids = fig.add_gridspec(nrows=1, ncols=1, left=0.04, right=0.47, bottom=bot, top=1.0, wspace=0.35)
ax = fig.add_subplot(grids[0, 0])
ax.bar(np.arange(1, len(all_vals) + 1), all_vals, yerr=all_errs, capsize=capsize, color=col_p)

# 绘制每个数据点
for i, corrs in enumerate(all_corrs):
    shifts = np.arange(1, len(corrs) + 1) - np.mean(np.arange(1, len(corrs) + 1))
    shifts = shifts / np.std(shifts) * 0.15
    ax.scatter(i + 1 + shifts, corrs, color=col_point, marker='.', s=3, alpha=0.5, zorder=100)

ax.set_xticks(np.arange(1, len(all_vals) + 1))
ax.set_xticklabels(all_ticklabels, rotation=45, ha="right")
ax.set_ylim(-0.05, 0.35)
ax.set_ylabel("correlation with\nthinking time")
ax.set_yticks(np.arange(0, 0.31, 0.1))

# 第二部分：绘制步骤的增量性能
all_vals, all_errs, all_ticklabels, all_diffs = [], [], [], []
for ip, p in enumerate(params):
    N, Lplan = p
    savename = f"{prefix}N{N}_Lplan{Lplan}"
    
    # 加载数据
    with open(f"{datadir}/perf_by_n_{savename}.bson", 'rb') as f:
        res_dict = pickle.load(f)
    
    seeds = sorted(res_dict.keys())
    ms = []
    t0, t1 = 1, 6
    for seed in seeds:
        dts, mindists, policies = [res_dict[seed][k] for k in ["dts", "mindists", "policies"]]
        keepinds = np.where(np.sum(np.isnan(dts[0, :, t0:t1]), axis=(1)) == 0 & (mindists[:, 2] >= 0))[0]
        new_dts = dts[:, keepinds, :]
        m = np.mean(new_dts[0, :, :], axis=1)
        ms.append(m[t0] - m[t1])

    ms = np.array(ms)
    m1, s1 = np.mean(ms), sem(ms)
    all_vals.append(m1)
    all_errs.append(s1)
    all_diffs.append(ms)
    all_ticklabels.append(str(p))

# 绘制增量性能图
grids = fig.add_gridspec(nrows=1, ncols=1, left=0.57, right=1.0, bottom=bot, top=1.0, wspace=0.35)
ax = fig.add_subplot(grids[0, 0])
ax.bar(np.arange(1, len(all_vals) + 1), all_vals, yerr=all_errs, capsize=capsize, color=col_p)

# 绘制每个数据点
for i, diffs in enumerate(all_diffs):
    shifts = np.arange(1, len(diffs) + 1) - np.mean(np.arange(1, len(diffs) + 1))
    shifts = shifts / np.std(shifts) * 0.15
    ax.scatter(i + 1 + shifts, diffs, color=col_point, marker='.', s=3, alpha=0.5, zorder=100)

ax.set_xticks(np.arange(1, len(all_vals) + 1))
ax.set_xticklabels(all_ticklabels, rotation=45, ha="right")
ax.set_ylim(0, 1.5)
ax.set_ylabel("$\Delta$steps")

# 第三部分：绘制 delta pi(a1)
grids = fig.add_gridspec(nrows=1, ncols=len(params), left=0.0, right=1.0, bottom=0, top=top, wspace=0.35)
for ip, p in enumerate(params):
    N, Lplan = p
    all_ms = []
    for i in range(2):  # 对奖励和非奖励的模拟
        ms = []
        for seed in range(51, 56):  # 选择种子
            if p == (140, 12):
                with open(f"{datadir}/causal_N{N}_Lplan{Lplan}_{seed}_1000_single.bson", 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(f"{datadir}/causal_N{N}_Lplan{Lplan}_{seed}1000_single.bson", 'rb') as f:
                    data = pickle.load(f)
            
            p_simulated_actions = data["p_simulated_actions"]
            p_simulated_actions_old = data["p_simulated_actions_old"]
            p_initial_sim = data["p_initial_sim"]
            p_continue_sim = data["p_continue_sim"]
            p_simulated_actions /= (1 - p_continue_sim)
            p_simulated_actions_old /= (1 - p_initial_sim)
            inds = np.where(np.isnan(np.sum(p_simulated_actions, axis=0)) == 0)[0]
            ms.append([np.mean(p_simulated_actions_old[i, inds]), np.mean(p_simulated_actions[i, inds])])

        ms = np.array(ms)
        all_ms.append(ms[1, :] - ms[0, :])

    ms = [np.mean(ms) for ms in all_ms]
    ss = [np.std(ms) / np.sqrt(Nseed) for ms in all_ms]
    ax = fig.add_subplot(grids[0, ip])
    ax.bar([1, 2], ms, yerr=ss, color=[col_p1, col_p2], capsize=capsize)

    if plot_points:
        shifts = np.arange(1, len(all_ms[0]) + 1) - np.mean(np.arange(1, len(all_ms[0]) + 1))
        shifts = shifts / np.std(shifts) * 0.2
        ax.scatter([1 + shifts, 2 + shifts], [all_ms[0], all_ms[1]], color=col_point, marker='.', s=15, zorder=100)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["succ", "un"])
    if ip == 0:
        ax.set_ylabel("$\Delta \pi(\hat{a}_1)$", labelpad=0)
        ax.set_yticks([-0.4, 0, 0.4])
    else:
        ax.set_yticks([])

    ax.set_title(str(p), fontsize=fsize)
    ax.set_ylim(-0.4, 0.4)
    ax.set_xlim([0.4, 2.6])
    ax.axhline(0.0, color="k", lw=1)

# 添加标签并保存
add_labels = True
if add_labels:
    y1, y2 = 1.07, 0.48
    x1, x2 = -0.09, 0.50
    plt.text(x1, y1, "A", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=fsize)
    plt.text(x2, y1, "B", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=fsize)
    plt.text(x1, y2, "C", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=fsize)

# 保存图形
plt.savefig("./figs/supp_hp_sweep.pdf", bbox_inches="tight")
plt.savefig("./figs/supp_hp_sweep.png", bbox_inches="tight")
plt.close()
