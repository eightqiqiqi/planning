import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from scipy.stats import sem

# 设置绘图参数
cm = 1 / 2.54  # 厘米转英寸的比例
fig = plt.figure(figsize=(15 * cm, 3 * cm))
bot, top = 0, 1.0  # 图形的上下边界
col_c = 'b'  # 条形图颜色
fsize_label = 10  # 标签字体大小
fsize_leg = 8  # 图例字体大小

# 初始化列表
RTs_play, TTs_play, base1s, base2s, rews = [], [], [], [], []

# 模拟加载数据和处理
wrapstr_list = ["", "_euclidean"]
for wrapstr in wrapstr_list:
    # 加载数据
    with open(f"{datadir}/human_all_data_play{wrapstr}.bson", 'rb') as f:
        data = pickle.load(f)
    all_rews_p, all_RTs_p, all_trial_nums_p = data["all_rews_p"], data["all_RTs_p"], data["all_trial_nums_p"]
    
    with open(f"{datadir}/human_all_data_follow{wrapstr}.bson", 'rb') as f:
        data = pickle.load(f)
    all_rews_f, all_RTs_f, all_trial_nums_f = data["all_rews_f"], data["all_RTs_f"], data["all_trial_nums_f"]
    
    # 计算均值并处理非离群用户
    means1 = np.nanmean(all_RTs_f, axis=1)
    means2 = np.nanmean(all_RTs_p, axis=1)
    keep = np.where(means1 < 690)[0]  # 非离群用户
    Nkeep = len(keep)
    means1, means2 = means1[keep], means2[keep]
    
    # 加载先验分布参数
    with open(f"{datadir}/guided_lognormal_params_delta{wrapstr}.bson", 'rb') as f:
        params = pickle.load(f)
    
    initial_delays = params["initial"][:, 2] + np.exp(params["initial"][:, 0] + params["initial"][:, 1] ** 2 / 2)
    later_delays = params["later"][:, 2] + np.exp(params["later"][:, 0] + params["later"][:, 1] ** 2 / 2)
    
    base1s.append(initial_delays[keep])
    base2s.append(later_delays[keep])
    RTs_play.append(means2)
    
    # 计算奖励均值
    rews.append([np.nansum(rew) / rew.shape[0] for rew in all_rews_p][keep])
    
    # 计算思维时间
    all_TTs = []
    for u in keep:
        new_TTs = []
        rts, tnums = all_RTs_p[u], all_trial_nums_p[u]
        initial, later = params["initial"][u, :], params["later"][u, :]
        
        def initial_post_mean(r):
            return calc_post_mean(r, muhat=initial[0], sighat=initial[1], deltahat=initial[2], mode=False)
        
        def later_post_mean(r):
            return calc_post_mean(r, muhat=later[0], sighat=later[1], deltahat=later[2], mode=False)
        
        tnum = 1
        for ep in range(rts.shape[0]):
            for b in range(np.sum(tnums[ep, :] > 0.5)):  # 对每个动作
                t, rt = tnums[ep, b], rts[ep, b]  # 试验编号和反应时间
                if t > 1.5:  # 如果处于利用阶段
                    if t == tnum:  # 与上次试验相同
                        new_TTs.append(later_post_mean(rt))
                    else:  # 新试验的第一次动作
                        new_TTs.append(initial_post_mean(rt))
                tnum = t
        all_TTs.append(np.nanmean(new_TTs))
    
    TTs_play.append(all_TTs)

# 设置绘图数据
titles = ["reaction", "thinking", "initial", "later", "rewards"]
ylabs = ["time (ms)", "thinking time (ms)", "time (ms)", "time (ms)", "avg. reward"]
datas = [RTs_play, TTs_play, base1s, base2s, rews]
inds = [1, 4]  # 选择绘制 "thinking time" 和 "avg. reward"
titles, datas, ylabs = [titles[i] for i in inds], [datas[i] for i in inds], [ylabs[i] for i in inds]

# 创建绘图
grids = fig.add_gridspec(nrows=1, ncols=len(datas), left=0.00, right=0.36, bottom=bot, top=top, wspace=0.6)

# 绘制每个数据集
for idat, data in enumerate(datas):
    torus, euclid = data
    NT, NE = len(torus), len(euclid)
    mus = [np.mean(torus), np.mean(euclid)]  # 用户的均值
    diff = mus[0] - mus[1]
    comb = np.concatenate([torus, euclid])
    ctrls = np.zeros(10000)
    
    # 计算p值
    for i in range(10000):
        newcomb = np.random.permutation(comb)
        ctrls[i] = np.mean(newcomb[:NT]) - np.mean(newcomb[NT:])
    
    print(f"{titles[idat]} means: {mus} p = {np.mean(ctrls > diff)}")
    
    # 绘制条形图
    ax = fig.add_subplot(grids[0, idat])
    ax.bar([1, 2], mus, color=col_c)
    
    # 绘制每个数据点
    ax.scatter(np.ones(NT) + np.random.randn(NT) * 0.1, torus, marker=".", s=6, color="k", zorder=100)
    ax.scatter(np.ones(NE) * 2 + np.random.randn(NE) * 0.1, euclid, marker=".", s=6, color="k", zorder=100)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["wrap", "no-wrap"], rotation=45, ha="right")
    ax.set_ylabel(ylabs[idat])

# 绘制奖励与反应时间的关系
grids = fig.add_gridspec(nrows=1, ncols=2, left=0.47, right=1.0, bottom=bot, top=top, wspace=0.6)
ax = fig.add_subplot(grids[0, 1])
for i in range(2):
    ax.scatter(RTs_play[i], rews[i], color=["k", col_c][i], marker=".", s=60)
ax.legend(["wrap", "no-wrap"], loc=(0.4, 0.65), handletextpad=0.4, borderaxespad=0.3, handlelength=1.0, fontsize=fsize_leg)
ax.set_xlabel("mean RT (ms)")
ax.set_ylabel("mean reward")
ax.set_yticks(np.arange(4, 17, 2))

# 绘制路径长度的分布
with open(f"{datadir}/wrap_and_nowrap_pairwise_dists.bson", 'rb') as f:
    dists = pickle.load(f)

ds = np.arange(1, 13)
hist_wraps = [np.sum(dists[0] == d) for d in ds]
hist_nowraps = [np.sum(dists[1] == d) for d in ds]
xs = np.concatenate([[d - 0.5, d + 0.5] for d in ds])
hwraps = np.concatenate([[h, h] for h in hist_wraps / np.sum(hist_wraps)])
hnowraps = np.concatenate([[h, h] for h in hist_nowraps / np.sum(hist_wraps)])
ax = fig.add_subplot(grids[0, 2])
ax.plot(xs, hwraps, color="k")
ax.plot(xs, hnowraps, color=col_c)
ax.axvline(np.mean(dists[0]), color="k", lw=1.5)
ax.axvline(np.mean(dists[1]), color=col_c, lw=1.5)
ax.set_xlabel("distance to goal")
ax.set_ylabel("frequency")

# 添加标签并保存
y1 = 1.16
x1, x2, x3, x4 = -0.05, 0.15, 0.40, 0.70
plt.text(x1, y1, "A", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
plt.text(x2, y1, "B", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
plt.text(x3, y1, "C", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
plt.text(x4, y1, "D", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)

# 保存图像
plt.savefig("./figs/supp_human_euclidean_comparison.pdf", bbox_inches="tight")
plt.savefig("./figs/supp_human_euclidean_comparison.png", bbox_inches="tight")
plt.close()
