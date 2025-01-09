import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import stats

# 设置图形大小
fig = plt.figure(figsize=(10.5, 7))  # 单位为英寸
grids = fig.add_gridspec(nrows=2, ncols=2, left=0.00, right=1.00, bottom=0, top=1.0, wspace=0.61, hspace=0.7)

# 加载数据
# 假设 seeds 和 datadir 是事先定义好的
seeds = [1, 2, 3, 4]  # 示例
datadir = './data/'  # 数据路径

# 初始化用于存储模型输出的列表
ms, unums = [], []

# 循环加载每个模型的结果
for seed in seeds:
    with open(f"{datadir}model_exploration_predictions_{seed}_plan_epoch.bson", 'rb') as f:
        data = pickle.load(f)
        unums, dec_perfs = data['unums'], data['dec_perfs']
        ms.append(dec_perfs)

# 合并所有 seed 的数据，并计算均值和标准误差
ms = np.hstack(ms)  # 沿着列方向拼接
m = np.mean(ms, axis=1)  # 计算均值
s = np.std(ms, axis=1) / np.sqrt(ms.shape[1])  # 计算标准误差

# 绘制第一张子图（准确率与访问状态数关系图）
ax = fig.add_subplot(grids[0, 0])
ax.plot(unums, 1 / (16 - unums), color='red', label="optimal", zorder=-1000)
ax.plot(unums, m, color='blue', label="agent")
ax.fill_between(unums, m - s, m + s, color='blue', alpha=0.2)
ax.set_xlabel("States Visited")
ax.set_ylabel("Accuracy")
ax.legend(frameon=False)
ax.set_ylim(0.0, 0.7)
ax.set_xlim(unums[0], unums[-1])

# 加载人类思维时间和行为数据
with open(f"{datadir}RT_predictions_N100_Lplan8_explore_1000.bson", 'rb') as f:
    data = pickle.load(f)
allsims, RTs, pplans = data['correlations'], data['RTs_by_u'], data['pplans_by_u']
RTs = np.concatenate(RTs)
pplans = np.concatenate(pplans)

# 计算分箱
bins = np.arange(0.05, 0.75, 0.05)
xs = 0.5 * (bins[:-1] + bins[1:])  # 分箱中心

# 生成随机打乱数据进行对照实验
RTs_shuff = np.random.permutation(RTs)
dat = [RTs[(pplans >= bins[i]) & (pplans < bins[i + 1])] for i in range(len(bins) - 1)]
dat_shuff = [RTs_shuff[(pplans >= bins[i]) & (pplans < bins[i + 1])] for i in range(len(bins) - 1)]

# 计算均值和标准误差
m = [np.mean(d) for d in dat]
s = [np.std(d) / np.sqrt(len(d)) for d in dat]
m_c = [np.mean(d) for d in dat_shuff]
s_c = [np.std(d) / np.sqrt(len(d)) for d in dat_shuff]

# 绘制第二张子图（人类思维时间与π(rollout)的关系）
ax = fig.add_subplot(grids[0, 1])
ax.bar(xs, m, color='blue', width=0.04, label="data")
ax.errorbar(xs, m, yerr=s, fmt='none', color='black', capsize=2, lw=1.5)
ax.errorbar(xs, m_c, yerr=s_c, fmt='-', color='red', capsize=2, lw=1.5, label="shuffle")
ax.set_xlabel(r"$\pi$ (rollout)")
ax.set_ylabel("Thinking Time (ms)")
ax.legend(frameon=False)

# 打印相关性均值和标准误差
m_allsims = np.mean(allsims, axis=0)
s_allsims = np.std(allsims, axis=0) / np.sqrt(allsims.shape[0])
print("Correlations mean and sem:", m_allsims, s_allsims)

# 绘制第三张子图（RL智能体思维时间与独立访问状态数关系图）
uvals = np.arange(2, 16)  # 考虑的独立状态数量
RTs_us = np.full((len(seeds), len(uvals)), np.nan)

# 对每个模型seed加载数据并计算思维时间
for iseed, seed in enumerate(seeds):
    with open(f"{datadir}model_unique_states_{seed}_1000.bson", 'rb') as f:
        data = pickle.load(f)
    RTs, unique_states = data['RTs'], data['unique_states']
    new_us, new_rts = [], []
    
    for b in range(RTs.shape[0]):  # 每个episode
        us = unique_states[b, :]
        inds = np.where(~np.isnan(us))[0][1:]  # 过滤掉NaN值
        rts = RTs[b, inds]
        new_us.append(us[inds])
        new_rts.append(rts)
    
    new_us = np.concatenate(new_us)
    new_rts = np.concatenate(new_rts)
    RTs_us[iseed, :] = [np.mean(new_rts[new_us == uval]) - 1 for uval in uvals] * 120

# 计算均值和标准误差
m_us = np.mean(RTs_us, axis=0)
s_us = np.std(RTs_us, axis=0) / np.sqrt(RTs_us.shape[0])

# 绘制RL智能体的思维时间与独立状态数量关系图
ax = fig.add_subplot(grids[1, 0])
ax.plot(uvals, m_us, ls="-", color='blue')
ax.fill_between(uvals, m_us - s_us, m_us + s_us, color='blue', alpha=0.2)
ax.set_xlabel("States Visited")
ax.set_ylabel("Thinking Time (ms)")
ax.set_xlim(uvals[0], uvals[-1])

# 绘制第四张子图（人类思维时间与独立状态数关系图）
with open(f"{datadir}/human_RT_and_rews_follow.bson", 'rb') as f:
    data = pickle.load(f)

keep = [i for i, RTs in enumerate(data['all_RTs']) if np.nanmean(RTs) < 690]
Nkeep = len(keep)

# 假设有指导的lognormal参数
with open(f"{datadir}/guided_lognormal_params_delta.bson", 'rb') as f:
    params = pickle.load(f)

with open(f"{datadir}unique_states_play.bson", 'rb') as f:
    data = pickle.load(f)

all_RTs, all_unique_states = data['all_RTs'], data['all_unique_states']
RTs_us_human = np.full((Nkeep, len(uvals)), np.nan)

for i_u, u in enumerate(keep):
    new_us, new_rts = [], []
    for b in range(len(all_RTs[u])):
        us = all_unique_states[u][b, :]
        inds = np.where(~np.isnan(us))[0][1:]
        rts = all_RTs[u][b, inds]
        later = params['later'][u, :]
        
        # 假设有一个函数来计算后验均值
        def later_post_mean(r):
            return calc_post_mean(r, muhat=later[0], sighat=later[1], deltahat=later[2], mode=False)
        
        rts = later_post_mean(rts)
        new_us.append(us[inds])
        new_rts.append(rts)
    
    new_us = np.concatenate(new_us)
    new_rts = np.concatenate(new_rts)
    RTs_us_human[i_u, :] = [np.mean(new_rts[new_us == uval]) for uval in uvals]

# 计算均值和标准误差
m_human = np.nanmean(RTs_us_human, axis=0)
s_human = np.nanstd(RTs_us_human, axis=0) / np.sqrt(RTs_us_human.shape[0])

# 绘制人类思维时间与独立状态数关系图
ax = fig.add_subplot(grids[1, 1])
ax.plot(uvals, m_human, 'k-')
ax.fill_between(uvals, m_human - s_human, m_human + s_human, color='black', alpha=0.2)
ax.set_xlabel("States Visited")
ax.set_ylabel("Thinking Time (ms)")
ax.set_xlim(uvals[0], uvals[-1])

# 添加标签并保存图像
ax.text(-0.18, 1.08, "A", ha="left", va="top", transform=fig.transFigure, fontweight)
# 添加标签并保存图像
ax.text(-0.18, 1.08, "A", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=12)
ax.text(0.43, 1.08, "B", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=12)
ax.text(-0.18, 0.47, "C", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=12)
ax.text(0.43, 0.47, "D", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=12)

# 保存图形为 PDF 和 PNG 格式
plt.savefig("./figs/supp_exploration.pdf", bbox_inches="tight")
plt.savefig("./figs/supp_exploration.png", bbox_inches="tight")

# 关闭图形窗口
plt.close()