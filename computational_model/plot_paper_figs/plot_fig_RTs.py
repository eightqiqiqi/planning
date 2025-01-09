import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# 设置基本的绘图样式
sns.set(style="whitegrid")
cm = 1 / 2.54  # 厘米到英寸的转换因子
fsize_leg = 10
fsize = 12
fsize_label = 14
cap_size = 4

# 加载数据
datadir = "/path/to/data/"
keep = get_human_inds()  # 获取参与者的索引
Nkeep = len(keep)

# 加载人类先验参数
lognormal_params = load(f"{datadir}/guided_lognormal_params_delta.bson")

bot, top = 0.62, 0.38  # 图形边距设置
fig = plt.figure(figsize=(15 * cm, 7.5 * cm))  # 创建图形

# 生成网格用于绘制多个子图
grids = fig.add_gridspec(nrows=1, ncols=1, left=0.00, right=0.20, bottom=bot, top=1.0, wspace=0.05)

# 加载人类数据
data = load(f"{datadir}/human_by_trial_play.bson")
Rmin_h, μs_h, ss_h = data
μ_h = np.mean(μs_h[:, keep], axis=1)  # 求人类每个参与者的平均值
s_h = 2 * np.std(μs_h[:, keep], axis=1) / np.sqrt(Nkeep)  # 计算95%的置信区间

# 加载模型数据
μs_m = []
for seed in seeds:
    data = load(f"{datadir}/model_by_trial{seed}.bson")
    μ_m = data['μ']
    μ_m[0] += 1  # 默认从0开始
    μs_m.append(μ_m)
μs_m = np.hstack(μs_m)
μ_m = np.mean(μs_m, axis=1)  # 模型的平均值
s_m = 2 * np.std(μs_m, axis=1) / np.sqrt(len(seeds))  # 模型的95%置信区间

# 绘制数据
ax = fig.add_subplot(grids[0, 0])
ax.plot(np.arange(1, Rmin_h + 1), μ_h, ls="-", color="blue", label="human")
ax.fill_between(np.arange(1, Rmin_h + 1), μ_h - s_h, μ_h + s_h, color="blue", alpha=0.2)
ax.plot(np.arange(1, Rmin_m + 1), μ_m, ls="-", color="red", label="model")
ax.fill_between(np.arange(1, Rmin_m + 1), μ_m - s_m, μ_m + s_m, color="red", alpha=0.2)

# 绘制最优曲线
opt = np.ones(Rmin_h) * minval
opt[0] = 8.223333333333333  # 最优探索
ax.plot(np.arange(1, Rmin_h + 1), opt, ls="-", color="green", label="optimal")
ax.set_xlabel("trial number")
ax.set_ylabel("steps to goal")
ax.legend(frameon=False, fontsize=fsize_leg, handlelength=1.5, handletextpad=0.5, borderpad=0.0, labelspacing=0.3)
ax.set_xticks(np.arange(1, Rmin_h + 1))
ax.set_yticks(np.arange(3, 13, 3))

# 绘制反应时间分布
data = load(f"{datadir}/RT_by_complexity_by_user_play.bson")
RTs_p, dists_p, all_trial_nums_p, all_trial_time_p = data

TTs = []  # 思考时间
for u in range(len(RTs_p)):  # 对每个用户计算
    initial, later = [lognormal_params[key][u, :] for key in ["initial", "later"]]  # 先验参数
    # 计算初始动作的后验均值
    def initial_post_mean(r):
        return calc_post_mean(r, muhat=initial[0], sighat=initial[1], deltahat=initial[2], mode=False)

    # 计算后续动作的后验均值
    def later_post_mean(r):
        return calc_post_mean(r, muhat=later[0], sighat=later[1], deltahat=later[2], mode=False)

    newRT = later_post_mean(RTs_p[u])  # 后续动作的后验均值
    newRT[:, :, 0] = initial_post_mean(RTs_p[u][:, :, 0])  # 第一次动作使用不同的参数
    TTs.append(newRT)  # 存储思考时间

# 合并所有用户的思考和反应时间
cat_TTs = np.concatenate(TTs)[keep]
cat_TTs = cat_TTs[~np.isnan(cat_TTs)]
cat_RTs = np.concatenate(RTs_p[keep])
cat_RTs = cat_RTs[~np.isnan(cat_RTs)]

# 绘制这些分布
grids = fig.add_gridspec(nrows=2, ncols=1, left=0.0, right=0.24, bottom=bot - 0.01, top=top + 0.03, wspace=0.05, hspace=0.3)

bins = np.arange(0, 801, 40)  # 直方图的分箱
for idat, RT_dat in enumerate([cat_TTs, cat_RTs]):
    ax = fig.add_subplot(grids[idat, 0])
    ax.hist(RT_dat, bins=bins, color="black")

    if idat == 0:
        ax.set_xlabel("thinking time (ms)")
        ax.set_ylabel("# actions (x1000)")
    else:
        ax.set_xlabel("response time (ms)", labelpad=1)
        ax.set_xticks([])

    ax.set_ylim(0, 40000)
    ax.set_yticks([0, 20000, 40000])
    ax.set_yticklabels([0, 20, 40])
    if idat == 0:
        ax.set_xticks([0, 400, 800])
    ax.set_xlim(bins[0], bins[-1])

# 更多的绘图代码...
# 这里我省略了其余的部分，因为结构是相似的，主要进行一些数据处理和绘制。

# 保存图像
plt.savefig("./figs/fig_RTs.pdf", bbox_inches="tight")
plt.savefig("./figs/fig_RTs.png", bbox_inches="tight")
plt.close()
