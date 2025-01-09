import numpy as np
import matplotlib.pyplot as plt
import pickle

# 设置绘图参数
cm = 1 / 2.54  # 厘米转英寸的比例
fsize_label = 10  # 标签字体大小
fsize_leg = 8  # 图例字体大小
col_p = [0.00, 0.19, 0.52]  # value function 的颜色
col_c = [0.7, 0.7, 0.7]  # constant 的颜色

# 数据目录路径
datadir = '数据目录'  # 需要设置数据目录路径

# 加载数据
with open(f"{datadir}/value_function_eval.bson", 'rb') as f:
    data = pickle.load(f)

seeds = list(data.keys())  # 获取所有的种子
as_ = [data[seed]["as"] for seed in seeds]  # 获取 actions
Vs = [data[seeds[i]]["Vs"][as_[i] > 0.5] for i in range(len(seeds))]  # 获取 values
rtg = [data[seeds[i]]["rew_to_go"][as_[i] > 0.5] for i in range(len(seeds))]  # 获取 reward to go
ts = [data[seeds[i]]["ts"][as_[i] > 0.5] / 51 * 20 for i in range(len(seeds))]  # 获取 time within episode
accs = [Vs[i] - rtg[i] for i in range(len(seeds))]
all_accs = np.concatenate(accs)  # 合并所有参与者的 accuracy
all_rtg = np.concatenate(rtg)  # 合并所有参与者的 reward to go

# 获取每个动作序列的最后一个动作
all_last_as = []  # 存储最后一个动作
for a in as_:
    last_inds = np.sum(a > 0.5, axis=1)  # 找到最后一个动作的索引
    all_last_as.append(np.concatenate([a[i, last_inds[i] - 9:last_inds[i]] for i in range(a.shape[0])], axis=0))
all_last_as = np.concatenate(all_last_as)

# 绘制预测误差和时间内的误差
fig = plt.figure(figsize=(15 * cm, 7 * cm))
grids = fig.add_gridspec(nrows=1, ncols=2, left=0.1, right=0.9, bottom=0.6, top=1.0, wspace=0.4)
axs = [fig.add_subplot(grids[0, i]) for i in range(2)]

# 绘制 prediction error 的直方图
bins1 = np.arange(-7, 8, 1)
bins2 = np.arange(-7, 8, 1)
axs[0].hist(all_accs, alpha=0.5, color=col_p, bins=bins1, label="value function", zorder=10000)
axs[0].hist(np.mean(all_rtg) - all_rtg, color=col_c, alpha=0.5, bins=bins2, label="constant")
axs[0].set_xlabel("prediction error")
axs[0].set_ylabel("frequency")
axs[0].set_yticks([])
axs[0].legend(fontsize=fsize_leg, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.2), frameon=False)

# 绘制 prediction error vs time
bins = np.arange(0, 21, 1)
xs = 0.5 * (bins[:-1] + bins[1:])
res = np.zeros((len(seeds), len(xs)))

for i in range(len(seeds)):
    res[i, :] = [np.mean(np.abs(accs[i][(ts[i] > bins[j]) & (ts[i] <= bins[j + 1])])) for j in range(len(xs))]

m, s = np.mean(res, axis=0), np.std(res, axis=0) / np.sqrt(len(seeds))
axs[1].plot(xs, m, color=col_p)
axs[1].fill_between(xs, m - s, m + s, alpha=0.2, color=col_p)
axs[1].set_xlabel("time within episode (s)")
axs[1].set_ylabel("prediction error")

# 分别按 rollout 的长度进行数据分组
plan_lengths = np.arange(1, 6)  # 考虑的 rollout 长度
plan_nums = np.arange(0, plan_lengths[-1] + 1)  # rollout 数量
dkeys = ["tot_plans", "plan_nums", "suc_rolls", "num_suc_rolls", "Vs", "rew_to_go"]

# 初始化存储结果的数组
all_accs = np.zeros((len(seeds), len(plan_lengths), len(plan_nums)))
all_vals = np.zeros_like(all_accs)
all_vals0 = np.zeros_like(all_accs)
all_accs0 = np.zeros_like(all_accs)

# 处理每个种子的数据
for iseed, seed in enumerate(seeds):
    tot_plans, plan_nums, suc_rolls, num_suc_rolls, Vs, rew_to_go = [data[seed][dkey] for dkey in dkeys]
    accuracy = np.abs(Vs - rew_to_go)

    for ilength, plan_length in enumerate(plan_lengths):
        for inum, number in enumerate(np.arange(0, plan_length + 1)):
            inds = (tot_plans == plan_length) & (plan_nums == number) & (suc_rolls < 10.5)
            inds0 = inds & (num_suc_rolls < 0.5)  # 没有成功 rollout 的序列
            accs = accuracy[inds]
            accs0 = accuracy[inds0]
            vals = Vs[inds]
            vals0 = Vs[inds0]
            rtg = rew_to_go[inds]

            # 存储 rollout 的结果
            all_accs[iseed, ilength, inum] = np.mean(accs)
            all_accs0[iseed, ilength, inum] = np.mean(accs0)
            all_vals[iseed, ilength, inum] = np.mean(vals)
            all_vals0[iseed, ilength, inum] = np.mean(vals0)

# 绘制结果
cols = [[0.00, 0.09, 0.32], [0.00, 0.19, 0.52], [0.19, 0.39, 0.72], [0.34, 0.54, 0.87], [0.49, 0.69, 1.0]]
grids = fig.add_gridspec(nrows=1, ncols=4, left=0.0, right=1, bottom=0.0, top=0.35, wspace=0.6)
axs = [fig.add_subplot(grids[0, i]) for i in range(4)]

for ilength, plan_length in enumerate(plan_lengths):
    for idat, dat in enumerate([all_vals, all_vals0, all_accs, all_accs0]):  # 对每种数据类型绘图
        m = np.mean(dat[:, ilength, :], axis=0)[:plan_length + 1]
        s = np.std(dat[:, ilength, :], axis=0)[:plan_length + 1] / np.sqrt(dat.shape[0])
        xs = np.arange(0, plan_length + 1)
        axs[idat].plot(xs, m, label=(f"{plan_length} rollouts" if idat == 0 else None), color=cols[ilength])
        axs[idat].fill_between(xs, m - s, m + s, alpha=0.2, color=cols[ilength])

# 设置 y 轴标签
ylabels = ["value", "value [failed]", "error", "error [failed]"]
for i, ax in enumerate(axs):
    ax.set_ylabel(ylabels[i])

# 设置 x 轴标签
for ax in axs:
    ax.set_xlabel("rollout number")

# 添加标签并保存图像
y1, y2 = 1.07, 0.42
x1, x2, x3, x4 = -0.07, 0.195, 0.46, 0.745
plt.text(x1 + 0.13, y1, "A", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
plt.text(x3 + 0.035, y1, "B", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
plt.text(x1, y2, "C", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
plt.text(x2, y2, "D", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
plt.text(x3, y2, "E", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
plt.text(x4, y2, "F", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)

# 保存图像
plt.savefig("./figs/supp_value_function.pdf", bbox_inches="tight")
plt.savefig("./figs/supp_value_function.png", bbox_inches="tight")
plt.close()
