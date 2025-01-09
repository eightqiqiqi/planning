import numpy as np
import matplotlib.pyplot as plt
import pickle

# 设置绘图参数
cm = 1 / 2.54  # 厘米转英寸的比例
fig = plt.figure(figsize=(10 * cm, 6 * cm))
bot, top = 0, 1.0  # 图形的上下边界
fsize_label = 10  # 标签字体大小
col_p = 'b'  # 曲线颜色

# 初始化一些参数
datadir = '数据目录'  # 需要设置数据目录路径
seeds = [1, 2, 3]  # 训练模型的种子
plan_epoch = 50  # 计划训练的周期数

# 加载数据：内部模型准确性数据
with open(f"{datadir}/internal_model_accuracy.bson", 'rb') as f:
    results = pickle.load(f)

epochs = []  # 存储训练周期
rews, states = [], []  # 存储奖励预测和状态预测的准确性

# 遍历每个训练好的模型
for seed in seeds:
    epochs_seed = sorted([k for k in results[seed].keys()])  # 获取每个模型的周期点
    epochs_seed = [e for e in epochs_seed if e <= plan_epoch]  # 选择小于等于计划周期的训练周期
    epochs.extend(epochs_seed)
    rews.append([results[seed][e]["rew"] for e in epochs_seed])  # 奖励预测准确性
    states.append([results[seed][e]["state"] for e in epochs_seed])  # 状态预测准确性

# 对模型的数据进行拼接
rews = np.hstack(rews)
states = np.hstack(states)

# 计算奖励和状态的均值及标准误差
mrs = np.mean(rews, axis=1)  # 奖励预测的均值
mss = np.mean(states, axis=1)  # 状态预测的均值
srs = np.std(rews, axis=1) / np.sqrt(len(seeds))  # 奖励预测的标准误差
sss = np.std(states, axis=1) / np.sqrt(len(seeds))  # 状态预测的标准误差

# 转换训练周期为每百万次训练的训练集大小
xs = np.array(epochs) * 40 * 200 / 1000000

# 绘制状态预测和奖励预测的图形
grids = fig.add_gridspec(nrows=2, ncols=2, left=0.00, right=1.00, bottom=0.0, top=1.0, wspace=0.5, hspace=0.30)
for idat, (dat, label) in enumerate([(mss, sss), (mrs, srs)]):
    for irange in range(2):  # 两种y轴范围：全范围和缩放范围
        ax = fig.add_subplot(grids[irange, idat])
        m, s = dat, label  # 提取均值和标准误差
        ax.plot(xs, m, ls="-", color=col_p)  # 绘制均值曲线
        ax.fill_between(xs, m - s, m + s, color=col_p, alpha=0.2)  # 绘制标准误差带

        if irange == 1:  # 缩放的y轴
            ax.set_xlabel("training episodes (x" + r"$10^6$" + ")")
            ax.set_xticks(np.arange(0, 9, 2))
            ax.set_ylim(0.99, 1.0002)
            ax.set_yticks([0.99, 1.0])
            ax.set_yticklabels(["0.99", "1.0"])
        else:  # 全范围
            ax.set_xticks([])
            ax.set_ylim(0.0, 1.02)
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.set_yticklabels(["0.00", "0.50", "1.00"])

        # 设置y轴标签
        if idat == 0:
            ax.set_ylabel("state prediction")
        else:
            ax.set_ylabel("reward prediction")

        ax.set_xlim(0, 8)

# 添加标签
y1 = 1.12
x1, x2 = -0.13, 0.46
plt.text(x1, y1, "A", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
plt.text(x2, y1, "B", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)

# 保存图形
plt.savefig(f"./figs/supp_internal_model.pdf", bbox_inches="tight")
plt.savefig(f"./figs/supp_internal_model.png", bbox_inches="tight")
plt.close()
