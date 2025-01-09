import numpy as np
import matplotlib.pyplot as plt
import pickle

# 设置绘图参数
cm = 1 / 2.54  # 厘米转英寸的比例
fig = plt.figure(figsize=(10 * cm, 3 * cm))
fsize_label = 10  # 标签字体大小
fsize_leg = 8  # 图例字体大小
cols = [
    [[0, 0, 0], [0.4, 0.4, 0.4], [0.7, 0.7, 0.7]],  # Human colors
    [[0.00, 0.19, 0.52], [0.24, 0.44, 0.77], [0.49, 0.69, 1.0]]  # Model colors
]

# 数据目录路径
datadir = '数据目录'  # 需要设置数据目录路径

# 加载数据
with open(f"{datadir}/RT_predictions_N100_Lplan8_1000.bson", 'rb') as f:
    data = pickle.load(f)

RTs_by_u = data["RTs_by_u"]
pplans_by_u = data["pplans_by_u"]
dists_by_u = data["dists_by_u"]
steps_by_u = data["steps_by_u"]

# 设置步骤和距离
xsteps = range(1, 4)  # 每个试验内的步骤
xdists = range(1, 7)  # 到目标的距离

# 创建子图
grids = fig.add_gridspec(nrows=1, ncols=2, left=0.00, right=1.00, bottom=0.0, top=1.0, wspace=0.35)

# 绘制 human 和 model 数据
for idat, dat in enumerate([RTs_by_u, pplans_by_u]):  # 对于 human 和 model
    ax = fig.add_subplot(grids[0, idat])
    for xstep in xsteps:  # 对于试验中的每个步骤
        mus = np.full((len(dat), len(xdists)), np.nan)  # 初始化数据数组
        for u in range(len(dat)):  # 对于每个模型/参与者
            for d in xdists:  # 对于每个到目标的距离
                # 相关的动作
                inds = np.where((dists_by_u[u] == d) & (steps_by_u[u] == -xstep))[0]
                if len(inds) >= 1:  # 如果满足条件的动作数量大于等于1
                    mus[u, d-1] = np.mean(dat[u][inds])  # 存储数据
        m = np.nanmean(mus, axis=0)  # 对模型/参与者取均值
        s = np.nanstd(mus, axis=0) / np.sqrt(np.sum(~np.isnan(mus), axis=0))  # 标准误差

        ax.plot(xdists, m, label=f"step = {xstep}", color=cols[idat][xstep-1])  # 绘制均值
        ax.fill_between(xdists, m-s, m+s, color=cols[idat][xstep-1], alpha=0.2)  # 绘制标准误差区域

    ax.set_xlabel("distance to goal")
    if idat == 0:
        ax.set_ylabel("thinking time")  # human 数据
    else:
        ax.set_ylabel(r"$\pi(\text{rollout})$")  # model 数据
    ax.legend(frameon=False, fontsize=fsize_leg)

# 添加标签并保存图形
y1 = 1.16
x1, x2 = -0.13, 0.45
plt.text(x1, y1, "A", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
plt.text(x2, y1, "B", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)

plt.savefig("./figs/supp_plan_by_step.pdf", bbox_inches="tight")
plt.savefig("./figs/supp_plan_by_step.png", bbox_inches="tight")
plt.close()

# 计算并打印 pi(rollout) 和 human RT 之间的残差相关性
allcors = []  # 存储相关性结果

for u in range(len(RTs_by_u)):  # 对每个用户
    mean_sub_RTs = []  # 存储均值减去后的 thinking times
    mean_sub_pplans = []  # 存储均值减去后的 rollout 概率

    for dist in range(1, 21):  # 对每个距离到目标
        for xstep in range(1, 101):  # 对每个试验步骤
            inds = (dists_by_u[u] == dist) & (steps_by_u[u] == -xstep)  # 查找对应的索引
            if np.sum(inds) >= 2:  # 至少需要 2 个数据点
                new_RTs = RTs_by_u[u][inds]
                new_pplans = pplans_by_u[u][inds]
                mean_sub_RTs.extend(new_RTs - np.mean(new_RTs))  # 减去均值后添加
                mean_sub_pplans.extend(new_pplans - np.mean(new_pplans))  # 减去均值后添加

    # 计算残差相关性
    cor = np.corrcoef(mean_sub_RTs, mean_sub_pplans)[0, 1]
    allcors.append(cor)

# 打印结果
print("mean and standard error of residual correlation:")
print(np.mean(allcors), " ", np.std(allcors) / np.sqrt(len(allcors)))
