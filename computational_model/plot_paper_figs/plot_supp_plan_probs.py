import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import norm

# 设置绘图参数
cm = 1 / 2.54  # 厘米转英寸的比例
fig = plt.figure(figsize=(5 * cm, 3 * cm))
fsize_label = 10  # 标签字体大小
col_p1 = 'r'  # 第一种柱状图的颜色
col_p2 = 'b'  # 第二种柱状图的颜色
col_point = 'k'  # 数据点颜色
capsize = 3  # 错误条端点的大小

# 初始化数据
datadir = '数据目录'  # 需要设置数据目录路径
seeds = [1, 2, 3]  # 训练模型的种子
plan_epoch = 50  # 计划训练的周期数

# 存储成功和失败的滚动预测的前后值
all_ms = []

# 对成功和失败的滚动预测进行处理
for i in range(2):  # i=0为成功的滚动，i=1为失败的滚动
    ms = []
    
    for seed in seeds:  # 遍历每个训练模型
        # 加载数据
        with open(f"{datadir}/causal_N100_Lplan8_{seed}_{plan_epoch}.bson", 'rb') as f:
            data = pickle.load(f)
        
        # 提取前滚动和后滚动的预测概率
        p_initial_sim, p_continue_sim = data["p_initial_sim"], data["p_continue_sim"]
        
        # 找到有效数据的索引（非NaN的值）
        inds = np.where(np.isnan(np.sum(p_continue_sim, axis=0)) == False)[0]
        
        # 存储数据（前后滚动的平均概率）
        ms.append([np.mean(p_initial_sim[i, inds]), np.mean(p_continue_sim[i, inds])])

    ms = np.hstack(ms)  # 合并各个模型的数据
    all_ms.append(ms[1, :])  # 存储后滚动的数据以供后续统计
    
    # 计算均值和标准误差
    m3 = np.mean(ms, axis=1)[:2]  # 前后滚动的均值
    s3 = 2 * np.std(ms, axis=1)[:2] / np.sqrt(len(seeds))  # 标准误差
    
    # 绘制柱状图
    ax = fig.add_subplot(1, 2, i + 1)
    ax.bar([1, 2], m3, yerr=s3, color=[col_p1, col_p2][i], capsize=capsize)  # 绘制柱状图并添加误差条
    
    # 绘制每个数据点
    shifts = np.arange(1, ms.shape[1] + 1)  # 数据点的x坐标
    shifts = (shifts - np.mean(shifts)) / np.std(shifts) * 0.2  # 添加一些抖动
    ax.scatter(np.ones_like(ms[0, :]) + shifts, ms[0, :], color=col_point, marker='.', s=15, zorder=100)
    ax.scatter(np.ones_like(ms[1, :]) * 2 + shifts, ms[1, :], color=col_point, marker='.', s=15, zorder=100)
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["pre", "post"])  # x轴标签
    if i == 0:  # 成功的滚动
        ax.set_ylabel(r"$\pi(\text{rollout})$", labelpad=-1.5)
        ax.set_title("succ.", fontsize=fsize_label)
        ax.set_yticks(np.arange(0.0, 0.9, 0.2))
    else:  # 失败的滚动
        ax.set_title("unsucc.", fontsize=fsize_label)
        ax.set_yticks([])

    ax.set_ylim(0.0, 0.8)
    ax.set_xlim(0.5, 2.5)

# 添加标签并保存图形
y1 = 1.16
x1, x2 = -0.25, 0.45
plt.text(x1, y1, "A", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
plt.text(x2, y1, "B", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)

plt.savefig("./figs/supp_plan_probs.pdf", bbox_inches="tight")
plt.savefig("./figs/supp_plan_probs.png", bbox_inches="tight")
plt.close()

# 计算成功和失败滚动之间的差异并输出
delta = all_ms[1] - all_ms[0]
print("post delta:", np.mean(delta), np.std(delta) / np.sqrt(len(delta)))

# 使用高斯分布计算p值
p_value = norm.cdf(0, loc=np.mean(delta), scale=np.std(delta) / np.sqrt(len(delta)))
print("Gaussian p =", p_value)
