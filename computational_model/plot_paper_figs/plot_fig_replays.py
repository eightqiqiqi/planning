import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

# 设置图形的风格
sns.set(style="whitegrid")

# 数据加载
def load_exp_data():
    # 假设这个函数加载实验数据（这里需要具体的文件路径和数据结构）
    # 目前使用占位的方式
    return rnames, exp_dict

# 模型数据加载
def load_model_data():
    # 加载模型数据（使用预定义的路径）
    with open("model_replay_analyses.bson", "rb") as f:
        res_dict = pickle.load(f)  # 假设数据存储为pickle格式
    return res_dict

# 设置绘图参数
bot, top = 0.62, 0.38  # 顶行和底行的边界
fig = plt.figure(figsize=(15 * 2.54, 7.5 * 2.54))  # 创建图形，单位为厘米

# 模型和实验数据加载
plot_experimental_replays = True
if plot_experimental_replays:
    rnames, exp_dict = load_exp_data()

# 加载模型数据
res_dict = load_model_data()
mod_dict = res_dict
seeds = sorted([k for k in mod_dict.keys()])  # 排序种子
Nseed = len(seeds)  # 模型数量

# 绘制实验重放示例
if plot_experimental_replays:
    example_rep = pickle.load(open("replaydir/figs/replay/widloski_Billy3_20181207_s1_examples.p", "rb"))
    reps = example_rep["reps"]
    walls = example_rep["walls"]
    home = example_rep["home"]

    # 创建网格
    ax = fig.add_subplot(111)
    # 绘制场地
    for x in np.arange(0, 5, 1) - 0.5:
        ax.plot([x, x], [-0.5, 4.5], color="k", lw=0.8)
        ax.plot([-0.5, 4.5], [x, x], color="k", lw=0.8)

    for x in range(1, 4):
        for y in range(1, 4):
            ax.scatter(x, y, s=200, marker=".", color="k")  # 画井
    ax.axis("off")

    for w in walls:  # 绘制墙壁
        ax.plot(w[:, 0], w[:, 1], color="k", lw=0.8)

    # 绘制重放轨迹
    cols = ["red", "blue"]  # 颜色
    labels = ["successful", "unsuccessful"]  # 标签
    lw_rep = 2
    for irep, rep in enumerate(reps):
        rep = rep + 0.7 * (irep / (len(reps) + 1) - 0.5)  # 调整位置
        ax.plot(rep[:, 0], rep[:, 1], color=cols[irep], label=labels[irep], lw=lw_rep, zorder=800)  # 绘制重放轨迹
        ax.scatter(rep[0, 0], rep[0, 1], color=cols[irep], s=150, zorder=1000, marker=".")  # 初始位置

    ax.plot(home[0], home[1], color="k", marker="x", markersize=12 * 0.8, ls="", mew=3)  # 绘制起始点
    ax.legend(frameon=False, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, 1.2), handlelength=1.5)

# 绘制穿墙的重放频率
grids = fig.add_gridspec(nrows=1, ncols=2, left=0.35, right=0.60, bottom=bot, top=1.0, wspace=0.15)
ylims = [0, 0.45]  # y轴范围

if plot_experimental_replays:
    batch_wall_probs = [exp_dict[n]["wall_crossings"][0] for n in rnames]  # 穿墙的概率
    batch_rand_wall_probs = [np.mean(exp_dict[n]["wall_crossings"][1:]) for n in rnames]  # 控制条件下穿墙的概率
    inds = [i for i, (bw, brw) in enumerate(zip(batch_wall_probs, batch_rand_wall_probs)) if bw > 0 or brw > 0]  # 过滤没有穿墙的会话
    batch_wall_probs = [batch_wall_probs[i] for i in inds]
    batch_rand_wall_probs = [batch_rand_wall_probs[i] for i in inds]

    # 计算平均值和标准误
    μ, s = np.mean(batch_wall_probs), np.std(batch_wall_probs) / np.sqrt(len(batch_wall_probs))  # 实际数据
    μr, sr = np.mean(batch_rand_wall_probs), np.std(batch_rand_wall_probs) / np.sqrt(len(batch_rand_wall_probs))  # 控制数据

    # 绘制实验数据
    ax = fig.add_subplot(grids[0, 0])
    ax.bar([0, 1], [μ, μr], yerr=[s, sr], color="green", capsize=5)
    ax.set_ylim(ylims)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["true", "ctrl"])
    ax.set_ylabel("$p$ (cross wall)")
    ax.set_title("experiment")

# 重复分析，使用计算模型数据
batch_wall_probs = [np.mean(mod_dict[seed]["batch_wall_probs"]) for seed in seeds]
batch_rand_wall_probs = [np.mean(mod_dict[seed]["batch_rand_wall_probs"]) for seed in seeds]

# 计算平均值和标准误
μ, s = np.mean(batch_wall_probs), np.std(batch_wall_probs) / np.sqrt(Nseed)  # 真实数据
μr, sr = np.mean(batch_rand_wall_probs), np.std(batch_rand_wall_probs) / np.sqrt(Nseed)  # 控制数据

# 绘制模型数据
ax = fig.add_subplot(grids[0, 1])
ax.bar([0, 1], [μ, μr], yerr=[s, sr], color="blue", capsize=5)
ax.set_ylim(ylims)
ax.set_xticks([0, 1])
ax.set_xticklabels(["true", "ctrl"])
ax.set_title("model")

# 其它绘制过程如上述步骤，继续添加对比和图像绘制
# 注意：本翻译为简化版，具体细节需要根据实际数据和需求进行调整
plt.show()
