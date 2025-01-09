import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import norm
import pickle
import os

# 设置全局变量
fsize = 10
fsize_leg = 8
fsize_label = 12
cm = 1 / 2.54
datadir = "../analysis_scripts/results/"
figdir = "./figs/"
lw_wall = 5
lw_arena = 1.3
linewidth = 3
npermute = 10000  # 重复测试次数
plot_points = True  # 如果数据点少于10个，则绘制单独的数据点（NN所需）
capsize = 3.5

# 设置绘图参数
plt.rc("font", size=fsize)
plt.rc("pdf", fonttype=42)
plt.rc("lines", linewidth=linewidth)
plt.rc("axes", linewidth=1)

# 关闭图形顶部和右侧的边框
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

plt.rc("font", family="sans-serif")
plt.rcParams["font.sans-serif"] = "arial"

# 设置全局颜色方案
col_h = np.array([0, 0, 0]) / 255  # 人类数据
col_p = np.array([76, 127, 210]) / 255  # RL智能体
col_p1 = col_p * 0.88  # 更暗的颜色
col_p2 = col_p + np.array([0.45, 0.35, 0.175])  # 更亮的颜色
col_c = np.array([0.6, 0.6, 0.6])  # 控制组
col_point = 0.5 * (col_c + col_h)  # 单独的数据点

# 选择全局模型
seeds = range(61, 66)
plan_epoch = 1000


# 获取人类参与者的索引
def get_human_inds():
    # 加载数据
    with open(f"{datadir}/human_RT_and_rews_follow.bson", "rb") as file:
        data = pickle.load(file)
    # 找出反应时间小于690ms的试验
    keep = [i for i, RTs in enumerate(data["all_RTs"]) if np.nanmean(RTs) < 690]
    return keep


# 绘制比较函数
def plot_comparison(ax, data, xticklabs=None, ylab="", xlab=None, col="k", col2=None, ylims=None, plot_title=None, yticks=None, rotation=0):
    if col2 is None:
        col2 = col
    niters = data.shape[0]
    m = np.nanmean(data, axis=0)
    s = np.nanstd(data, axis=0) / np.sqrt(niters)
    xs = np.arange(1, data.shape[1] + 1)

    # 绘制每个数据点
    for n in range(niters):
        ax.scatter(xs, data[n, :], color=col2, s=50, alpha=0.6, marker=".")
    
    # 绘制每条线
    for n in range(niters):
        ax.plot(xs, data[n, :], ls=":", color=col2, alpha=0.6, linewidth=linewidth * 2 / 3)

    # 绘制误差条
    ax.errorbar(xs, m, yerr=s, fmt="-", color=col, capsize=capsize)

    # 设置x轴范围和标签
    ax.set_xlim(1 - 0.5, xs[-1] + 0.5)
    ha = "center" if rotation == 0 else "right"
    ax.set_xticks(xs)
    ax.set_xticklabels(xticklabs, rotation=rotation, ha=ha, rotation_mode="anchor")
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_ylim(ylims)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.set_title(plot_title, fontsize=fsize)


# 计算后验均值（假设一个响应时间'r'）
def calc_post_mean(r, deltahat=0, muhat=0, sighat=0, mode=False):
    # 计算给定响应时间的后验均值思考时间
    if r < deltahat + 1:
        return 0.0  # 如果响应时间小于最小延迟，则返回0

    if mode:
        post_delay = deltahat + np.exp(muhat - sighat**2)
        post_delay = min(r, post_delay)  # 最多等于响应时间
        return r - post_delay

    k1, k2 = 0, r - deltahat  # 积分范围
    term1 = np.exp(muhat + sighat**2 / 2)
    term2 = norm.cdf((np.log(k2) - muhat - sighat**2) / sighat) - norm.cdf((np.log(k1) - muhat - sighat**2) / sighat)
    term3 = norm.cdf((np.log(k2) - muhat) / sighat) - norm.cdf((np.log(k1) - muhat) / sighat)
    post_delay = (term1 * term2 / term3 + deltahat)  # 添加delta，得到后验均值延迟
    return r - post_delay  # 后验均值思考时间 = 响应时间 - 后验均值延迟


# 置换检验（Permutation Test）
def permutation_test(arr1, arr2):
    # 检验arr1是否大于arr2
    rands = np.zeros(npermute)
    for n in range(npermute):
        inds = np.random.choice([True, False], len(arr1))
        b1, b2 = arr1[inds], arr2[~inds]
        rands[n] = np.nanmean(b1 - b2)
    trueval = np.nanmean(arr1 - arr2)
    return rands, trueval


# 加载实验数据
replaydir = "../../replay_analyses/"
plot_experimental_replays = False  # 默认值为False，如果没有运行这些分析

def load_exp_data(summary=False):
    # 加载实验回放数据
    # 如果'summary'为True，加载总结数据（用于补充图表）
    # 否则加载完整的实验数据集

    resdir = replaydir + "results/summary_data/" if summary else replaydir + "results/decoding/"

    # 获取文件名
    fnames = [f for f in os.listdir(resdir) if "succ" not in f]
    rnames = [f[9:-2] for f in fnames]  # 动物名字和ID（即会话）

    res_dict = {}  # 用于存储结果的字典
    for f in fnames:
        with open(f"{resdir}/{f}", "rb") as file:
            res = pickle.load(file)
        res_dict[f[9:-2]] = res  # 将结果存储到字典中

    return rnames, res_dict  # 返回会话名字和结果字典
