import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import spearmanr

# 定义绘图辅助函数
def load_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def permtest(m, label):
    # 进行置换检验以检测显著性
    cval = np.corrcoef(np.arange(1, len(m) + 1), m)[0, 1]  # 计算相关性
    Niter = 10000
    ctrl = np.zeros(Niter)
    for n in range(Niter):
        ctrl[n] = np.corrcoef(np.random.permutation(len(m)) + 1, m)[0, 1]
    print(f"{label} 的相关性: {cval}, 置换p值 = {np.mean(ctrl > cval)}")

# 初始化图形
fig, axs = plt.subplots(1, 2, figsize=(10, 7))

# 加载数据
datadir = "your_data_directory"  # 这里需要替换为实际的数据目录
res_dict = load_data(f"{datadir}/rew_and_plan_by_n.bson")
meanrews = res_dict['meanrews']
pfracs = res_dict['planfracs']
seeds = res_dict['seeds']
Nhiddens = res_dict['Nhiddens']
epochs = res_dict['epochs']

# 提取奖励和计划数据
mms = np.mean(meanrews, axis=1)[:, 0, :]  # 计算各个代理的平均值
sms = np.std(meanrews, axis=1)[:, 0, :] / np.sqrt(len(seeds))  # 标准误差
mps = np.mean(pfracs, axis=1)[:, 0, :]
sps = np.std(pfracs, axis=1)[:, 0, :] / np.sqrt(len(seeds))

# 转换为每个训练周期的次数
xs = epochs * 40 * 200 / 1000000

# 绘制奖励和计划数据的图表
for ax, (m, s) in zip(axs, [(mms, sms), (mps, sps)]):
    for ihid, Nhidden in enumerate(Nhiddens):  # 针对每个网络大小
        frac = (Nhidden - min(Nhiddens)) / (max(Nhiddens) - min(Nhiddens))
        frac = (0.45 * frac + 0.76)  # 用于计算颜色渐变
        col = plt.cm.viridis(frac)  # 使用 viridis 色图来表示不同的网络大小
        ax.plot(xs, m[ihid, :], ls="-", color=col, label=Nhidden)
        ax.fill_between(xs, m[ihid, :] - s[ihid, :], m[ihid, :] + s[ihid, :], color=col, alpha=0.2)
    
    ax.set_xlabel("training episodes (x10^6)")
    if ax == axs[0]:
        ax.legend(frameon=False, fontsize=10, handlelength=1.5, handletextpad=0.5, borderpad=0.0, labelspacing=0.05)
        ax.set_ylabel("mean reward")
        ax.set_ylim(0, 9)
    else:
        ax.set_ylabel("$p$(rollout)")
        ax.set_ylim(0, 0.65)
        ax.axhline(0.2, ls="--", color="k")
    
    ax.set_xticks(np.arange(0, 9, 2))
    ax.set_xlim(0, 8)

# 加载人类数据
data_follow = load_data(f"{datadir}/human_RT_and_rews_follow.bson")
means = [np.nanmean(RTs) for RTs in data_follow["all_RTs"]]
keep = np.where(np.array(means) < 690)[0]  # 过滤掉异常用户

data = load_data(f"{datadir}/human_all_data_play.bson")
all_rews_p = data['all_rews_p']
all_RTs_p = data['all_RTs_p']
all_trial_nums_p = data['all_trial_nums_p']

params = load_data(f"{datadir}/guided_lognormal_params_delta.bson")

# 计算推理和延迟时间
all_TTs, all_DTs = [], []
for u in keep:
    rts, tnums = all_RTs_p[u], all_trial_nums_p[u]
    new_TTs, new_DTs = [np.full_like(rts, np.nan) for _ in range(2)]  # 初始化思维时间和延迟时间
    initial, later = params["initial"][u, :], params["later"][u, :]
    
    def initial_post_mean(r):
        return calc_post_mean(r, muhat=initial[0], sighat=initial[1], deltahat=initial[2], mode=False)
    
    def later_post_mean(r):
        return calc_post_mean(r, muhat=later[0], sighat=later[1], deltahat=later[2], mode=False)
    
    tnum = 1
    for ep in range(rts.shape[0]):  # 对每个episode进行遍历
        for b in range(np.sum(tnums[ep, :] > 0.5)):  # 对每个动作进行遍历
            t, rt = tnums[ep, b], rts[ep, b]
            if b > 1.5:  # 跳过第一次的动作
                if t == tnum:
                    new_TTs[ep, b] = later_post_mean(rt)
                else:
                    new_TTs[ep, b] = initial_post_mean(rt)
                new_DTs[ep, b] = rt - new_TTs[ep, b]  # 延迟时间是响应时间减去思维时间
            tnum = t
    all_TTs.append(new_TTs)
    all_DTs.append(new_DTs)

# 合并数据
rews_by_episode = np.hstack([np.nansum(rews, axis=1) for rews in all_rews_p])
RTs_by_episode = np.hstack([np.nanmedian(RTs, axis=1) for RTs in all_RTs])
TTs_by_episode = np.hstack([np.nanmedian(TTs, axis=1) for TTs in all_TTs])
DTs_by_episode = np.hstack([np.nanmedian(DTs, axis=1) for DTs in all_DTs])

# 绘制奖励图表
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(np.arange(1, len(rews_by_episode) + 1), np.mean(rews_by_episode, axis=1), label="mean reward")
ax.fill_between(np.arange(1, len(rews_by_episode) + 1), np.mean(rews_by_episode, axis=1) - np.std(rews_by_episode, axis=1), 
                np.mean(rews_by_episode, axis=1) + np.std(rews_by_episode, axis=1), alpha=0.2)

# 设置轴标签等
ax.set_xlabel("episode")
ax.set_ylabel("mean reward")
ax.set_ylim(6.5, 9.5)
ax.set_xticks(np.arange(0, 40, 12))
ax.set_xlim(0, 38)

# 保存图像
plt.savefig("./figs/supp_fig_by_size.pdf", bbox_inches="tight")
plt.savefig("./figs/supp_fig_by_size.png", bbox_inches="tight")
plt.close()

# 计算并打印前五个和最后五个episode的平均奖励和RT
mr1, mr2 = np.mean(mr[:5]), np.mean(mr[34:38])
print(f"前五个奖励的平均值: {mr1}")
print(f"后五个奖励的平均值: {mr2}")

mt = np.mean(RTs_by_episode, axis=1)
mt1, mt2 = np.mean(mt[:5]), np.mean(mt[34:38])
print(f"前五个RT的平均值: {mt1}")
print(f"后五个RT的平均值: {mt2}")

# 计算奖励和RT的比值
scale_up = mr1 * mt1 / mt2
print(f"scale up: {scale_up}")

# 打印相对时间和相对奖励
print(f"相对时间: {abs(mt2 - mt1) / mt1 * 100}")
print(f"相对奖励: {abs(mr2 - mr1) / mr1 * 100}")
