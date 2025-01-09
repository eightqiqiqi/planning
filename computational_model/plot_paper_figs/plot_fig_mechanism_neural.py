import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle

# 加载数据
datadir = "你的数据文件夹路径/"
with open(datadir + "planning_as_pg_new.bson", "rb") as f:
    res_dict = pickle.load(f)
print("使用新的数据集！")

seeds = sorted(res_dict.keys())

# PCA分析：对比隐藏状态更新的真实值与预测值
seed = 62  # 示例种子值
alphas = res_dict[seed]["jacs"]  # 真实的隐藏状态更新
actions = np.array(res_dict[seed]["sim_as"], dtype=int)  # 模拟的动作
betas = res_dict[seed]["sim_gs"]  # 预测的隐藏状态更新
betas = np.concatenate([betas[i:i+1, :, actions[i]] for i in range(len(actions))])  # 合并所有预测值

# 执行PCA分析，获取主成分
pca = PCA(n_components=3)
Zb = pca.fit_transform(betas - betas.mean(axis=0))  # 投影到主成分空间
Za = pca.transform(alphas - alphas.mean(axis=0))  # 投影到主成分空间

# 绘制PCA结果
fig = plt.figure(figsize=(17 * 2.54, 3.0 * 2.54))  # 设置图像大小
ax = fig.add_subplot(111, projection='3d')

cols = ['blue', 'orange', 'green', 'red']  # 设定不同动作的颜色
for a in range(4):  # 遍历每个动作
    meanb = Zb[:, actions == a].mean(axis=1)  # 计算该动作下预测的平均值
    meanb = meanb / np.linalg.norm(meanb)  # 归一化
    ax.plot([0, meanb[0]], [0, meanb[1]], [0, meanb[2]], linestyle='-', color=cols[a], linewidth=2)  # 绘制预测值
    ax.scatter(meanb[0], meanb[1], meanb[2], color=cols[a], s=50)  # 绘制预测的终点

    meana = Za[:, actions == a].mean(axis=1)  # 计算该动作下真实值的平均值
    meana = meana / np.linalg.norm(meana)  # 归一化
    ax.plot([0, meana[0]], [0, meana[1]], [0, meana[2]], linestyle=':', color=cols[a], linewidth=3)  # 绘制真实值

# 添加标签
ax.set_xlabel("PC 1", labelpad=-16, rotation=9)
ax.set_ylabel("PC 2", labelpad=-17, rotation=107)
ax.set_zlabel("PC 3", labelpad=-17, rotation=92)
ax.view_init(elev=35., azim=75.)
ax.legend(['$\\bf \\alpha$ (PG)', '$\\bf \\alpha$ (RNN)'], frameon=False, ncol=2, bbox_to_anchor=(0.5, 1.05), loc="upper center")

# 设置绘图参数
ax.set_xticks([])  # 不显示X轴的刻度
ax.set_yticks([])  # 不显示Y轴的刻度
ax.set_zticks([])  # 不显示Z轴的刻度

# 绘制期望与实际之间的差异（PCA空间中的角度对比）
meanspca, meanspca2 = [[] for _ in range(2)]  # 初始化存储PCA对比结果的列表
for jkey in ["jacs", "jacs_shift", "jacs_shift2"]:  # 遍历不同的键（不同的动作）
    for seed in seeds:  # 遍历所有种子值
        sim_as, sim_a2s = res_dict[seed]["sim_as"], res_dict[seed]["sim_a2s"]  # 获取模拟的动作
        jacs, gs, gs2 = res_dict[seed][jkey], res_dict[seed]["sim_gs"], res_dict[seed]["sim_gs2"]  # 获取隐藏状态更新
        
        # 计算PCA投影
        jacs, gs, gs2 = [arr - arr.mean(axis=0) for arr in [jacs, gs, gs2]]  # 去均值
        jacs, gs, gs2 = [arr / np.linalg.norm(arr, axis=1) for arr in [jacs, gs, gs2]]  # 归一化
        
        pca, pca2 = [PCA(n_components=3).fit(arr) for arr in [gs, gs2]]  # 对预测的隐藏状态更新进行PCA
        Za, Zb = pca.transform(jacs), pca.transform(gs)  # 投影到主成分空间
        Za2, Zb2 = pca2.transform(jacs), pca2.transform(gs2)  # 投影到主成分空间
        
        meanspca.append(np.mean(np.sum(Za * Zb, axis=1)))  # 计算cosine相似度
        meanspca2.append(np.mean(np.sum(Za2 * Zb2, axis=1)))  # 计算cosine相似度

# 绘制期望与实际的角度对比（PCA空间中的cosine相似度）
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
for i, res in enumerate([meanspca, meanspca2]):  # 遍历第一和第二个动作的结果
    ax = axs[i]
    ax.bar([1, 2], np.mean(res, axis=0), yerr=np.std(res, axis=0), capsize=5)  # 绘制条形图并添加误差条
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["$\\bf \\alpha$ (RNN)", "$\\bf \\alpha$ (RNN_ctrl)"])
    ax.set_ylabel("$\\cos \\theta$")
    ax.set_title(f"Action {i+1}")
plt.tight_layout()

# 绘制网络规模与回报的关系图
with open(datadir + "rew_and_plan_by_n.bson", "rb") as f:
    res_dict = pickle.load(f)

meanrews, pfracs, seeds, Nhiddens, epochs = res_dict["meanrews"], res_dict["planfracs"], res_dict["seeds"], res_dict["Nhiddens"], res_dict["epochs"]

# 只考虑第二个epoch开始的数据
i1 = 2
N = np.sum(epochs <= 2)
mms = np.mean(meanrews, axis=1)[:, 0, i1:N]  # 平均奖励
sms = np.std(meanrews, axis=1)[:, 0, i1:N] / np.sqrt(len(seeds))  # 标准误差
mps = np.mean(pfracs, axis=1)[:, 0, i1:N]  # 平均回合频率
sps = np.std(pfracs, axis=1)[:, 0, i1:N] / np.sqrt(len(seeds))  # 标准误差

fig, ax = plt.subplots(figsize=(6, 4))
ax.axhline(0.2, linestyle='-', color='grey')  # 绘制基准线
for i, Nhidden in enumerate(Nhiddens):  # 遍历每个网络规模
    ax.plot(mms[i], mps[i], label=str(Nhidden))  # 绘制网络规模与回报之间的关系
    ax.fill_between(mms[i], mps[i] - sps[i], mps[i] + sps[i], alpha=0.2)  # 绘制标准误差区域

ax.set_xlabel("Mean reward")
ax.set_ylabel("Rollout frequency")
ax.legend(title="Network size")

# 保存结果图
plt.tight_layout()
plt.savefig("./figs/fig_mechanism_neural.pdf")
plt.savefig("./figs/fig_mechanism_neural.png")
plt.close()
