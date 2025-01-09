# 本脚本考虑了不同模型规模下，奖励和回合概率在学习过程中的变化

# 导入必要的库
import numpy as np
import time
import bson
import random
from .anal_utils import seeds,greedy_actions,loaddir,datadir
from ..walls_train import recover_model,build_environment
from ..src.model import ModularModel,forward_modular
from ..src.a2c import run_episode

# 打印分析开始信息
print("分析不同模型规模下的学习和回合情况")

# 设置超参数（不计算损失）
loss_hp = {"loss1": 0, "loss2": 0, "loss3": 0, "loss4": 0}

# 设置隐藏单元数量、训练周期
Nhiddens = [60, 80, 100]  # 隐藏层单元数
epochs = np.arange(0, 1001, 50)  # 训练的epoch范围

# 创建存储结果的容器
meanrews = np.zeros((len(Nhiddens), len(seeds), len(epochs)))  # 平均奖励
pfracs = np.zeros((len(Nhiddens), len(seeds), len(epochs)))  # 回合概率

# 对每种网络大小进行循环
for ihid, Nhidden in enumerate(Nhiddens):
    for iseed, seed in enumerate(seeds):  # 对每个随机种子进行循环
        for iepoch, epoch in enumerate(epochs):  # 对每个epoch进行循环

            # 构建模型文件名
            filename = f"N{Nhidden}_T50_Lplan8_seed{seed}_{epoch}"
            
            # 加载模型参数
            network, opt, store, hps, policy, prediction = recover_model(f"{loaddir}{filename}")

            # 初始化环境和代理
            Larena = hps["Larena"]
            model_properties, wall_environment, model_eval = build_environment(
                Larena, hps["Nhidden"], hps["T"], Lplan=hps["Lplan"], greedy_actions=greedy_actions
            )
            m = ModularModel(model_properties, network, policy, prediction, forward_modular)

            random.seed(1)  # 设置随机种子，保证可复现
            batch_size = 5000  # 需要考虑的环境数量
            tic = time.time()  # 记录开始时间
            
            # 运行实验
            L, ys, rews, as_, world_states, hs = run_episode(
                m, wall_environment, loss_hp, batch=batch_size, calc_loss=False
            )

            # 计算回合概率和平均奖励
            plan_frac = np.sum(as_ == 5) / np.sum(as_ > 0.5)  # 采取回合动作的比例
            mean_rew = np.sum(rews > 0.5) / batch_size  # 平均奖励

            print(f"N={Nhidden}, seed={seed}, epoch={epoch}, avg rew={mean_rew}, rollout fraction={plan_frac}")
            
            # 存储结果
            meanrews[ihid, iseed, iepoch] = mean_rew
            pfracs[ihid, iseed, iepoch] = plan_frac

# 保存所有结果
res_dict = {
    "seeds": seeds,
    "Nhiddens": Nhiddens,
    "epochs": epochs,
    "meanrews": meanrews,
    "planfracs": pfracs
}

# save results
filename = datadir + "rew_and_plan_by_n.bson"
with open(filename, 'wb') as f:
    f.write(bson.BSON.encode({'res_dict': res_dict}))
