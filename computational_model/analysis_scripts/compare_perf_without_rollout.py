import os
import numpy as np
import time
import pickle
import bson
from ..walls_train import recover_model,build_environment
from ..src.model import ModularModel,forward_modular
from ..src.walls_build import run_episode
from .anal_utils import seeds,datadir

# 比较带有rollouts和不带rollouts的模型性能
print("comparing performance with and without rollouts")

# 设置损失超参数（这里我们不计算损失）
loss_hp = {"param1": 0, "param2": 0, "param3": 0, "param4": 0}

# 设置标志：是否采用贪婪策略
greedy_actions = True
# 模型训练周期
epoch = "plan_epoch"  # 根据实际情况替换
# 存储结果的字典
results = {}
# 设置每批环境数量
batch_size = 50000

# 遍历每个独立训练的模型
for seed in seeds:
    results[seed] = {}  # 为每个模型初始化结果字典

    for plan in [False, True]:  # 不进行rollouts (False) 或进行rollouts (True)
        np.random.seed(1)  # 设置随机种子以保证可重复性

        # 加载模型参数
        filename = f"N100_T50_Lplan8_seed{seed}_{epoch}"
        network, opt, store, hps, policy, prediction = recover_model(os.path.join("loaddir", filename))  # 加载模型
        Larena = hps["Larena"]  # 从模型超参数中提取相关配置
        # 构建环境，决定是否允许rollouts
        model_properties, wall_environment, model_eval = build_environment(
            Larena, hps["Nhidden"], hps["T"], Lplan=hps["Lplan"], 
            greedy_actions=greedy_actions, no_planning=not plan
        )       
        # 构建模型
        m = ModularModel(model_properties, network, policy, prediction, forward_modular)
        Nstates = Larena**2  # 状态数

        # 记录时间
        tic = time.time()
        # 运行模型，在环境中让代理进行动作（并行化）
        L, ys, rews, as_, world_states, hs = run_episode(
            m, wall_environment, loss_hp, batch=batch_size, calc_loss=False
        )

        # print a brief summary
        print("\n", seed, " rollouts: ", plan)
        print(np.sum(rews > 0.5) / batch_size, " ", time.time() - tic)
        print("planning fraction: ", np.sum(as_ > 4.5) / np.sum(as_ > 0.5))
        results[seed][plan] = rews  # write result before moving on

# save results
filename = f"{datadir}/performance_with_out_planning.bson"
with open(filename, 'wb') as f:
    f.write(bson.BSON.encode({'results': results}))
