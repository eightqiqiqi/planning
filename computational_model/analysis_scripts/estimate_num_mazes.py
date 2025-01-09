import numpy as np
import os
import pickle
import bson
from .anal_utils import Larena,N,Lplan,datadir
from ..walls_train import build_environment

# 假设必要的模型和函数已被导入
# 例如：build_environment 等

# 估算任务集所覆盖的环境空间
print("estimating the total number of possible tasks")

# 构建环境
# 假设 build_environment 是一个已定义的函数，返回环境属性、墙壁环境和模型评估
model_properties, wall_environment, model_eval = build_environment(Larena, N, 50, Lplan=Lplan)

batch = 50000  # 创建的环境数量
Nstates = Larena**2  # 唯一状态的数量（因此也是潜在奖励位置的数量）
all_Npairs, all_Nids = [], []  # 总的配对比较数和相同比较数
Nseeds = 10  # 重复10次以量化不确定性
# 进行10次独立的种子初始化
for seed in range(1, Nseeds + 1):
    np.random.seed(seed)  # 设置随机种子保证可重复性
    # 创建环境
    # 假设 wall_environment.initialize 是一个已定义的函数，返回世界状态和代理输入
    world_state, agent_input = wall_environment.initialize(np.zeros(2), np.zeros(2), batch, model_properties)
    Ws = world_state.environment_state.wall_loc  # 获取所有墙壁位置

    Npairs, Nid = 0, 0  # 初始化比较计数
    for b1 in range(batch):  # 遍历每个环境
        if b1 % 10000 == 0:
            print(f"seed {seed} of {Nseeds}, environment {b1}: {Npairs} pairwise comparisons, {Nid} identical")
        for b2 in range(b1 + 1, batch):  # 对比不同的环境
            Npairs += 1  # 增加一个配对比较
            # 判断这两个迷宫是否相同
            Nid += int(np.array_equal(Ws[:, :, b1], Ws[:, :, b2]))  # 判断是否相同

    # 计算相同迷宫的比例
    frac_id = Nid / Npairs
    print(f"fraction identical: {frac_id}")  # 相同迷宫的比例
    print(f"effective task space: {Nstates / frac_id}")  # 任务空间的有效大小（墙壁布局的反比）
    all_Npairs.append(Npairs)  # 将配对比较数加入结果列表
    all_Nids.append(Nid)  # 将相同迷宫数加入结果列表

# 保存结果供将来参考
result = {"Npairs": all_Npairs, "Nids": all_Nids}
# save results
filename = f"{datadir}/estimate_num_mazes.bson"
with open(filename, 'wb') as f:
    f.write(bson.BSON.encode({'result': result}))

# 计算有效任务空间
task_spaces = Nstates * all_Npairs / all_Nids  # effective task space for each seed
num_mazes = np.mean(task_spaces)  # mean
err = np.std(task_spaces) / np.sqrt(len(task_spaces))  # standard error
print(f"effective task space: {num_mazes}, sem: {err}")  # 打印有效任务空间和标准误差
