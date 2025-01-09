import numpy as np
import os
import pickle
from pymongo import bson
from .anal_utils import datadir
from ..src.walls_baselines import dist_to_rew
from ..src.maze import maze
from ..src.walls import onehot_from_loc

# 设置常数：要比较的迷宫数量
N = 10000  # 迷宫数量

# 连接数组的函数，相当于Julia的cat3
def cat3(a, b):
    return np.concatenate((a, b), axis=2)

w_wraps = np.concatenate([maze(4, wrap=True) for _ in range(N)], axis=2)  # 生成环形迷宫
w_nowraps = np.concatenate([maze(4, wrap=False) for _ in range(N)], axis=2)  # 生成欧几里得迷宫

# 可能的目标位置
ps = onehot_from_loc(4, np.arange(1, 17))  # Julia 中的 1:16 对应 Python 中的 1 到 16

# 全到全的距离
dists_wraps = np.zeros((N, 16, 16))
dists_nowraps = np.zeros((N, 16, 16))

for i1 in range(N):  # 对每个迷宫
    for i2 in range(16):  # 对每个目标位置
        # 计算从所有起点到目标的距离
        # 注意：Python 的切片是左闭右开，因此 i2:i2+1
        dists_wraps[i1, i2, :] = dist_to_rew(ps[:, i2:i2+1], w_wraps[:, :, i1:i1+1], 4)
        dists_nowraps[i1, i2, :] = dist_to_rew(ps[:, i2:i2+1], w_nowraps[:, :, i1:i1+1], 4)

# 忽略自我距离
dists_wraps = dists_wraps[dists_wraps > 0.5]
dists_nowraps = dists_nowraps[dists_nowraps > 0.5]

dists = [dists_wraps, dists_nowraps]

# 保存数据
filename = f"{datadir}/wrap_and_nowrap_pairwise_dists.bson"
with open(filename, 'wb') as f:
    f.write(bson.BSON.encode({'dists': dists}))