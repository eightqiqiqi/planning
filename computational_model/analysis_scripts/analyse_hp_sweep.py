# 本脚本用于在不同的网络大小和回放长度下重复一些关键分析，以评估结果的鲁棒性

# 导入一些脚本
import numpy as np
from repeat_human_actions import repeat_human_actions
from perf_by_rollout_number import run_perf_by_rollout_number
from behaviour_by_success import run_causal_rollouts

# 设置全局变量，避免默认模型下运行分析
run_default_analyses = False
# 导入分析功能

# 恢复默认分析设置
run_default_analyses = True

# print("使用不同的超参数重复分析")
print("repeating analyses with different hyperparameters")

# 设置前缀和种子
prefix = ""
seeds = range(51, 56)  # 使用一个独立的种子集合
sizes = [60, 100, 140]  # 考虑的模型大小
Lplans = [4, 8, 12]  # 考虑的规划时间长度

# 对每个网络大小和规划时间进行循环
for N in sizes:  # 遍历每个网络大小
    for Lplan in Lplans:  # 遍历每个规划时间长度
        print(f"running N={N}, L={Lplan}")
        
        # 与人类反应时间的相关性分析
        repeat_human_actions(seeds=seeds, N=N, Lplan=Lplan, epoch=0, prefix="hp_")
        
        # 随着回放次数的变化，性能分析
        run_perf_by_rollout_number(seeds=seeds, N=N, Lplan=Lplan, epoch=0, prefix="hp_")
        
        # 成功/失败回放后策略的变化分析
        run_causal_rollouts(seeds=seeds, N=N, Lplan=Lplan, epoch=0, prefix="hp_")
