# 本脚本用于使用不同的网络大小和回放长度重复一些关键分析，以评估结果的鲁棒性

# 导入必要的脚本
import numpy as np

# 设置全局变量，避免默认模型下运行分析
run_default_analyses = False
# 导入分析功能
from repeat_human_actions import repeat_human_actions
from perf_by_rollout_number import run_perf_by_rollout_number
from behaviour_by_success import run_causal_rollouts
from .anal_utils import epoch

# 恢复默认分析设置
run_default_analyses = True

#print("使用不同的超参数重复分析")
print("repeating analyses with different hyperparameters")

# 设置前缀和种子
prefix = "variable_"
seeds = range(61, 66)  # 种子范围
N, Lplan = 100, 8  # 网络大小和规划时间

print(f"running N={N}, L={Lplan}")
# correlation with human RT ####
repeat_human_actions(seeds=seeds, N=N, Lplan=Lplan, epoch=epoch, prefix=prefix, model_prefix=prefix)

# change in performance with replay number ###
run_perf_by_rollout_number(seeds=seeds, N=N, Lplan=Lplan, epoch=epoch, prefix=prefix, model_prefix=prefix)

# change in policy after successful/unsuccessful replay ####
run_causal_rollouts(seeds=seeds, N=N, Lplan=Lplan, epoch=epoch, prefix=prefix, model_prefix=prefix)