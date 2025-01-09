# 导入必要的模块
import matplotlib.pyplot as plt
import subprocess as sb

#### 主要图表部分 ####

# 图2：绘制响应时间图
print("\n绘制响应时间图")
# 执行绘制图形的脚本（Python 中的等效代码）
sb.run(["python", "plot_fig_RTs.py"])

# 图3：绘制RNN行为图
print("\n绘制RNN行为图")
# 执行绘制图形的脚本（Python 中的等效代码）
sb.run(["python", "plot_fig_mechanism_behav.py"])

# 图4：绘制重放图
print("\n绘制重放图")
# 执行绘制图形的脚本（Python 中的等效代码）
sb.run(["python", "plot_fig_replays.py"])

# 图5：绘制PG图
print("\n绘制PG图")
# 执行绘制图形的脚本（Python 中的等效代码）
sb.run(["python", "plot_fig_mechanism_neural.py"])

#### 附加图表部分 ####

# 图S1：绘制附加的欧几里得比较图
print("\n绘制附加的欧几里得比较图")
# 执行绘制图形的脚本（Python 中的等效代码）
sb.run(["python", "plot_supp_human_euc_comparison.py"])

# 图S2：绘制附加的人类数据图
print("\n绘制附加的人类数据图")
# 执行绘制图形的脚本（Python 中的等效代码）
sb.run(["python", "plot_supp_human_summary.py"])

# 图S3：绘制附加的学习分析图
print("\n绘制附加的学习分析图")
# 执行绘制图形的脚本（Python 中的等效代码）
sb.run(["python", "plot_supp_fig_network_size.py"])

# 图S4：绘制附加的每步响应时间图
print("\n绘制附加的每步响应时间图")
# 执行绘制图形的脚本（Python 中的等效代码）
sb.run(["python", "plot_supp_RT_by_step.py"])

# 图S5：绘制附加的超参数搜索图
print("\n绘制附加的超参数搜索图")
# 目前注释掉此图，视需要可解除注释
# sb.run(["python", "plot_supp_hp_sweep.py"])

# 图S6：绘制附加的值函数分析图
print("\n绘制附加的值函数分析图")
# 执行绘制图形的脚本（Python 中的等效代码）
sb.run(["python", "plot_supp_values.py"])

# 图S7：绘制附加的内部模型图
print("\n绘制附加的内部模型图")
# 执行绘制图形的脚本（Python 中的等效代码）
sb.run(["python", "plot_supp_internal_model.py"])

# 图S8：绘制附加的可变回滚持续时间分析图
print("\n绘制附加的可变回滚持续时间分析图")
# 执行绘制图形的脚本（Python 中的等效代码）
sb.run(["python", "plot_supp_variable.py"])

# 图S9：绘制附加的探索分析图
print("\n绘制附加的探索分析图")
# 执行绘制图形的脚本（Python 中的等效代码）
sb.run(["python", "plot_supp_exploration.py"])

# 图S13：绘制附加的重规划概率图
print("\n绘制附加的重规划概率图")
# 执行绘制图形的脚本（Python 中的等效代码）
sb.run(["python", "plot_supp_plan_probs.py"])
