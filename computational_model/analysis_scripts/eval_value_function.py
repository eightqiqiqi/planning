# 在此脚本中，我们分析代理在使用或不使用 rollouts 时，值函数估计的变化。

# 导入需要的库和模型
import numpy as np
import time
import bson
from utils import recover_model, build_environment, run_episode, ModularModel
from .anal_utils import plan_epoch,seeds,loaddir,datadir
from ..src.a2c import forward_modular
from ..src.loss_hyperparameters import LossHyperparameters

#print("比较使用和不使用 rollouts 的性能")
print("comparing performance with and without rollouts")

# 初始化参数
loss_hp = LossHyperparameters(0, 0, 0, 0)  # 我们不计算损失

greedy_actions = True
epoch = plan_epoch
results = {}  # 用于存储结果的字典
batch_size = 10000  # 环境的数量
data = {}

for seed in seeds:  # 对于每个独立训练的模型
    results[seed] = {}  # 当前模型的结果
    np.random.seed(1)  # 设置随机种子以确保可重复性

    filename = f"N100_T50_Lplan8_seed{seed}_{epoch}"  # 模型文件名
    print(f"\nrunning {filename}")
    network, opt, store, hps, policy, prediction = recover_model(f"{loaddir}{filename}")  # 加载模型参数
    Larena = hps["Larena"]

    # 构建环境，注意是否允许 rollouts
    model_properties, wall_environment, model_eval = build_environment(
        Larena, hps["Nhidden"], hps["T"], Lplan=hps["Lplan"], greedy_actions=greedy_actions
    )
    m = ModularModel(model_properties, network, policy, prediction, forward_modular)  # 构建模型
    Nstates = Larena ** 2

    tic = time.time()
    L, ys, rews, as_, world_states, hs = run_episode(
        m, wall_environment, loss_hp, batch=batch_size, calc_loss=False
    )  # 让代理在环境中执行动作（并行化）

    # 打印简要的总结
    print(np.sum(rews > 0.5) / batch_size, " ", time.time() - tic)
    print("planning fraction: ", np.sum(as_ > 4.5) / np.sum(as_ > 0.5))

    # 提取每一集中的时间
    ts = np.concatenate([state.environment_state.time for state in world_states])
    # 计算所有的 reward-to-go
    rew_to_go = np.sum(rews > 0.5, axis=1) - np.concatenate([np.zeros(batch_size), np.cumsum(rews[:, 1:], axis=1)], axis=1)

    # compute all value functions
    Naction = wall_environment.dimensions.Naction
    Vs = ys[Naction, :, :]
    accuracy = np.abs(Vs - rew_to_go)  # accuracy of the value function

    # 初始化 rollouts 的统计数据
    plan_nums = np.zeros_like(accuracy, dtype=int)  # rollout 迭代次数
    tot_plans = np.zeros_like(accuracy, dtype=int)  # 总的 rollout 次数
    suc_rolls = np.zeros_like(accuracy, dtype=int)  # 这是否是成功的 rollout 响应
    num_suc_rolls = np.zeros_like(accuracy, dtype=int)  # 这一序列中的成功 rollout 次数
    for b in range(batch_size):  # 对每个 episode
        plan_num, init_plan = 0, 0  # 初始化
        if np.sum(rews[b, :]) > 0.5:  # 如果至少找到一次目标
            for anum in range(np.argmax(rews[b, :] == 1) + 1, np.sum(as_[b, :] > 0.5)):  # 对每次迭代
                a = as_[b, anum]

                if a == 5 and rews[b, anum - 1] != 1:  # 如果是规划且没有获得奖励
                    plan_num += 1  # 更新当前序列的 rollout 数量
                    if plan_num == 1:  # 刚开始进行规划
                        init_plan = anum  # 记录这一序列开始的迭代
                        assert np.sum(world_states[anum].planning_state.plan_input[:, b]) < 0.5  # 确保上一次没有规划
                    else:
                        # 之前已经进行了规划，应该有规划输入
                        assert np.sum(world_states[anum].planning_state.plan_input[:, b]) > 0.5
                    plan_nums[b, anum] = plan_num - 1  # 在生成此输出之前的 rollout 数量
                    suc_rolls[b, anum] = world_states[anum].planning_state.plan_input[-1, b]  # 是否成功响应了 rollout
                else:
                    if plan_num > 0:  # 刚完成规划
                        tot_plans[b, init_plan:anum] = plan_num  # 这一序列中的总 rollout 数量
                        plan_nums[b, anum] = plan_num
                        # 确保我们刚刚进行了规划
                        assert np.sum(world_states[anum].planning_state.plan_input[:, b]) > 0.5
                        suc_rolls[b, anum] = world_states[anum].planning_state.plan_input[-1, b]  # 是否成功响应了 rollout
                        num_suc_rolls[b, init_plan:anum] = np.sum(suc_rolls[b, init_plan:anum])  # 这一序列中的成功 rollout 数量
                    plan_num = 0  # 重置规划计数器

    # 保存相关数据
    data[seed] = {
        "tot_plans": tot_plans,
        "plan_nums": plan_nums,
        "suc_rolls": suc_rolls,
        "num_suc_rolls": num_suc_rolls,
        "Vs": Vs,
        "rew_to_go": rew_to_go,
        "as": as_,
        "ts": ts
    }

# 保存结果
with open(datadir + "value_function_eval.bson", 'wb') as f:
    f.write(bson.BSON.encode({'data': data}))  # 使用 BSON 保存数据