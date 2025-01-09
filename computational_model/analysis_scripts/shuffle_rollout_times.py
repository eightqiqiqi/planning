import random
import numpy as np
import time
import pickle
import bson
from .anal_utils import plan_epoch,seeds,loaddir, datadir,greedy_actions
from ..walls_train import recover_model,build_environment
from ..src.model import ModularModel,forward_modular
from ..src.a2c import zeropad_data,GRUind
# 假设这些是相应的分析工具和模型接口
# 需要根据实际情况实现相关模型的加载、环境的构建等

print("comparing performance with real and shuffled rollout times")

epoch = plan_epoch  # model training epoch to use for evaluation (default to final epoch)
results = {}  # container for storing results
batch = 50000  # number of episodes to simulate

for seed in seeds:  # for each independently trained model

    plan_ts, Nact, Nplan = [], [], []  # containers for storing data
    results[seed] = {}  # dict for this model
    for shuffle in [False, True]:  # run both the non-shuffled and shuffled replays
        np.random.seed(1)  # set random seed for identical arenas across the two scenarios

        # load model parameters and create environment
        network, opt, store, hps, policy, prediction = recover_model(f"{loaddir}N100_T50_Lplan8_seed{seed}_{epoch}")
        model_properties, environment, model_eval = build_environment(
            hps["Larena"], hps["Nhidden"], hps["T"], Lplan=hps["Lplan"], greedy_actions=greedy_actions, no_planning=shuffle
        )
        m = ModularModel(model_properties, network, policy, prediction, forward_modular)  # construct model

        # extract some useful parameters
        ed = environment.dimensions
        Nhidden = m.model_properties.Nhidden
        T = ed.T

        # initialize environment
        world_state, agent_input = environment.initialize(
            np.zeros(2), np.zeros(2), batch, m.model_properties, initial_params=[]
        )
        agent_state = world_state.agent_state
        h_rnn = m.network[GRUind].cell.state0 + np.zeros((Nhidden, batch), dtype=np.float32)  # expand hidden state

        rews, as_ = [], []
        rew = np.zeros(batch)
        iter = 0
        while np.any(world_state.environment_state.time < (T + 1 - 1e-2)):  # run until completion
            iter += 1  # count iteration
            h_rnn, agent_output, a = m.forward(m, ed, agent_input, h_rnn)  # RNN step
            a[rew > 0.5] = 1.0  # no 'planning' at reward

            if shuffle:  # if we're shuffling the replay times
                for b in range(batch):  # for each episode
                    if iter in plan_ts[b]:  # this is a shuffled time
                        if rew[b] < 0.5:  # if we're not at the reward
                            a[b] = 5.0  # perform a rollout
                        else:  # if we're at the reward location, resample a new rollout iteration
                            remaining = set(range(iter + 1, Nact[b] - 3))  # set of remaining iterations
                            options = remaining - set(plan_ts[b])  # consider iterations not already planned
                            if len(options) > 0:  # if there are other iterations left
                                plan_ts[b].add(np.random.choice(list(options)))  # sample a new rollout iteration

            active = world_state.environment_state.time < (T + 1 - 1e-2)  # active episodes
            # take an environment step
            rew, agent_input, world_state, predictions = environment.step(
                agent_output, a, world_state, environment.dimensions, m.model_properties, m, h_rnn
            )
            rew, agent_input, a = zeropad_data(rew, agent_input, a, active)  # mask episodes that are finished
            rews.append(rew)
            as_.append(a)  # store rewards and actions from this iteration

        rews = np.concatenate(rews)  # combine rewards into array
        as_ = np.concatenate(as_)  # combine actions into array
        Nact = np.sum(as_ > 0.5, axis=0)  # number of actions in each episode
        Nplan = np.sum(as_ > 4.5, axis=0)  # number of plans in each episode
        plan_ts = [set(np.random.permutation(max(Nact[b] - 3, Nplan[b]))[:Nplan[b]]) for b in range(batch)]  # resample planning iterations

        # print some summary data
        print(f"\n{seed} shuffled: {shuffle}")
        print(f"reward: {np.sum(rews > 0.5) / batch}")  # reward
        print(f"rollout fraction: {np.sum(as_ > 4.5) / np.sum(as_ > 0.5)}")  # planning fraction
        results[seed][shuffle] = rews  # store the rewards for this experiment

# store result
with open(f"{datadir}/performance_shuffled_planning.bson", 'wb') as f:
    f.write(bson.BSON.encode({'results': results}))