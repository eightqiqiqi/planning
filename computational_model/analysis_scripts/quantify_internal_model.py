# 该脚本量化内在世界模型在训练过程中的准确性

import numpy as np
import random
import os
import bson
from .anal_utils import seeds,loaddir,greedy_actions,datadir
from ..src.a2c import zeropad_data
from ..walls_train import recover_model,build_environment
from ..src.model import ModularModel
from ..src.walls import onehot_from_state
from ..src.model import ModularModel,forward_modular
from ..src.a2c import GRUind

print("quantifying accuracy of the internal world model over training")

batch = 1000  # number of environments to consider
results = {}  # dictionary for storing results

for seed in seeds:  # for each independently trained RL agent
    results[seed] = {}  # results for this model
    for epoch in range(0, 1001, 50):  # for each training epoch

        # seed random seed for reproducibility
        np.random.seed(1)

        filename = f"N100_T50_Lplan8_seed{seed}_{epoch}"  # model to load
        network, opt, store, hps, policy, prediction = recover_model(loaddir + filename)  # load model parameters

        # initialize environment and model
        Larena = hps["Larena"]
        model_properties, environment, model_eval = build_environment(
            Larena, hps["Nhidden"], hps["T"], Lplan=hps["Lplan"], greedy_actions=greedy_actions
        )
        m = ModularModel(model_properties, network, policy, prediction, forward_modular)

        # extract some useful parameters
        ed = environment.dimensions
        Nout = m.model_properties.Nout
        Nhidden = m.model_properties.Nhidden
        T, Naction, Nstates = ed.T, ed.Naction, ed.Nstates

        # initialize reward probabilities and state
        world_state, agent_input = environment.initialize(
            np.zeros(2), np.zeros(2), batch, m.model_properties, initial_params=[]
        )
        agent_state = world_state.agent_state
        h_rnn = m.network[GRUind].cell.state0 + np.zeros((Nhidden, batch), dtype=np.float32)  # expand hidden state

        # containers for storing prediction results
        rew_preds = np.full((batch, 200), np.nan)
        state_preds = np.full((batch, 200), np.nan)
        exploit = np.zeros(batch, dtype=bool)  # are we in the exploitation phase
        iter = 1  # iteration number
        rew, old_rew = np.zeros(batch), np.zeros(batch)  # containers for storing reward information

        # iterate through RL agent/environment
        while np.any(world_state.environment_state.time < (T + 1 - 1e-2)):
            iter += 1  # update iteration number
            h_rnn, agent_output, a = m.forward(m, ed, agent_input, h_rnn)  # RNN step
            active = world_state.environment_state.time < (T + 1 - 1e-2)  # active episodes

            old_rew[:] = rew[:]  # did I get reward on previous timestep?
            # update environment given action and current state
            rew, agent_input, world_state, predictions = environment.step(
                agent_output, a, world_state, environment.dimensions, m.model_properties,
                m, h_rnn
            )

            # extract true next state and reward location
            strue = np.argmax(onehot_from_state(Larena, world_state.agent_state), axis=1)
            rtrue = np.argmax(world_state.environment_state.reward_location, axis=1)

            # calculate reward prediction accuracy
            i1, i2 = (Naction + Nstates + 2), (Naction + Nstates + 1 + Nstates)  # indices of corresponding output
            rpred = np.argmax(agent_output[i1:i2, :], axis=0)  # extract prediction output
            inds = np.where(exploit & active)[0]  # only consider exploitation
            rew_preds[inds, iter] = (rpred == rtrue)[inds].astype(np.float64)  # store binary 'success' data

            # calculate state accuracy
            i1, i2 = (Naction + 2), (Naction + 1 + Nstates)  # indices of corresponding output
            spred = np.argmax(agent_output[i1:i2, :], axis=0)  # extract prediction output
            inds = np.where((old_rew < 0.5) & active)[0]  # ignore teleportation step
            state_preds[inds, iter] = (spred == strue)[inds].astype(np.float64)  # store binary 'success' data

            exploit[old_rew > 0.5] = True  # indicate which episodes are in the exploitation phase
            rew, agent_input, a = zeropad_data(rew, agent_input, a, active)  # mask if episode is finished

        print(seed, epoch, np.nanmean(rew_preds), np.nanmean(state_preds))
        results[seed][epoch] = {"rew": np.nanmean(rew_preds), "state": np.nanmean(state_preds)}  # store result

with open(f"{datadir}/internal_model_accuracy.bson", 'wb') as f:
    f.write(bson.BSON.encode({'results': results}))
