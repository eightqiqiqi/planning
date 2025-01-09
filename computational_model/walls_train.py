import os
import argparse
import torch
import torch.optim 
from torch.optim import Adam
import time
import random
import numpy as np
import logging
from torch.optim import Adam
from functools import reduce

from src.ToPlanOrNotToPlan import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def parse_commandline():
    parser = argparse.ArgumentParser(description="RL Model Training")

    parser.add_argument("--Nhidden", type=int, default=100, help="Number of hidden units")
    parser.add_argument("--Larena", type=int, default=4, help="Arena size (per side)")
    parser.add_argument("--T", type=int, default=50, help="Number of timesteps per episode")
    parser.add_argument("--Lplan", type=int, default=8, help="Maximum planning horizon")
    parser.add_argument("--load", type=bool, default=False, help="Load previous model")
    parser.add_argument("--load_epoch", type=int, default=0, help="Epoch to load")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="./", help="Directory to save models")
    parser.add_argument("--beta_p", type=float, default=0.5, help="Relative importance of predictive loss")
    parser.add_argument("--prefix", type=str, default="", help="Prefix for model name")
    parser.add_argument("--load_fname", type=str, default="", help="Model file to load")
    parser.add_argument("--n_epochs", type=int, default=1001, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=40, help="Batch size")
    parser.add_argument("--lrate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--constant_rollout_time", type=bool, default=True, help="Constant rollout time")

    return parser.parse_args()

def main():

    ##### global parameters #####
    args = parse_commandline()
    print(args)

    # extract command line arguments
    Larena = args.Larena
    Lplan = args.Lplan
    Nhidden = args.Nhidden
    T = args.T
    load = args.load
    seed = args.seed
    save_dir = args.save_dir
    prefix = args.prefix
    load_fname = args.load_fname
    n_epochs = int(args.n_epochs)
    βp = float(args.beta_p)
    batch_size = int(args.batch_size)
    lrate = float(args.lrate)
    constant_rollout_time = bool(args.constant_rollout_time)

    os.makedirs(save_dir, exist_ok=True)
    random.seed(seed)  # set random seed

    loss_hp = {
        'βp': βp,
        'βv': 0.05,
        'βe': 0.05,
        'βr': 1.0
    }
    

    # build RL environment
    model_properties, wall_environment, model_eval = build_environment(
        Larena, Nhidden, T, Lplan=Lplan, constant_rollout_time=constant_rollout_time
    )
    # build RL agent
    m = build_model(model_properties, 5)
    # construct summary string
    mod_name = create_model_name(Nhidden, T, seed, Lplan, prefix=prefix)

    # training parameters
    n_batches, save_every = 200, 50
    opt = optim.Adam(m.parameters(), lr=lrate)

    # used to keep track of progress
    rews, preds = [], []
    epoch = 0  # start at epoch 0
    print("model name", mod_name)
    print("training info", n_epochs, n_batches, batch_size)

    if load:  # if we load a previous model
        if load_fname == "":  # filename not specified; fall back to default
            fname = os.path.join(save_dir, "models", f"{mod_name}_{args.load_epoch}")
        else:  # load specific model
            fname = os.path.join(save_dir, "models", load_fname)
        
        # load the parameters and initialize model
        network, opt, store, hps, policy, prediction = recover_model(fname)
        m = ModularModel(model_properties, network, policy, prediction, forward_modular)

        # load the learning curve from the previous model
        rews, preds = store[0], store[1]
        epoch = len(rews)  # start where we were
        if load_fname != "":  # loaded pretrained model; reset optimizer
            opt = Adam(m.parameters(), lr=lrate)

    prms = list(m.parameters())  # model parameters
    print("parameter length:", len(prms))
    for p in prms:
        print(p.size())

    Nthread = torch.get_num_threads()  # number of threads available
    multithread = Nthread > 1  # multithread if we can
    print("multithreading", Nthread)
    thread_batch_size = int(np.ceil(batch_size / Nthread))  # distribute batch evenly across threads
    # construct function without arguments for Flux
    closure = lambda: model_loss(m, wall_environment, loss_hp, thread_batch_size) / Nthread

    def gmap_grads(g1, g2):
        return [g1[i] + g2[i] for i in range(len(g1))]  # define map function for reducing gradients

    # function for training on a single match
    def loop(batch, closure):
        if multithread:  # distribute across threads?
            all_gs = [None] * Nthread  # vector of gradients for each thread
            for i in range(Nthread):  # on each thread
                rand_roll = np.random.rand(100 * batch * i)  # run through some random numbers
                gs = torch.autograd.grad(closure(), prms)  # compute gradient
                all_gs[i] = gs  # save gradient
            gs = reduce(gmap_grads, all_gs)  # sum across our gradients
        else:
            gs = torch.autograd.grad(closure(), prms)  # if we're not multithreading, just compute a simple gradient
        return opt.step()  # update model parameters

    t0 = time.time()  # wallclock time
    while epoch < n_epochs:
        epoch += 1  # count epochs
        # flush stdout
        print()

        Rmean, pred, mean_a, first_rew = model_eval(
            m, batch_size, loss_hp
        )  # evaluate performance
        m.reset()  # reset model

        if (epoch - 1) % save_every == 0:  # occasionally save our model
            os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)
            filename = os.path.join(save_dir, "models", f"{mod_name}_{epoch - 1}")
            store = [rews, preds]
            save_model(m, store, opt, filename, wall_environment, loss_hp, Lplan=Lplan)

        # print progress
        elapsed_time = round(time.time() - t0, 1)
        print(f"progress: epoch={epoch} t={elapsed_time} R={Rmean} pred={pred} plan={mean_a} first={first_rew}")
        rews.append(Rmean)
        preds.append(pred)
        plot_progress(rews, preds)  # plot progress

        for batch in range(1, n_batches + 1):  # for each batch
            loop(batch, closure)  # perform an update step

    m.reset()  # reset model state
    # save model
    filename = os.path.join(save_dir, "results", mod_name)
    store = [rews, preds]
    return save_model(m, store, opt, filename, wall_environment, loss_hp, Lplan=Lplan)

if __name__ == "__main__":
    main()