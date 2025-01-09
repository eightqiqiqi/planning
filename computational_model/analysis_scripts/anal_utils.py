# This script loads some useful libraries and sets various global defaults.
# Load necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
import torch
import torch.nn as nn
import torch.optim as optim
import bson

# Set some default paths
datadir = "./results/"  # directory to write results to
loaddir = "../models/"  # directory to load models from

# Select default global models
seeds = range(61, 66)  # random seeds
plan_epoch = 1000  # training epoch to use for evaluation (1000 is final epoch)
greedy_actions = True  # sample actions greedily at test time
N = 100  # number of units
Lplan = 8  # planning horizon
Larena = 4  # arena size
prefix = ""  # model name prefix
epoch = plan_epoch  # redundant

# Lognormal helper functions
def lognorm(x, mu=0, sig=0, delta=0):
    # PDF for shifted lognormal distribution (shift = delta)
    if x <= delta:
        return 0
    return 1 / ((x - delta) * sig * np.sqrt(2 * np.pi)) * np.exp(- (np.log(x - delta) - mu)**2 / (2 * sig**2))

def Phi(x):
    # Standard normal CDF
    return stats.norm.cdf(x)

def calc_post_mean(r, deltahat=0, muhat=0, sighat=0):
    # Compute posterior mean thinking time for a given response time 'r'
    # deltahat, muhat, and sighat are the parameters of the lognormal prior over delays
    if r < deltahat + 1:
        return 0
    k1, k2 = 0, r - deltahat  # Integration limits
    term1 = np.exp(muhat + sighat**2 / 2)
    term2 = Phi((np.log(k2) - muhat - sighat**2) / sighat) - Phi((np.log(k1) - muhat - sighat**2) / sighat)
    term3 = Phi((np.log(k2) - muhat) / sighat) - Phi((np.log(k1) - muhat) / sighat)
    post_delay = (term1 * term2 / term3 + deltahat)  # Add back delta for posterior mean delay
    return r - post_delay  # Posterior mean thinking time is response minus mean delay
