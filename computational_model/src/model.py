import torch
import torch.nn as nn
import torch.optim as optim
from .a2c import forward_modular

class ModelProperties:
    def __init__(self, Nout, Nhidden, Nin, Lplan, greedy_actions, no_planning=False):
        self.Nout = Nout
        self.Nhidden = Nhidden
        self.Nin = Nin
        self.Lplan = Lplan
        self.greedy_actions = greedy_actions
        self.no_planning = no_planning

class ModularModel(nn.Module): 
    def __init__(self, model_properties, network, policy, prediction, forward):
        super(ModularModel, self).__init__()
        self.model_properties = model_properties
        self.network = network
        self.policy = policy
        self.prediction = prediction
        self.forward = forward

    def forward(self, *args, **kwargs):
        return self.forward_function(self, *args, **kwargs)
    
def ModelProperties_func(Nout, Nhidden, Nin, Lplan, greedy_actions, no_planning=False):
    return ModelProperties(Nout, Nhidden, Nin, Lplan, greedy_actions, no_planning)

def Modular_model(mp, Naction, Nstates=None, neighbor=False):
    # Define our model
    network = torch.nn.GRU(mp.Nin, mp.Nhidden)
    policy = torch.nn.Sequential(torch.nn.Linear(mp.Nhidden, Naction + 1))  # policy and value function
    Npred_out = mp.Nout - Naction - 1
    prediction = torch.nn.Sequential(
        torch.nn.Linear(mp.Nhidden + Naction, Npred_out),
        torch.nn.ReLU(),
        torch.nn.Linear(Npred_out, Npred_out)
    )
    return ModularModel(mp, network, policy, prediction, forward_modular)


def build_model(mp, Naction):
    return Modular_model(mp, Naction)

def create_model_name(Nhidden, T, seed, Lplan, prefix=""):
    # Define some useful model name
    mod_name = (
        f"{prefix}"
        f"N{Nhidden}"
        f"_T{T}"
        f"_Lplan{Lplan}"
        f"_seed{seed}"
    )
    return mod_name

