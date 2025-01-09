import numpy as np

def U_prior(state, Naction):
    # Uniform prior (same as the original in Julia)
    batch = state.shape[1]  # assuming state is a 2D numpy array where the second dimension is the batch size
    return np.ones((Naction, batch), dtype=np.float32) / Naction

def prior_loss(agent_output, state, active, mod):
    # Return -KL[q || p] (KL divergence regularization with uniform prior)
    act = active.astype(np.float32).reshape(-1, 1)  # Assuming active is a 1D array, reshape it to (N, 1)
    Naction = len(mod['policy'][0]['bias']) - 1  # Assuming mod is a dictionary, and policy is a list of layers
    logp = np.log(U_prior(state, Naction))  # KL regularization with uniform prior
    logπ = agent_output[:Naction, :]  # Assuming agent_output is a 2D array
    if mod['model_properties']['no_planning']:
        logπ = logπ[:Naction-1, :] - np.log(np.sum(np.exp(logπ[:Naction-1, :]), axis=0, keepdims=True))
        logp = logp[:Naction-1, :] - np.log(np.sum(np.exp(logp[:Naction-1, :]), axis=0, keepdims=True))
    logp = logp * act
    logπ = logπ * act
    lprior = np.sum(np.exp(logπ) * (logp - logπ))  # KL divergence term (negative)
    
    return lprior
