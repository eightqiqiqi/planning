import torch
import numpy as np

def gmap(f, prms, *gss):
    gsout = {p: None for p in prms}
    return gmap_(f, gsout, *gss)

def gmap_(f, gsout, *gss):
    for ip, p in enumerate(gsout.keys()):
        gsout[p] = f(*(_getformap(gs, p) for gs in gss))
    return gsout

def _getformap(gs, p):
    g = gs.get(p, None)
    return np.zeros_like(p) if g is None else g
