import numpy as np

class LossHyperparameters:
    def __init__(self, βv: np.float32, βe: np.float32, βp: np.float32, βr: np.float32):
        self.βv = βv
        self.βe = βe
        self.βp = βp
        self.βr = βr

def LossHyperparameters(βv, βe, βp, βr):
    return LossHyperparameters(βv, βe, βp, βr)
