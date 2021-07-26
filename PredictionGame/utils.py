import numpy as np
import matplotlib.pyplot as plt

def getReturns(rewards, gamma, normalize=False):
    _R = 0
    Gs = []
    for r in rewards[]:
        _R = r + gamma * _R
        Gs.insert(0, _R)
    if normalize==True:
        return (Gs-np.mean(Gs))/np.std(Gs)
    return Gs