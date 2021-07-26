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


class Controller():
    def __init__(self):
        self.states = []
        self.rewards = []
        self.actions = []
        self.log_probs = []

    def insert_traj(self, state, reward, as, log_probs):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(as)
        self.log_probs.append(log_probs)

    def 