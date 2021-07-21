import torch
import torchvision
import torch.nn as nn
import numpy as np

class Aria(nn.Module):
    def __init__(self):
        super(Aria).__init__()

        self.obs_Mod = None
        self.action_Mod = None
        self.msg_Enc = None
        self.msg_Dec = None
        self.rep_Mod = None
        self.zObs = None
    def embed_Obs(self, obs):
        self.zObs = self.obs_Mod(obs) 
    def forward(self, msg):
        zMsg = self.msg_Enc(msg)
        z = self.rep_Mod(torch.concat((zMsg, self.zObs)))
        action = self.action_Mod(z)
        message = self.msg_Dec(z)
        return action, message
