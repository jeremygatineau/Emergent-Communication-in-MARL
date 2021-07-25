import torch
import torchvision
import torch.nn as nn
import numpy as np

class Aria(nn.Module):
    def __init__(self, batch_size):
        super(Aria).__init__()
        self.hiddenDim = 10
        self.obs_Mod = lin_Mod([1+self.hiddenDim, 5])
        self.action_Mod = lin_Mod([self.hiddenDim, 1], sftmx = True)
        self.msg_Enc = lin_Mod([4, 5], sftmx = False)
        self.msg_Dec = lin_Mod([self.hiddenDim, 4], sftmx=True)
        self.rep_Mod = lin_Mod([self.hiddenDim, self.hiddenDim])
        self.last_state = torch.zeros([batch_size, self.hiddenDim])

    """def embed_Obs(self, obs):
        self.zObs = self.obs_Mod(obs) """
        
    def forward(self, obs, msg):
        inO = torch.concat([obs, self.last_state], -1)
        o = self.obs_Mod(inO)
        m = self.msg_Enc(msg)
        new_state = self.rep_Mod(torch.concat([o, m], -1))
        self.last_state=new_state
        action = self.action_Mod(new_state)
        message = self.msg_Dec(new_state)

        return action, message

class lin_Mod(nn.Module):
    def __init__(self, sizes = [1, 5, 6, 10, 10], sftmx = False):
        super(lin_Mod).__init__()
        self.sizes = sizes
        L = []
        
        for i, s in enumerate(sizes[:-1]):
            L.append(nn.Linear(sizes[i], sizes[i+1]))
            if i==len(sizes)-1 and sftmx==True:
                L.append(nn.Softmax())
            else :
                L.append(nn.ReLU())
        self.mod = nn.ModuleList(L)
    
    def forward(self, x):
        return self.mod(x)