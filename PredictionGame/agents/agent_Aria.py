import torch
import torchvision
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
class Aria(nn.Module):
    def __init__(self,opt_params):
        super(Aria, self).__init__()
        self.batch_size = opt_params["batch_size"]
        self.gamma = opt_params["gamma"]
        self.hiddenDim = 10
        self.obs_Mod = lin_Mod([2+self.hiddenDim, 5])
        self.action_Mod = lin_Mod([self.hiddenDim, 1], sftmx = True)
        self.msg_Enc = lin_Mod([4, 5], sftmx = False)
        self.msg_Dec = lin_Mod([self.hiddenDim, 4], sftmx=True)
        self.rep_Mod = lin_Mod([self.hiddenDim, self.hiddenDim])
        self.last_state = torch.zeros([1, self.hiddenDim], dtype=float)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt_params["lr"])

        self.saved_act_Logp = []
        self.saved_msg_Logp = []
        self.saved_rewards = []
        self.saved_states = []
    """def embed_Obs(self, obs):
        self.zObs = self.obs_Mod(obs) """
        
    def forward(self, obs, msg):
        inO = torch.cat([obs, self.last_state], -1)
        o = self.obs_Mod(inO)
        m = self.msg_Enc(msg)
        new_state = self.rep_Mod(torch.cat([o, m], -1))
        self.last_state=new_state
        action = self.action_Mod(new_state)
        message = self.msg_Dec(new_state)

        return action, message

    def select_action(self, obs, msg):
        obs_t = torch.tensor([obs])
        msg_t = torch.tensor([msg])
        action, message = self.forward(obs_t, msg_t)
        a_distrib = Categorical(action)
        m_distrib = Categorical(message)
        a = a_distrib.sample()
        m = m_distrib.sample()
        self.saved_act_Logp.append(a_distrib.log_prob(a))
        self.saved_msg_Logp.append(m_distrib.log_prob(m))
        return a, m

    def train_on_batch(self, state, reward):
        self.saved_obs.append(state[0])
        self.saved_downlink_msgs.append(state[1])
        self.saved_reward(reward)
        self.batch_counter += 1
        if self.batch_counter >= self.batch_size:
            
            returns = self.getReturns(normalize=True)  
            self.optimizer.zero_grad()
            obs_tensor = torch.FloatTensor(self.saved_obs[:self.batch_size])
            returns_tensor = torch.FloatTensor(returns)
            loss = -(returns_tensor*(self.saved_act_Logp+self.saved_msg_Logp)).mean()
            loss.backward()
            self.optimizer.step()

            self.saved_act_Logp = []
            self.saved_msg_Logp = []
            self.saved_rewards = []
            self.saved_states = []
            return loss.item()
        return None
    def getReturns(self, normalize=False):
        _R = 0
        Gs = []
        for r in self.saved_rewards[:self.batch_size]:
            _R = r + self.gamma * _R
            Gs.insert(0, _R)
        if normalize==True:
            return (Gs-np.mean(Gs))/np.std(Gs)
        return Gs
class lin_Mod(nn.Module):
    def __init__(self, sizes = [2, 5, 6, 10, 10], sftmx = False):
        super(lin_Mod, self).__init__()
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
        for m in self.mod:
            x_ = m(x)
        return x_