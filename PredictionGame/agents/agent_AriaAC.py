import torch
import torchvision
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import torch.nn.utils as utils
from torch.autograd import Variable
class AriaAC:
    def __init__(self,opt_params, with_memory=True, split= False, aidi=None):
        self.aidi = aidi
        self.batch_size = opt_params["batch_size"]
        self.gamma = opt_params["gamma"]
        self.vocabulary_size = opt_params["vocab_size"]
        self.memory_size = opt_params["memory_size"]
        self.hidden_dim = opt_params["hidden_size"]
        self.gc = opt_params["grad_clamp"]
        self.eps = np.finfo(np.float32).eps.item()
        self.with_memory = with_memory
        self.split = split

        if split: 
            self.modT = ariaModelSplit(batch_size=self.batch_size, vocabulary_size=self.vocabulary_size, memory_size=self.memory_size, hidden_dim=self.hidden_dim, with_memory=self.with_memory).float().train()
        #self.modI = ariaModel(batch_size=self.batch_size, vocabulary_size=self.vocabulary_size, memory_size=self.memory_size ,with_memory=self.with_memory).float().eval()
        else:
            self.modT = ariaModel(batch_size=self.batch_size, vocabulary_size=self.vocabulary_size, memory_size=self.memory_size, hidden_dim=self.hidden_dim, with_memory=self.with_memory).float().train()
        if self.with_memory:
            if split:
                self.hid_states = [[torch.zeros(1, 2*self.memory_size).detach(), torch.zeros(1, 2*self.memory_size).detach()]]
            else:
                self.hid_states = [torch.zeros(1, 2*self.memory_size).detach()]
        else:
            self.hid_states = None

        self.optimizer = torch.optim.Adam(self.modT.parameters(), lr=opt_params["lr"])
        
        self.saved_a_lp = torch.zeros(self.batch_size)
        self.saved_m_lp = torch.zeros(self.batch_size)
        self.saved_rewards = torch.zeros(self.batch_size)
        self.saved_values = torch.zeros(self.batch_size)
        self.saved_entropies = torch.zeros(self.batch_size)

    
    def select_actionTraing(self, obs, msg, bt):
        obs_t = torch.tensor([obs], dtype=torch.float32)
        msg_t = torch.tensor([msg], dtype=torch.float32)
        
        if self.with_memory:
            action, message, hid_state, value = self.modT.forward(obs_t.float(), msg_t.float(), self.hid_states[-1])
        else:
            action, message, _, value = self.modT.forward(obs_t.float(), msg_t.float(), None)
        
        a_distrib = Categorical(torch.cat([action, 1-action], -1))
        m_distrib = Categorical(message)
        a = a_distrib.sample()
        m = m_distrib.sample()
        a_entropy = a_distrib.entropy() 
        m_entropy = m_distrib.entropy()
        self.saved_a_lp[bt] = a_distrib.log_prob(a)
        self.saved_m_lp[bt] = m_distrib.log_prob(m)
        self.saved_entropies[bt] = a_entropy+m_entropy
        if self.with_memory:
            self.hid_states.append(hid_state)
        self.saved_values[bt] = value
        return a, m, torch.cat([action, 1-action], -1), message
    
    def train_online(self, rewards):
        returns = self.getReturns(rewards)
        #adv = rewards[:-1]-self.saved_values[:-1]+self.gamma*self.saved_values[1:] # TD error
        #policy_loss = -(self.saved_a_lp[:-1] + self.saved_m_lp[:-1])*adv.detach()
        #value_loss =  nn.functional.smooth_l1_loss(rewards[:-1]+self.gamma*self.saved_values[1:], self.saved_values[:-1], reduction='none')# adv.pow(2)

        adv = returns-self.saved_values # TD error with returns, should be better cause no bias
        policy_loss = -(self.saved_a_lp + self.saved_m_lp)*adv.detach()
        value_loss =  nn.functional.smooth_l1_loss(returns, self.saved_values, reduction='none')# adv.pow(2)

        entropy_loss = -self.saved_entropies.mean()
        loss = policy_loss.mean() + value_loss.mean() + self.eps*entropy_loss
        loss.backward(retain_graph=True)
        if self.gc is not None:
            self.grad_clamp(self.modT.parameters(), self.gc)
        self.optimizer.step()

        # Reset buffers after training on batch
        if self.with_memory:
            if self.split:
                self.hid_states = [[self.hid_states[-1][0].detach(), self.hid_states[-1][1].detach()]]
            else:
                self.hid_states = [self.hid_states[-1].detach()]
        self.saved_a_lp = torch.zeros(self.batch_size, dtype=torch.float32)
        self.saved_m_lp = torch.zeros(self.batch_size, dtype=torch.float32)
        self.saved_rewards = torch.zeros(self.batch_size, dtype=torch.float32)
        self.saved_values = torch.zeros(self.batch_size, dtype=torch.float32)
        self.saved_entropies = torch.zeros(self.batch_size, dtype=torch.float32)
        
        return policy_loss, value_loss, entropy_loss

    def getReturns(self, rewards, normalize=False):
        _R = 0
        Gs = []
        for r in rewards:
            _R = r + self.gamma * _R
            Gs.insert(0, _R)
        if normalize==True:
            return (Gs-np.mean(Gs))/(np.std(Gs)+self.eps)
        return Gs
    def grad_clamp(self, parameters, clip=0.1):
        for p in parameters:
            if p.grad is not None:
                p.grad.clamp_(min=-clip)
                p.grad.clamp_(max=clip)
class ariaModel(nn.Module):
    def __init__(self, batch_size, vocabulary_size, hidden_dim=10, memory_size=8, with_memory=True):
        super(ariaModel, self).__init__()
        self.hiddenDim = hidden_dim
        self.memory_size = memory_size
        self.with_memory = with_memory
        self.batch_size = batch_size
        self.vocabulary_size = vocabulary_size
        self.obs_Mod = nn.Sequential(nn.Linear(2, self.hiddenDim//2), nn.ReLU())
        self.msg_Enc = nn.Sequential(nn.Linear(self.vocabulary_size, self.hiddenDim//2), nn.ReLU(), nn.Linear(self.hiddenDim//2, self.hiddenDim//2), nn.ReLU())
        self.rep_Mod = nn.Sequential(nn.Linear(self.hiddenDim, self.hiddenDim), nn.ReLU())
        if self.with_memory:
            self.memory = nn.LSTMCell(self.hiddenDim, self.memory_size)
            self.action_Mod = nn.Sequential(nn.Linear(self.memory_size, 1), nn.Sigmoid())
            self.msg_Dec = nn.Sequential(nn.Linear(self.memory_size, self.memory_size), nn.Linear(self.memory_size, self.vocabulary_size), nn.Softmax(dim=-1))
            self.value_head = nn.Sequential(nn.Linear(self.memory_size, self.memory_size), nn.Linear(self.memory_size, 1))
        else : 
            self.memory = None
            self.action_Mod = nn.Sequential(nn.Linear(self.hiddenDim, 1), nn.Sigmoid())
            self.msg_Dec = nn.Sequential(nn.Linear(self.hiddenDim, self.hiddenDim), nn.Linear(self.hiddenDim, self.vocabulary_size), nn.Softmax(dim=-1))
            self.value_head = nn.Sequential(nn.Linear(self.hiddenDim, self.hiddenDim), nn.Linear(self.hiddenDim, 1))
       
    def forward(self, obs, msg, memory):
        o = self.obs_Mod(obs)
        m = self.msg_Enc(msg)
        z = self.rep_Mod(torch.cat([o, m], -1))
        if self.with_memory:
            hz, cz = self.memory(z, (memory[:, :self.memory_size], memory[:, self.memory_size:]))
            
            out_memory = torch.cat([hz, cz], dim=1)
        else:
            hz = z
            out_memory = None
        action = self.action_Mod(hz)
        message = self.msg_Dec(hz)
        value = self.value_head(hz)
        return action, message, out_memory, value

class ariaModelSplit(nn.Module):
    def __init__(self, batch_size, vocabulary_size, hidden_dim=10, memory_size=8, with_memory=True):
        super(ariaModelSplit, self).__init__()
        self.hiddenDim = hidden_dim
        self.memory_size = memory_size
        self.with_memory = with_memory
        self.batch_size = batch_size
        self.vocabulary_size = vocabulary_size
        self.obs_Mod_A = nn.Sequential(nn.Linear(2, self.hiddenDim//2), nn.ReLU())
        self.msg_Enc_A = nn.Sequential(nn.Linear(self.vocabulary_size, self.hiddenDim//2), nn.ReLU(), nn.Linear(self.hiddenDim//2, self.hiddenDim//2), nn.ReLU())
        self.rep_Mod_A = nn.Sequential(nn.Linear(self.hiddenDim, self.hiddenDim), nn.ReLU())
        self.obs_Mod_M = nn.Sequential(nn.Linear(2, self.hiddenDim//2), nn.ReLU())
        self.msg_Enc_M = nn.Sequential(nn.Linear(self.vocabulary_size, self.hiddenDim//2), nn.ReLU(), nn.Linear(self.hiddenDim//2, self.hiddenDim//2), nn.ReLU())
        self.rep_Mod_M = nn.Sequential(nn.Linear(self.hiddenDim, self.hiddenDim), nn.ReLU(), nn.Linear(self.hiddenDim, self.hiddenDim), nn.ReLU(), nn.Linear(self.hiddenDim, self.hiddenDim), nn.ReLU(), nn.Linear(self.hiddenDim, self.hiddenDim), nn.ReLU())
        if self.with_memory:
            self.memory_A = nn.LSTMCell(self.hiddenDim, self.memory_size)
            self.memory_M = nn.LSTMCell(self.hiddenDim, self.memory_size)

            self.action_Mod = nn.Sequential(nn.Linear(self.memory_size, 1), nn.Sigmoid())
            self.msg_Dec = nn.Sequential(nn.Linear(self.memory_size, self.memory_size), nn.Linear(self.memory_size, self.vocabulary_size), nn.Softmax(dim=-1))
            self.value_head = nn.Sequential(nn.Linear(self.memory_size, self.memory_size), nn.Linear(self.memory_size, 1))
        else : 
            self.memory_A = None
            self.memory_M = None
            self.action_Mod = nn.Sequential(nn.Linear(self.hiddenDim, 1), nn.Sigmoid())
            self.msg_Dec = nn.Sequential(nn.Linear(self.hiddenDim, self.hiddenDim), nn.Linear(self.hiddenDim, self.vocabulary_size), nn.Softmax(dim=-1))
            self.value_head = nn.Sequential(nn.Linear(self.hiddenDim, self.hiddenDim), nn.Linear(self.hiddenDim, 1))
       
    def forward(self, obs, msg, memory, value_only=False, action_only=False):
        
        memoryA, memoryM = memory
        action = None
        message = None
        out_memory = [None, None]
        value = None
        if not value_only:
            # action encoding
            oA = self.obs_Mod_A(obs)
            mA = self.msg_Enc_A(msg)
            zA = self.rep_Mod_A(torch.cat([oA, mA], -1))
            if self.with_memory:
                hzA, czA = self.memory_A(zA, (memoryA[:, :self.memory_size], memoryA[:, self.memory_size:]))
            
                out_memory[0] = torch.cat([hzA, czA], dim=1)
            else:
                hzA = zA
            
            action = self.action_Mod(hzA)
            message = self.msg_Dec(hzA)
        if not action_only:
            # value encoding
            oM = self.obs_Mod_M(obs)
            mM = self.msg_Enc_M(msg)
            zM = self.rep_Mod_M(torch.cat([oM, mM], -1))
            if self.with_memory:
                hzM, czM = self.memory_M(zM, (memoryM[:, :self.memory_size], memoryM[:, self.memory_size:]))
                
                out_memory[1] = torch.cat([hzM, czM], dim=1)
            else:
                hzM = zM
            value = self.value_head(hzM)
        return action, message, out_memory, value

