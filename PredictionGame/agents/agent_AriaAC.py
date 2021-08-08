import torch
import torchvision
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import torch.nn.utils as utils
from torch.autograd import Variable
class AriaAC(nn.Module):
    def __init__(self,opt_params, with_memory=False):
        super(AriaAC, self).__init__()
        self.batch_size = opt_params["batch_size"]
        self.gamma = opt_params["gamma"]
        self.eps = np.finfo(np.float32).eps.item()
        self.hiddenDim = 2
        self.memory_size = 2
        self.with_memory = with_memory
        self.obs_Mod = lin_Mod([2, self.hiddenDim//2])
        self.action_Mod = lin_Mod([self.hiddenDim, 2], sftmx = True)
        self.msg_Enc = lin_Mod([4, self.hiddenDim//2], sftmx = False)
        self.msg_Dec = lin_Mod([self.hiddenDim, 4], sftmx=True)
        self.rep_Mod = lin_Mod([self.hiddenDim, self.hiddenDim])
        if self.with_memory:
            self.memory = nn.LSTMCell(self.hiddenDim, self.memory_size)
            self.memories = [torch.zeros([1, 2*self.memory_size], dtype=torch.float32) for _ in range(self.batch_size+1)]
        else : 
            self.memory = None
            self.memories = None
        self.value_head = lin_Mod([self.hiddenDim, 1])
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt_params["lr"])

        self.saved_act_Logp = []
        self.saved_values = []
        self.saved_entropies = []
        self.saved_msg_Logp = []
        self.saved_hid_states = []
        self.saved_rewards = []
        self.saved_obs = []
        self.saved_downlink_msgs = []
        self.batch_counter = 0
    """def embed_Obs(self, obs):
        self.zObs = self.obs_Mod(obs) """
        
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

    def select_action(self, obs, msg):
        obs_t = torch.tensor([obs], dtype=torch.float32)
        msg_t = torch.tensor([msg], dtype=torch.float32)
        
        if self.with_memory:
            action, message, hid_state, value = self.forward(obs_t.float(), msg_t.float(), self.memories[self.batch_counter])
            self.memories[self.batch_counter+1] = hid_state
        else:
            action, message, _, value = self.forward(obs_t.float(), msg_t.float(), None)
        
        
        a_distrib = Categorical(action)
        m_distrib = Categorical(message)
        a = a_distrib.sample()
        m = m_distrib.sample()
        a_entropy = a_distrib.entropy().sum() 
        m_entropy = m_distrib.entropy().sum() 
        self.saved_act_Logp.append(a_distrib.log_prob(a))
        self.saved_msg_Logp.append(m_distrib.log_prob(m))
        self.saved_values.append(value)
        return a, m, a_entropy + m_entropy

    def train_on_batch(self, state, reward, entropy):
        self.saved_obs.append(state[0])
        self.saved_downlink_msgs.append(state[1])
        self.saved_rewards.append(reward)
        self.saved_entropies.append(entropy)
        self.batch_counter += 1
        if self.batch_counter >= self.batch_size:
            
            self.optimizer.zero_grad()
            returns = self.getReturns(normalize=False)
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + self.eps)

            loss = 0
            for a_logp, m_logp, ret, val, entro in zip(self.saved_act_Logp, self.saved_msg_Logp, returns, self.saved_values, self.saved_entropies):
                advantage = ret - val.item()
                policy_loss = -(a_logp + m_logp)*advantage.detach()        
                value_loss = advantage.pow(2)
                loss += policy_loss+value_loss#+0.001*entro
            
            loss.backward(retain_graph=True)
            self.optimizer.step()
            rewards = np.copy(self.saved_rewards)
            self.reset_batch()
            return loss.item(), rewards
        return None, None

    def getReturns(self, normalize=False):
        _R = 0
        Gs = []
        for r in self.saved_rewards[:self.batch_size]:
            _R = r + self.gamma * _R
            Gs.insert(0, _R)
        if normalize==True:
            return (Gs-np.mean(Gs))/np.std(Gs)
        return Gs
    def reset_batch(self):
        self.memories = [torch.zeros([1, 2*self.memory_size], dtype=torch.float32) for _ in range(self.batch_size+1)]
        self.saved_act_Logp = []
        self.saved_values = []
        self.saved_entropies = []
        self.saved_msg_Logp = []
        self.saved_hid_states = []
        self.saved_rewards = []
        self.saved_obs = []
        self.saved_downlink_msgs = []
        self.batch_counter = 0
class lin_Mod(nn.Module):
    def __init__(self, sizes = [2, 5, 6, 10, 10], sftmx = False):
        super(lin_Mod, self).__init__()
        self.sizes = sizes
        L = []
        
        for i, s in enumerate(sizes[:-1]):
            L.append(nn.Linear(sizes[i], sizes[i+1]))
            if i==len(sizes)-2 and sftmx==True:
                L.append(nn.Softmax(dim=-1))
            else :
                L.append(nn.ReLU())
        self.mod = nn.ModuleList(L)
    
    def forward(self, x):
        x_ = x
        for m in self.mod:
            x_ = m(x_)
        return x_
class ariaActor(nn.Module):
    def __init__(self, hidden_dim=10):
        super(ariaActor, self).__init__()
        self.hiddenDim = hidden_dim
        self.obs_Mod = lin_Mod([2+self.hiddenDim, 5])
        self.action_Mod = lin_Mod([self.hiddenDim, 2], sftmx = True)
        self.msg_Enc = lin_Mod([4, 5], sftmx = False)
        self.msg_Dec = lin_Mod([self.hiddenDim, 4], sftmx=True)
        self.rep_Mod = lin_Mod([self.hiddenDim, self.hiddenDim])
       
    def forward(self, obs, msg, last_state):
        inO = torch.cat([obs, last_state[self.batch_counter]], -1)
        o = self.obs_Mod(inO)
        m = self.msg_Enc(msg)
        new_state = self.rep_Mod(torch.cat([o, m], -1))
        action = self.action_Mod(new_state)
        message = self.msg_Dec(new_state)

        return action, message, new_state

class ariaCritic(nn.Module):
    def __init__(self, hidden_dim=10):
        super(ariaCritic, self).__init__()
        self.hiddenDim = hidden_dim
        self.obs_Mod = lin_Mod([2+self.hiddenDim, 5])
        self.action_Mod = lin_Mod([self.hiddenDim, 2], sftmx = True)
        self.msg_Enc = lin_Mod([4, 5], sftmx = False)
        self.msg_Dec = lin_Mod([self.hiddenDim, 4], sftmx=True)
        self.rep_Mod = lin_Mod([self.hiddenDim, self.hiddenDim])
       
    def forward(self, obs, msg, last_state):
        inO = torch.cat([obs, last_state[self.batch_counter]], -1)
        o = self.obs_Mod(inO)
        m = self.msg_Enc(msg)
        new_state = self.rep_Mod(torch.cat([o, m], -1))
        action = self.action_Mod(new_state)
        message = self.msg_Dec(new_state)

        return action, message, new_state