import torch
import torchvision
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import torch.nn.utils as utils
from torch.autograd import Variable
class AriaAC:
    def __init__(self,opt_params, with_memory=True, aidi=None):
        self.aidi = aidi
        self.batch_size = opt_params["batch_size"]
        self.gamma = opt_params["gamma"]
        self.vocabulary_size = opt_params["vocab_size"]
        self.epsilon = opt_params["eps"]
        self.replay_size = opt_params["replay_size"]
        self.training_loops = opt_params["training_loops"]
        self.memory_size = opt_params["memory_size"]
        self.eps = np.finfo(np.float32).eps.item()
        self.with_memory = with_memory
        
        self.modI = ariaModel(batch_size=self.batch_size, vocabulary_size=self.vocabulary_size, with_memory=self.with_memory).float().eval()
        self.modT = ariaModel(batch_size=self.batch_size, vocabulary_size=self.vocabulary_size, with_memory=self.with_memory).float().train()
        if self.with_memory:
            self.memories_ongoing = torch.zeros([1, 2*self.memory_size], dtype=torch.float32)
            self.memories_inital = torch.zeros([1, 2*self.memory_size], dtype=torch.float32)
        else:
            self.memories_ongoing = None
            self.memories_inital = None

        self.optimizer = torch.optim.Adam(self.modT.parameters(), lr=opt_params["lr"])

        self.saved_rewards = []
        self.saved_obs = []
        self.saved_downlink_msgs = []
        self.minibatch_counter = 0
        self.batch_counter = 0
        
    

    def select_action(self, obs, msg):

        obs_t = torch.tensor([obs], dtype=torch.float32)
        msg_t = torch.tensor([msg], dtype=torch.float32)
        
        if self.with_memory:
            action, message, hid_state, _ = self.modI.forward(obs_t.float(), msg_t.float(), self.memories_ongoing)
            self.memories_ongoing = hid_state.detach()
        else:
            action, message, _, _ = self.modI.forward(obs_t.float(), msg_t.float(), None)

        a_distrib = Categorical(torch.cat([action, 1-action], -1))
        m_distrib = Categorical(message)
        a = a_distrib.sample()
        m = m_distrib.sample()
        return a, m
    
    def select_actionTraing(self, obs, msg, last_state):
    
        obs_t = torch.tensor([obs], dtype=torch.float32)
        msg_t = torch.tensor([msg], dtype=torch.float32)
        
        if self.with_memory:
            action, message, hid_state, value = self.modT.forward(obs_t.float(), msg_t.float(), last_state)
        else:
            action, message, _, value = self.modT.forward(obs_t.float(), msg_t.float(), None)
        
        a_distrib = Categorical(torch.cat([action, 1-action], -1))
        m_distrib = Categorical(message)
        a = a_distrib.sample()
        m = m_distrib.sample()
        a_entropy = a_distrib.entropy() 
        m_entropy = m_distrib.entropy()
        return a_distrib.log_prob(a), m_distrib.log_prob(m), value, hid_state, a_entropy + m_entropy


    def train_on_batch(self, state, reward):
        #self.popBuffer()
        self.pushBufferEpisode(state[0], state[1], reward)
        self.minibatch_counter += 1
        if self.minibatch_counter >= self.batch_size:
            
            self.optimizer.zero_grad()
            returns = self.getReturns(normalize=True)
            returns = torch.tensor(returns)
            rewards = torch.tensor(self.saved_rewards)
            policy_loss = 0
            value_loss = 0
            entropy_loss = 0
            last_state = self.memories_inital.detach()
            saved_act_Logp = []
            for i in range(self.batch_size-1):
                a_lp, m_lp, val, hid_state, entropy = self.select_actionTraing(self.saved_obs[i], self.saved_downlink_msgs[i], last_state)
                last_state = hid_state
                saved_act_Logp.append(a_lp)
                advantage = returns[i]-val.item()
                policy_loss += -(a_lp + m_lp)*advantage.detach()        
                value_loss += advantage.pow(2)
                entropy_loss += self.epsilon*entropy
            self.memories_inital = last_state.detach()
            policy_loss /= self.batch_size
            value_loss /= self.batch_size
            loss = policy_loss+ value_loss #+ entropy_loss
            loss.backward(retain_graph=True)
            self.optimizer.step()
            mean_policy = torch.cat(saved_act_Logp, 0).exp().mean(dim=0)
            rewards = np.copy(self.saved_rewards)
            self.reset_Buffer()
            self.batch_counter+=1
            if self.batch_counter%30==0:
                self.modI.load_state_dict(self.modI.state_dict())
            return np.round([policy_loss.item(), value_loss.item(), entropy_loss.item()], 2), rewards, mean_policy
        return None, None, None

    def getReturns(self, normalize=False):
        _R = 0
        Gs = []
        for r in self.saved_rewards[:self.batch_size]:
            _R = r + self.gamma * _R
            Gs.insert(0, _R)
        if normalize==True:
            return (Gs-np.mean(Gs))/(np.std(Gs)+self.eps)
        return Gs
    
    
    def popBuffer(self):
        self.saved_rewards[:-1] = self.saved_rewards[1:]
        self.saved_obs[:-1] = self.saved_obs[1:]
        self.saved_downlink_msgs[:-1] = self.saved_downlink_msgs[1:]
    def sampleBatch(self):
        indices = np.random.randint(0, self.replay_size, self.batch_size)
        return indices
    def pushBufferEpisode(self, obs, msg, reward):
        self.saved_obs.append(obs)
        self.saved_downlink_msgs.append(msg)
        self.saved_rewards.append(reward)

    def reset_Buffer(self):
        self.saved_hid_states = []
        self.saved_rewards = []
        self.saved_obs = []
        self.saved_downlink_msgs = []
        self.minibatch_counter = 0
class lin_Mod(nn.Module):
    def __init__(self, sizes, sftmx = False):
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
class ariaModel(nn.Module):
    def __init__(self, batch_size, vocabulary_size, hidden_dim=8, memory_size=8, with_memory=True):
        super(ariaModel, self).__init__()
        self.hiddenDim = hidden_dim
        self.memory_size = memory_size
        self.with_memory = with_memory
        self.batch_size = batch_size
        self.vocabulary_size = vocabulary_size
        self.obs_Mod = lin_Mod([2, self.hiddenDim//2])
        self.msg_Enc = lin_Mod([4, self.hiddenDim//2, self.hiddenDim//2], sftmx = False)
        self.rep_Mod = lin_Mod([self.hiddenDim, self.hiddenDim])
        if self.with_memory:
            self.memory = nn.LSTMCell(self.hiddenDim, self.memory_size)
            self.action_Mod = nn.Sequential(lin_Mod([self.memory_size, 1], sftmx = False), nn.Sigmoid())
            self.msg_Dec = lin_Mod([self.memory_size, self.memory_size, self.vocabulary_size], sftmx=True)
            self.value_head = lin_Mod([self.memory_size, self.memory_size, 1])
        else : 
            self.memory = None
            self.action_Mod = lin_Mod([self.hiddenDim, 1], sftmx = True)
            self.value_head = lin_Mod([self.hiddenDim, 1])
       
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
