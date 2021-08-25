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
        
        self.modI = ariaModel(batch_size=self.batch_size, vocabulary_size=self.vocabulary_size, memory_size=self.memory_size ,with_memory=self.with_memory).float().eval()
        self.modT = ariaModel(batch_size=self.batch_size, vocabulary_size=self.vocabulary_size, memory_size=self.memory_size, with_memory=self.with_memory).float().train()
        if self.with_memory:
            self.memories = [torch.zeros([1, 2*self.memory_size], dtype=torch.float32)]
        else:
            self.memories = None

        self.optimizer = torch.optim.Adam(self.modT.parameters(), lr=opt_params["lr"])
        self.saved_a_lp = []
        self.saved_m_lp = []
        self.saved_rewards = []
        self.saved_obs = []
        self.saved_downlink_msgs = []
        self.minibatch_counter = 0
        self.batch_counter = 0
        
    

    def select_action(self, obs, msg):

        obs_t = torch.tensor([obs], dtype=torch.float32)
        msg_t = torch.tensor([msg], dtype=torch.float32)
        
        if self.with_memory:
            action, message, hid_state, _ = self.modI.forward(obs_t.float(), msg_t.float(), self.memories[-1])
            
            if len(self.saved_obs) <self.replay_size:
                self.memories.append(hid_state.detach())
            else:
                self.memories[-1] = hid_state.detach()
        else:
            action, message, _, _ = self.modI.forward(obs_t.float(), msg_t.float(), None)

        a_distrib = Categorical(torch.cat([action, 1-action], -1))
        m_distrib = Categorical(message)
        #a = torch.argmax(a_distrib.probs, axis=1)
        #m = torch.argmax(m_distrib.probs, axis=1)
        a = a_distrib.sample()
        m = m_distrib.sample()
        self.saved_a_lp.append(torch.cat([action, 1-action], -1))
        self.saved_m_lp.append(message)
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
        return a, m, a_distrib.log_prob(a), m_distrib.log_prob(m), value, hid_state, a_entropy + m_entropy, torch.cat([action, 1-action], -1), message
    
    def train_online(self, reward, a_lp, m_lp, val):
        
        adv = reward[:-1]-val[:-1]+self.gamma*val[1:]
        policy_loss = -(a_lp[:-1] + m_lp[:-1])*adv.detach()
        value_loss = adv.pow(2)
        loss = policy_loss.mean()+ value_loss.mean()
        loss.backward()
        self.optimizer.step()
        return policy_loss, value_loss, 0 

    def train_on_batch(self, state, reward):    
        if len(self.saved_obs)<self.replay_size:
            self.pushBufferEpisode(state[0], state[1], reward)
            self.minibatch_counter += 1
            return None, None, None
        self.popBuffer()
        self.pushBufferEpisode(state[0], state[1], reward)
        self.minibatch_counter += 1
        if self.minibatch_counter >= self.batch_size:
            
            for _ in range(self.training_loops):
                self.optimizer.zero_grad()
                i0 = np.random.randint(self.batch_size, self.replay_size)
                returns = self.getReturns(self.saved_rewards[i0-self.batch_size:i0], normalize=False)
                returns = torch.tensor(returns, dtype=torch.float32)
                rewards = torch.tensor(self.saved_rewards[i0-self.batch_size:i0], dtype=torch.float32)
                rewards -= rewards.mean()
                policy_loss = 0
                value_loss = torch.tensor(0, dtype=torch.float32)
                entropy_loss = 0
                last_state = self.memories[i0-self.batch_size]
                saved_act_Logp = []
               
                for i in range(self.batch_size-1):
                    obs_t = torch.tensor([self.saved_obs[i0+i-self.batch_size]], dtype=torch.float32)
                    msg_t = torch.tensor([self.saved_downlink_msgs[i0+i-self.batch_size]], dtype=torch.float32)
                    obs_t_ = torch.tensor([self.saved_obs[i0+i+1-self.batch_size]], dtype=torch.float32)
                    msg_t_ = torch.tensor([self.saved_downlink_msgs[i0+i+1-self.batch_size]], dtype=torch.float32)
                    
                    if self.with_memory:
                        action, message, hid_state, val = self.modT.forward(obs_t.float(), msg_t.float(), last_state)
                    else:
                        action, message, _, val = self.modT.forward(obs_t.float(), msg_t.float(), None)
                    _, _, _, val_ = self.modT.forward(obs_t_.float(), msg_t_.float(), hid_state)

                    a_distrib = Categorical(torch.cat([action, 1-action], -1))
                    m_distrib = Categorical(message)
                    a = a_distrib.sample()
                    m = m_distrib.sample()
                    a_entropy = a_distrib.entropy() 
                    m_entropy = m_distrib.entropy()
                    last_state = hid_state
                    a_lp = a_distrib.log_prob(a)
                    m_lp = m_distrib.log_prob(m)
                    saved_act_Logp.append(torch.log(action))
                    Bi_a = self.saved_a_lp[i0+i-self.batch_size][0]
                    Fi_a = nn.functional.gumbel_softmax(torch.cat([action, 1-action], -1).log(), tau=1)[0]
                    
                    rho_a = torch.clamp(Fi_a[a]/Bi_a[a], min=0, max=5).detach()

                    Bi_m= self.saved_m_lp[i0+i-self.batch_size][0]
                    Fi_m = nn.functional.gumbel_softmax(message.log(), tau=1)[0]
                   
                    rho_m = torch.clamp(Fi_m[m]/Bi_m[m], min=0, max=5).detach()
                    advantage = rewards[i]-val
                    #policy_loss += -(Fi_a[a].log()+Fi_m[m].log())*advantage.detach() # GBS 
                    #policy_loss += -(Fi_a[a].log()*rho_a.detach()+Fi_m[m].log()*rho_m.detach())*advantage.detach()  # GSB prioritized sampling   
                    policy_loss += -(a_lp)*advantage.detach()  # straight AC
                    #policy_loss += -(a_lp+m_lp)*returns[i] # reinforce no baseline
                    
                    #value_loss += nn.functional.smooth_l1_loss(val, returns[i].reshape([1, 1])) # advantage.pow(2)
                    value_loss += advantage.pow(2).mean()
                    entropy_loss += self.epsilon*(a_entropy+m_entropy)
                
                loss = (policy_loss + value_loss)/self.batch_size
                loss.backward(retain_graph=True)
                utils.clip_grad_norm_(self.modT.parameters(), 0.1)
                self.optimizer.step()
                mean_policy = torch.cat(saved_act_Logp, 0).exp().mean(dim=0)
                rewards = np.copy(self.saved_rewards)
            self.batch_counter+=1
            self.minibatch_counter = 0
            if self.batch_counter%1==0:
                self.modI.load_state_dict(self.modT.state_dict())
            return np.round([policy_loss.item()/self.batch_size, value_loss.item()/self.batch_size, entropy_loss.item()/self.batch_size], 4), rewards, mean_policy
        return None, None, None

    def getReturns(self, rewards, normalize=False):
        _R = 0
        Gs = []
        for r in rewards:
            _R = r + self.gamma * _R
            Gs.insert(0, _R)
        if normalize==True:
            return (Gs-np.mean(Gs))/(np.std(Gs)+self.eps)
        return Gs
    
    
    def popBuffer(self):
        self.saved_a_lp[:-1] = self.saved_a_lp[1:]
        self.saved_m_lp[:-1] = self.saved_m_lp[1:]
        self.saved_rewards[:-1] = self.saved_rewards[1:]
        self.saved_obs[:-1] = self.saved_obs[1:]
        self.saved_downlink_msgs[:-1] = self.saved_downlink_msgs[1:]
        self.memories[:-1] = self.memories[1:]
    
    def pushBufferEpisode(self, obs, msg, reward):
        if len(self.saved_obs) <self.replay_size:
            self.saved_obs.append(obs)
            self.saved_downlink_msgs.append(msg)
            self.saved_rewards.append(reward)
        else:
            self.saved_obs[-1] = obs
            self.saved_downlink_msgs[-1] = msg
            self.saved_rewards[-1] = reward

    def reset_Buffer(self):
        self.saved_a_lp = []
        self.saved_m_lp = []
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
    def __init__(self, batch_size, vocabulary_size, hidden_dim=10, memory_size=8, with_memory=True):
        super(ariaModel, self).__init__()
        self.hiddenDim = hidden_dim
        self.memory_size = memory_size
        self.with_memory = with_memory
        self.batch_size = batch_size
        self.vocabulary_size = vocabulary_size
        self.obs_Mod = nn.Sequential(nn.Linear(2, self.hiddenDim//2), nn.ReLU())
        self.msg_Enc = nn.Sequential(nn.Linear(4, self.hiddenDim//2), nn.ReLU(), nn.Linear(self.hiddenDim//2, self.hiddenDim//2), nn.ReLU())
        self.rep_Mod = nn.Sequential(nn.Linear(self.hiddenDim, self.hiddenDim), nn.ReLU())
        if self.with_memory:
            self.memory = nn.LSTMCell(self.hiddenDim, self.memory_size)
            self.action_Mod = nn.Sequential(nn.Linear(self.memory_size, 1), nn.Sigmoid())
            self.msg_Dec = nn.Sequential(nn.Linear(self.memory_size, self.memory_size), nn.Linear(self.memory_size, self.vocabulary_size), nn.Softmax(dim=-1))
            self.value_head = nn.Sequential(nn.Linear(self.memory_size, self.memory_size), nn.Linear(self.memory_size, 1))
        else : 
            self.memory = None
            self.action_Mod = nn.Sequential(nn.Linear(self.hiddenDim, 1), nn.Sigmoid())
            self.value_head = nn.Sequential(nn.Linear(self.hiddenDim, self.hiddenDim), nn.Linear(self.memory_size, 1))
       
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

