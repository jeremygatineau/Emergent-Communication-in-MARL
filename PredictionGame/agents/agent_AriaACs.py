import torch
import torchvision
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import torch.nn.utils as utils
from torch.autograd import Variable
class AriaACs:
    def __init__(self,opt_params, with_memory=True, split=False, aidi=None):
        self.aidi = aidi
        self.batch_size = opt_params["batch_size"]
        self.gamma = opt_params["gamma"]
        self.vocabulary_size = opt_params["vocab_size"]
        self.memory_size = opt_params["memory_size"]
        self.hidden_dim = opt_params["hidden_size"]
        self.gc = opt_params["grad_clamp"]
        self.replay_size = opt_params["replay_size"]
        self.training_loops = opt_params["training_loops"]
        self.split = split
        self.eps = np.finfo(np.float32).eps.item()
        self.with_memory = with_memory
        
        if split: 
            self.modT = ariaModelSplit(batch_size=self.batch_size, vocabulary_size=self.vocabulary_size, memory_size=self.memory_size, hidden_dim=self.hidden_dim, with_memory=self.with_memory).float().train()
            self.modI = ariaModelSplit(batch_size=self.batch_size, vocabulary_size=self.vocabulary_size, memory_size=self.memory_size, hidden_dim=self.hidden_dim, with_memory=self.with_memory).float().eval()
        else:
            self.modT = ariaModel(batch_size=self.batch_size, vocabulary_size=self.vocabulary_size, memory_size=self.memory_size, hidden_dim=self.hidden_dim, with_memory=self.with_memory).float().train()
            self.modI = ariaModel(batch_size=self.batch_size, vocabulary_size=self.vocabulary_size, memory_size=self.memory_size, hidden_dim=self.hidden_dim, with_memory=self.with_memory).float().eval()
        
        if self.with_memory:
            if split:
                self.hid_states = [[torch.zeros(1, 2*self.memory_size).detach(), torch.zeros(1, 2*self.memory_size).detach()]]
            else:
                self.hid_states = [torch.zeros(1, 2*self.memory_size).detach()]

        else:
            self.memoriesActor = None

        self.optimizer = torch.optim.RMSprop(self.modT.parameters(), lr=opt_params["lr"])

        self.saved_a = []
        self.saved_m = []
        self.saved_rewards = []
        self.saved_obs = []
        self.saved_downlink_msgs = []
        self.minibatch_counter = 0
        self.batch_counter = 0
        
    

    def select_action(self, obs, msg):
        obs_t = torch.tensor([obs], dtype=torch.float32)
        msg_t = torch.tensor([msg], dtype=torch.float32)
        
        if self.with_memory:
            action, message, hid_state, _ = self.modI.forward(obs_t.float(), msg_t.float(), self.hid_states[-1])
            if len(self.saved_obs) <self.replay_size:
                if self.split:
                    self.hid_states.append([hid_state[0].detach(), hid_state[1].detach()])
                else:
                    self.hid_states.append(hid_state.detach())
            else:
                if self.split:
                    self.hid_states[-1] = [hid_state[0].detach(), hid_state[1].detach()]
                else:
                    self.hid_states[-1] = hid_state.detach()
                
        else:
            action, message, _, _ = self.modI.forward(obs_t.float(), msg_t.float(), None)

        a_distrib = Categorical(torch.cat([action, 1-action], -1))
        m_distrib = Categorical(message)
        a = torch.argmax(a_distrib.probs, axis=1)
        m = torch.argmax(m_distrib.probs, axis=1)
        return a, m
    
    def select_actionTraining(self, obs, msg, last_state, a_i, m_i):
        obs_t = torch.tensor([obs], dtype=torch.float32)
        msg_t = torch.tensor([msg], dtype=torch.float32)
        if self.with_memory:
            action, message, hid_state_actor, val = self.modT.forward(obs_t.float(), msg_t.float(), last_state)
        else:
            action, message, _, val = self.modT.forward(obs_t.float(), msg_t.float(), None)

        a_distrib = Categorical(torch.cat([action, 1-action], -1))
        m_distrib = Categorical(message)
        a_entropy = a_distrib.entropy() 
        m_entropy = m_distrib.entropy()
        last_state = hid_state_actor
        a_lp = a_distrib.log_prob(a_i)
        m_lp = m_distrib.log_prob(m_i)
        return a_lp, m_lp, val, a_entropy+m_entropy, last_state, torch.cat([action, 1-action], -1), message

    def train_on_batch(self, state, reward, actions):    
        if len(self.saved_obs)<self.replay_size:
            self.poppushBufferEpisode(state[0], state[1], reward, actions, pop=False)
            self.minibatch_counter += 1
            return None, None, None
        self.poppushBufferEpisode(state[0], state[1], reward, actions)
        self.minibatch_counter += 1
        if self.minibatch_counter >= self.batch_size:
            
            for _ in range(self.training_loops):
                self.optimizer.zero_grad()
                i0 = np.random.randint(self.batch_size, self.replay_size)
                #returns = torch.tensor(self.getReturns(self.saved_rewards[i0-self.batch_size:i0], normalize=False), dtype=torch.float32)
                rewards = torch.tensor(self.saved_rewards[i0-self.batch_size:i0], dtype=torch.float32)
                #rewards -= rewards.mean()
                last_state = self.hid_states[i0-self.batch_size]
                
                val_t = torch.zeros(self.batch_size, dtype=torch.float32)
                a_lp_t = torch.zeros(self.batch_size, dtype=torch.float32)
                m_lp_t = torch.zeros(self.batch_size, dtype=torch.float32)
                entropy_t = torch.zeros(self.batch_size, dtype=torch.float32)
                mean_a_pol = []
                for i in range(self.batch_size):
                    obs = self.saved_obs[i0+i-self.batch_size]
                    msg = self.saved_downlink_msgs[i0+i-self.batch_size]
                    a_i = self.saved_a[i0+i-self.batch_size]
                    m_i = self.saved_a[i0+i-self.batch_size]
                    a_lp, m_lp, val, entropy, last_state, a_dist, msg_dist = self.select_actionTraining(obs, msg, last_state, a_i, m_i)
                    mean_a_pol.append(a_dist[0, 0].item())
                    val_t[i] = val
                    a_lp_t[i] = a_lp
                    m_lp_t[i] = m_lp
                    entropy_t[i] = -entropy
                adv = rewards[:-1]-val_t[:-1]+self.gamma*val_t[1:]
                policy_loss = -(a_lp_t[:-1] + m_lp_t[:-1])*adv.detach()
                value_loss =  nn.functional.smooth_l1_loss(rewards[:-1]+self.gamma*val_t[1:], val_t[:-1], reduction='none')# adv.pow(2)
                entropy_loss = entropy_t.mean()
                loss = policy_loss.mean() + value_loss.mean() + self.eps*entropy_loss
                loss.backward(retain_graph=True)
                
                if self.gc is not None:self.grad_clamp(self.modT.parameters(), self.gc)
                self.optimizer.step()
            self.batch_counter+=1
            self.minibatch_counter = 0
            if self.batch_counter%100==0:
                self.modI.load_state_dict(self.modT.state_dict())
            return (policy_loss, value_loss, entropy_loss), mean_a_pol, rewards
        return None, None, None

    def train_on_batch_old(self, state, reward, actions):    
        if len(self.saved_obs)<self.replay_size:
            self.pushBufferEpisode(state[0], state[1], reward, actions)
            self.minibatch_counter += 1
            return None, None, None
        self.popBuffer()
        self.pushBufferEpisode(state[0], state[1], reward, actions)
        self.minibatch_counter += 1
        if self.minibatch_counter >= self.batch_size:
            
            for _ in range(self.training_loops):
                self.optimizer.zero_grad()
                i0 = np.random.randint(self.batch_size, self.replay_size)
                returns = self.getReturns(self.saved_rewards[i0-self.batch_size:i0], normalize=False)
                returns = torch.tensor(returns, dtype=torch.float32)
                rewards = torch.tensor(self.saved_rewards[i0-self.batch_size:i0], dtype=torch.float32)
                #rewards -= rewards.mean()
                policy_loss = torch.zeros(self.batch_size-1, dtype=torch.float32)
                value_loss = torch.zeros(self.batch_size-1, dtype=torch.float32)
                entropy_loss = 0
                last_state_actor = self.hid_states[i0-self.batch_size]
                
                val_t = torch.zeros(self.batch_size-1, dtype=torch.float32)
                a_lp_t = torch.zeros(self.batch_size-1, dtype=torch.float32)
                m_lp_t = torch.zeros(self.batch_size-1, dtype=torch.float32)
                entropy_t = torch.zeros(self.batch_size-1, dtype=torch.float32)

                for i in range(self.batch_size-1):
                    obs_t = torch.tensor([self.saved_obs[i0+i-self.batch_size]], dtype=torch.float32)
                    msg_t = torch.tensor([self.saved_downlink_msgs[i0+i-self.batch_size]], dtype=torch.float32)
                    a_i = self.saved_a[i0+i-self.batch_size]
                    m_i = self.saved_m[i0+i-self.batch_size]
                    if self.with_memory:
                        action, message, hid_state_actor, val = self.modT.forward(obs_t.float(), msg_t.float(), last_state_actor)
                    else:
                        action, message, _, val = self.modT.forward(obs_t.float(), msg_t.float(), None)

                    a_distrib = Categorical(torch.cat([action, 1-action], -1))
                    m_distrib = Categorical(message)
                    a_entropy = a_distrib.entropy() 
                    m_entropy = m_distrib.entropy()
                    last_state_actor = hid_state_actor
                    a_lp = a_distrib.log_prob(a_i)
                    m_lp = m_distrib.log_prob(m_i)
                    val_t[i] = val
                    a_lp_t[i] = a_lp
                    m_lp_t[i] = m_lp
                    entropy_t[i] = -(a_entropy+m_entropy)
                    #Bi_a = self.saved_a_lp[i0+i-self.batch_size][0]
                    #Fi_a = nn.functional.gumbel_softmax(torch.cat([action, 1-action], -1).log(), tau=1)[0]
                    
                    #rho_a = torch.clamp(Fi_a[a_i]/Bi_a[a_i], min=0, max=5).detach()

                    #Bi_m= self.saved_m_lp[i0+i-self.batch_size][0]
                    #Fi_m = nn.functional.gumbel_softmax(message.log(), tau=1)[0]
                   
                    #rho_m = torch.clamp(Fi_m[a_i]/Bi_m[m_i], min=0, max=5).detach()
                    #bsl = self.modTCritic(1)
                    #advantage = rewards[i]-bsl
                    #advantage = rewards[i]-val + self.gamma*val_
                    #policy_loss += -(Fi_a[a].log()+Fi_m[m].log())*advantage.detach() # GBS 
                    #policy_loss += -( Fi_a[a].log()*rho_a.detach()+Fi_m[m].log()*rho_m.detach())*advantage.detach()  # GSB prioritized sampling   
                    #policy_loss[i] = -(a_lp+m_lp)*advantage.detach()  # straight AC
                    #policy_loss += -(a_lp+m_lp)*returns[i] # reinforce no baseline
                    #bsl_loss = nn.functional.mse_loss(self.modTCritic(1), rewards[i].reshape(bsl.shape))
                    #value_loss[i] = advantage.pow(2).mean()
                    #value_loss += nn.functional.smooth_l1_loss(val, returns[i].reshape([1, 1])) # advantage.pow(2)
                    #entropy_loss += -(a_entropy+m_entropy)
                adv = rewards[:-1]-val_t[:-1]+self.gamma*val_t[1:]
                policy_loss = -(a_lp_t[:-1] + m_lp_t[:-1])*adv.detach()
                value_loss =  nn.functional.smooth_l1_loss(rewards[:-1]+self.gamma*val_t[1:], val_t[:-1], reduction='none')# adv.pow(2)
                entropy_loss = entropy_t.mean()
                loss = policy_loss.mean() + value_loss.mean() + self.eps*entropy_loss
                loss.backward(retain_graph=True)
                
                self.grad_clamp(self.modT.parameters(), self.gc)
                self.grad_clamp(self.modT.parameters(), self.gc)
                self.optimizer.step()
                mean_policy = torch.cat(a_lp_t, 0).exp().mean(dim=0)
                rewards = np.copy(self.saved_rewards)
            self.batch_counter+=1
            self.minibatch_counter = 0
            if self.batch_counter%100==0:
                self.modI.load_state_dict(self.modT.state_dict())
            return policy_loss, value_loss, entropy_loss
        return None, None, None
    def grad_clamp(self, parameters, clip=0.1):
        for p in parameters:
            if p.grad is not None:
                p.grad.clamp_(min=-clip)
                p.grad.clamp_(max=clip)

    def getReturns(self, rewards, normalize=False):
        _R = 0
        Gs = []
        for r in rewards:
            _R = r + self.gamma * _R
            Gs.insert(0, _R)
        if normalize==True:
            return (Gs-np.mean(Gs))/(np.std(Gs)+self.eps)
        return Gs
    
    def poppushBufferEpisode(self, obs, msg, reward, actions, pop=True):
        if pop:
            self.saved_a[:-1] = self.saved_a[1:]
            self.saved_m[:-1] = self.saved_m[1:]
            self.saved_rewards[:-1] = self.saved_rewards[1:]
            self.saved_obs[:-1] = self.saved_obs[1:]
            self.saved_downlink_msgs[:-1] = self.saved_downlink_msgs[1:]
            self.hid_states[:-1] = self.hid_states[1:]
        if len(self.saved_obs) <self.replay_size:
            self.saved_a.append(actions[0])
            self.saved_m.append(actions[1])
            self.saved_obs.append(obs)
            self.saved_downlink_msgs.append(msg)
            self.saved_rewards.append(reward)
        else:
            self.saved_a[-1] = actions[0]
            self.saved_m[-1] = actions[1]
            self.saved_obs[-1] = obs
            self.saved_downlink_msgs[-1] = msg
            self.saved_rewards[-1] = reward

    def reset_Buffer(self):
        self.saved_a = []
        self.saved_m = []
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
class ariaActor(nn.Module):
    def __init__(self, batch_size, vocabulary_size, hidden_dim=10, memory_size=8, with_memory=True):
        super(ariaActor, self).__init__()
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
        else : 
            self.memory = None
            self.action_Mod = nn.Sequential(nn.Linear(self.hiddenDim, 1), nn.Sigmoid())
       
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
        return action, message, out_memory

class Baseline(nn.Module):
    
    def __init__(self):
        super(Baseline, self).__init__()
        self.bias = nn.Parameter(torch.ones(1))
    def forward(self, bs):
        batch_bias = (self.bias + 1.).expand(bs,1)
        return batch_bias
class ariaCritic(nn.Module):
    def __init__(self, batch_size, vocabulary_size, hidden_dim=10, memory_size=8, with_memory=True):
        super(ariaCritic, self).__init__()
        self.hiddenDim = hidden_dim
        self.memory_size = memory_size
        self.with_memory = with_memory
        self.batch_size = batch_size
        self.vocabulary_size = vocabulary_size
        self.obs_Mod = nn.Sequential(nn.Linear(2, self.hiddenDim), nn.ReLU())
        self.msg_Enc = nn.Sequential(nn.Linear(4, self.hiddenDim), nn.ReLU(), nn.Linear(self.hiddenDim, self.hiddenDim), nn.ReLU())
        self.rep_Mod = nn.Sequential(nn.Linear(self.hiddenDim*3, self.hiddenDim), nn.ReLU())
        self.act_enc = nn.Sequential(nn.Linear(1 + self.vocabulary_size, self.hiddenDim), nn.ReLU())
        if self.with_memory:
            self.memory = nn.LSTMCell(self.hiddenDim, self.memory_size)
            self.value_head =  nn.Sequential(nn.Linear(self.memory_size, self.memory_size), nn.ReLU(), nn.Linear(self.memory_size, 1), nn.Sigmoid())
            
        else : 
            self.memory = None
            self.value_head = nn.Sequential(nn.Linear(self.hiddenDim, 1))
       
    def forward(self, obs, dl_msg, memory, action, ul_msg):
        o = self.obs_Mod(obs)
        m = self.msg_Enc(dl_msg)
        act = self.act_enc(torch.cat([action, ul_msg], -1))
        z = self.rep_Mod(torch.cat([o, m, act], -1))
        if self.with_memory:
            hz, cz = self.memory(z, (memory[:, :self.memory_size], memory[:, self.memory_size:]))
            
            out_memory = torch.cat([hz, cz], dim=1)
        else:
            hz = z
            out_memory = None
        value = self.value_head(hz)
        return out_memory, value

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

