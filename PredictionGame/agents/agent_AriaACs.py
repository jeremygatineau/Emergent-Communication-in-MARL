import torch
import torchvision
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import torch.nn.utils as utils
from torch.autograd import Variable
class AriaACs:
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
        
        self.modIActor = ariaActor(batch_size=self.batch_size, vocabulary_size=self.vocabulary_size, memory_size=self.memory_size ,with_memory=self.with_memory).float().eval()
        self.modTActor = ariaActor(batch_size=self.batch_size, vocabulary_size=self.vocabulary_size, memory_size=self.memory_size, with_memory=self.with_memory).float().train()
        self.modTCritic = Baseline()
        #ariaCritic(batch_size=self.batch_size, vocabulary_size=self.vocabulary_size, memory_size=self.memory_size, with_memory=self.with_memory).float().train()

        if self.with_memory:
            self.memoriesActor = [torch.zeros([1, 2*self.memory_size], dtype=torch.float32)]
            self.memoriesCritic = [torch.zeros([1, 2*self.memory_size], dtype=torch.float32)]

        else:
            self.memoriesActor = None
            self.memoriesCritic = None

        #self.optimizerActor = torch.optim.Adam(self.modTActor.parameters(), lr=opt_params["lr"])
        #self.optimizerCritic = torch.optim.Adam(self.modTCritic.parameters(), lr=opt_params["lr"])

        self.optimizerActor = torch.optim.RMSprop(self.modTActor.parameters(), lr=opt_params["lr"])
        self.optimizerCritic = torch.optim.RMSprop(self.modTCritic.parameters(), lr=opt_params["lr"])

        

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
            action, message, hid_state = self.modIActor.forward(obs_t.float(), msg_t.float(), self.memoriesActor[-1])
            #hid_state_critic, _ = self.modTCritic.forward(obs_t.float(), msg_t.float(), self.memoriesCritic[-1], action, message)
            if len(self.saved_obs) <self.replay_size:
                self.memoriesActor.append(hid_state.detach())
                #self.memoriesCritic.append(hid_state_critic.detach())
            else:
                self.memoriesActor[-1] = hid_state.detach()
                #self.memoriesCritic[-1] = hid_state_critic.detach()
        else:
            action, message, _, = self.modIActor.forward(obs_t.float(), msg_t.float(), None)

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
            action, message, hid_state = self.modTActor.forward(obs_t.float(), msg_t.float(), last_state)
        else:
            action, message, _ = self.modTActor.forward(obs_t.float(), msg_t.float(), None)
        
        a_distrib = Categorical(torch.cat([action, 1-action], -1))
        m_distrib = Categorical(message)
        a = a_distrib.sample()
        m = m_distrib.sample()
        a_entropy = a_distrib.entropy() 
        m_entropy = m_distrib.entropy()
        return a_distrib.log_prob(a), m_distrib.log_prob(m), hid_state, a_entropy + m_entropy, a_distrib.probs, m_distrib.probs


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
                self.optimizerActor.zero_grad()
                self.optimizerCritic.zero_grad()
                i0 = np.random.randint(self.batch_size, self.replay_size)
                returns = self.getReturns(self.saved_rewards[i0-self.batch_size:i0], normalize=False)
                returns = torch.tensor(returns, dtype=torch.float32)
                rewards = torch.tensor(self.saved_rewards[i0-self.batch_size:i0], dtype=torch.float32)
                #rewards -= rewards.mean()
                policy_loss = 0
                value_loss = torch.tensor(0, dtype=torch.float32)
                entropy_loss = 0
                last_state_actor = self.memoriesActor[i0-self.batch_size]
                last_state_critic = self.memoriesActor[i0-self.batch_size]
                saved_act_Logp = []
               
                for i in range(self.batch_size-1):
                    obs_t = torch.tensor([self.saved_obs[i0+i-self.batch_size]], dtype=torch.float32)
                    msg_t = torch.tensor([self.saved_downlink_msgs[i0+i-self.batch_size]], dtype=torch.float32)
                    obs_t_ = torch.tensor([self.saved_obs[i0+i+1-self.batch_size]], dtype=torch.float32)
                    msg_t_ = torch.tensor([self.saved_downlink_msgs[i0+i+1-self.batch_size]], dtype=torch.float32)
                    
                    if self.with_memory:
                        action, message, hid_state_actor = self.modTActor.forward(obs_t.float(), msg_t.float(), last_state_actor)
                        hid_state_critic, val = self.modTCritic.forward(obs_t.float(), msg_t.float(), last_state_critic, action, message)
                    else:
                        action, message, _ = self.modTActor.forward(obs_t.float(), msg_t.float(), None)
                    _, val_ = self.modTCritic.forward(obs_t_.float(), msg_t_.float(), hid_state_critic, action, message)

                    a_distrib = Categorical(torch.cat([action, 1-action], -1))
                    m_distrib = Categorical(message)
                    a = a_distrib.sample()
                    m = m_distrib.sample()
                    a_entropy = a_distrib.entropy() 
                    m_entropy = m_distrib.entropy()
                    last_state_actor = hid_state_actor
                    #last_state_critic = hid_state_critic
                    a_lp = a_distrib.log_prob(a)
                    m_lp = m_distrib.log_prob(m)
                    saved_act_Logp.append(a_lp)
                    Bi_a = self.saved_a_lp[i0+i-self.batch_size][0]
                    Fi_a = nn.functional.gumbel_softmax(torch.cat([action, 1-action], -1).log(), tau=1)[0]
                    
                    rho_a = torch.clamp(Fi_a[a]/Bi_a[a], min=0, max=5).detach()

                    Bi_m= self.saved_m_lp[i0+i-self.batch_size][0]
                    Fi_m = nn.functional.gumbel_softmax(message.log(), tau=1)[0]
                   
                    rho_m = torch.clamp(Fi_m[m]/Bi_m[m], min=0, max=5).detach()
                    bsl = self.modTCritic(1)
                    advantage = rewards[i]-bsl
                    #policy_loss += -(Fi_a[a].log()+Fi_m[m].log())*advantage.detach() # GBS 
                    #policy_loss += -( Fi_a[a].log()*rho_a.detach()+Fi_m[m].log()*rho_m.detach())*advantage.detach()  # GSB prioritized sampling   
                    policy_loss += -(a_lp+m_lp)*advantage.detach()  # straight AC
                    #policy_loss += -(a_lp+m_lp)*returns[i] # reinforce no baseline
                    bsl_loss = nn.functional.mse_loss(self.modTCritic(1), rewards[i].reshape(bsl.shape))
                    #value_loss += nn.functional.smooth_l1_loss(val, returns[i].reshape([1, 1])) # advantage.pow(2)
                    entropy_loss += self.epsilon*(a_entropy+m_entropy)
                policy_loss /=self.batch_size
                policy_loss.backward(retain_graph=True)
 
                bsl_loss.backward(retain_graph=True)
                
                self.grad_clamp(self.modTActor.parameters(), 0.1)
                self.grad_clamp(self.modTCritic.parameters(), 0.1)
                self.optimizerActor.step()
                self.optimizerCritic.step()
                mean_policy = torch.cat(saved_act_Logp, 0).exp().mean(dim=0)
                rewards = np.copy(self.saved_rewards)
            self.batch_counter+=1
            self.minibatch_counter = 0
            if self.batch_counter%1==0:
                self.modIActor.load_state_dict(self.modTActor.state_dict())
            return np.round([policy_loss.item()/self.batch_size, value_loss.item()/self.batch_size, entropy_loss.item()/self.batch_size], 4), rewards, mean_policy
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
    
    
    def popBuffer(self):
        self.saved_a_lp[:-1] = self.saved_a_lp[1:]
        self.saved_m_lp[:-1] = self.saved_m_lp[1:]
        self.saved_rewards[:-1] = self.saved_rewards[1:]
        self.saved_obs[:-1] = self.saved_obs[1:]
        self.saved_downlink_msgs[:-1] = self.saved_downlink_msgs[1:]
        self.memoriesActor[:-1] = self.memoriesActor[1:]
        self.memoriesCritic[:-1] = self.memoriesCritic[1:]
    
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
class ariaActor(nn.Module):
    def __init__(self, batch_size, vocabulary_size, hidden_dim=10, memory_size=8, with_memory=True):
        super(ariaActor, self).__init__()
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
        else : 
            self.memory = None
            self.action_Mod = lin_Mod([self.hiddenDim, 1], sftmx = True)
       
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
        self.act_enc = lin_Mod([1 + self.vocabulary_size, self.hiddenDim])
        self.obs_Mod = lin_Mod([2, self.hiddenDim])
        self.msg_Enc = lin_Mod([4, self.hiddenDim//2, self.hiddenDim], sftmx = False)
        self.rep_Mod = lin_Mod([self.hiddenDim*3, self.hiddenDim])
        if self.with_memory:
            self.memory = nn.LSTMCell(self.hiddenDim, self.memory_size)
            self.value_head = lin_Mod([self.memory_size, self.memory_size, 1])
        else : 
            self.memory = None
            self.value_head = lin_Mod([self.hiddenDim, 1])
       
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

