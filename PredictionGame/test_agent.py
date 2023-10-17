# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from oneDtoyTask.oneDtoyTask import ToyTask, OneDfield, TwoWayComChannel
from agents.agent_AriaRE import AriaRE
from agents.agent_AriaAC import AriaAC
from IPython import display
import wandb
import PIL
import matplotlib
matplotlib.use('Agg')


epochs = int(1e3)
opt_params = {"lr":3e-3, "batch_size":128, \
              "gamma":0.99, "vocab_size":2, "training_loops":1, \
              "memory_size":20, "hidden_size": 30, "replay_size":120, \
              "eps":0.01, "cross_reward_coef":0.3, "grad_clamp":None}
run = wandb.init(config=opt_params, project='EC-MARL TOY PB', entity='jjer125')

agent0 = AriaAC(opt_params=opt_params, split=True, with_memory=True, aidi=0)
agent1 = AriaAC(opt_params=opt_params, split=True, with_memory=True, aidi=1)    
np.random.seed(1)
field = OneDfield(speed=1)
Task = ToyTask(field=field,\
               observationMappingFct=lambda x: (x>0.5).astype(int), \
               comChannel=TwoWayComChannel(), \
               vocabulary_size=opt_params["vocab_size"], \
               cross_r_coef=opt_params["cross_reward_coef"])

wandb.watch((agent0.modT, agent1.modT), log="all", log_freq=5)
table = wandb.Table(columns=["Epoch#", "batch_pred A0", "batch_pred A1"])
run.log({"Batch Predictions": table})
# %%
def plot_preds(obsers, preds):
    display.clear_output(wait=True)
    plt.clf()
    obsers = np.array(obsers)
    preds = np.array(preds)
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    axs[0].plot(range(obsers.shape[0]), obsers[:, 0])
    axs[0].plot(range(obsers.shape[0]), preds[:, 0])
    axs[0].legend(['observations', 'predictions'])
    axs[0].set_title('Agent 0')
    axs[0].set_xlabel('Epoch')

    axs[1].plot(range(obsers.shape[0]), obsers[:, 1])
    axs[1].plot(range(obsers.shape[0]), preds[:, 1])
    axs[1].legend(['observations', 'predictions'])
    axs[1].set_title('Agent 1')
    axs[1].set_xlabel('Epoch')
    plt.grid(True)
    
    display.display(plt.gcf())
    return fig
def get_images(obsers, preds):
    fig0 = plt.figure(0)
    plt.plot(obsers[-opt_params["batch_size"]:, 0])
    plt.plot(preds[-opt_params["batch_size"]:, 0])
    plt.legend(['observations', 'predictions'])
    plt.title('Agent 0')
    plt.xlabel('Epoch')
    fig0.canvas.draw()
    im0 = PIL.Image.frombytes('RGB', fig0.canvas.get_width_height(),fig0.canvas.tostring_rgb())
    plt.close()
    fig1 = plt.figure(1)
    plt.plot(obsers[-opt_params["batch_size"]:, 1])
    plt.plot(preds[-opt_params["batch_size"]:, 1])
    plt.legend(['observations', 'predictions'])
    plt.title('Agent 1')
    plt.xlabel('Epoch')
    fig1.canvas.draw()
    im1 = PIL.Image.frombytes('RGB', fig1.canvas.get_width_height(),fig1.canvas.tostring_rgb())
    plt.close()
    return im0, im1

# %%
obs, downlink_msgs = Task.reset()
epoch = 0
observations = []
predictions = []
while epoch<epochs:
    rs0 = torch.zeros(opt_params["batch_size"])
    rs1 = torch.zeros(opt_params["batch_size"])
    a_ps0_, a_ps1_, m_ps0_, m_ps1_= [], [], [], []
    agent0.optimizer.zero_grad()
    agent1.optimizer.zero_grad()
    for bt in range(opt_params["batch_size"]):
        
        a0, m0, a_ps0, m_ps0=  agent0.select_actionTraing(obs[0], downlink_msgs[0], bt)
        a1, m1, a_ps1, m_ps1=  agent1.select_actionTraing(obs[1], downlink_msgs[1], bt)
        
        a_ps0_.append(a_ps0[0, 0].item())
        a_ps1_.append(a_ps1[0, 0].item())
        mu_ = np.zeros((2, opt_params["vocab_size"]))
        mu_[0, m0.item()] = 1
        mu_[1, m1.item()] = 1
        
        predictions.append((a0.item(), a1.item()))
        (obs_, downlink_msgs_), r, done = Task.step([a0.item(), a1.item()], mu_)
        observations.append([obs_[0][1], obs_[1][0]])
        rs0[bt] = r[0]
        rs1[bt] = r[1]
    loss0 = agent0.train_online(rs0)
    loss1 = agent1.train_online(rs1)
    # only returns losses 
    obs = obs_
    downlink_msgs = downlink_msgs_
    if loss0 is not None:
        
        wandb.log({"policy loss A0": loss0[0], "value loss A0": loss0[1], \
                   "entropy loss A0": loss0[2],"reward A0": rs0.mean().item(), \
                   "policy loss A1": loss1[0], "value loss A1": loss1[1], \
                   "entropy loss A1": loss1[2], "reward A1": rs1.mean().item(),\
                   "mean policy A0": np.mean(a_ps0_), "mean policy A1": np.mean(a_ps1_)})
        

        if epoch%50==0:
            im0, im1 = get_images(np.array(observations), np.array(predictions))
            print("Training epoch ", epoch)
            table.add_data(epoch,  wandb.Image(im0),  wandb.Image(im1))
            run.log({"Batch Predictions": table})
        epoch+=1
        observations = []
        predictions = []

