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


epochs = 10000
opt_params = {"lr":0.001, "training_loops":1, "batch_size":50, \
              "replay_size": 51, "gamma":0.99, "vocab_size":4, \
              "memory_size":8, "eps":0.001}
run = wandb.init(config=opt_params, project='EC-MARL TOY PB', entity='jjer125')

agent0 = AriaAC(opt_params=opt_params, with_memory=True, aidi=0)
agent1 = AriaAC(opt_params=opt_params, with_memory=True, aidi=1)
np.random.seed(1)
field = OneDfield(speed=1)
Task = ToyTask(field=field, observationMappingFct=lambda x: (x>0.5).astype(int), comChannel=TwoWayComChannel())

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
losses = []
rewards1 = []
rewards2 = []
epoch = 0
observations = []
predictions = []
while epoch<epochs:
    
    a0, m0u=  agent0.select_action(obs[0], downlink_msgs[0])
    a1, m1u= agent1.select_action(obs[1], downlink_msgs[1])
    mu_ = np.zeros((2, opt_params["vocab_size"]))
    mu_[0, m0u.item()] = 1
    mu_[1, m1u.item()] = 1
    
    predictions.append((a0.item(), a1.item()))
    (obs_, downlink_msgs_), r, done = Task.step([a0.item(), a1.item()], [mu_[0], mu_[1]])
    observations.append([obs_[0][1], obs_[1][0]])
    loss0, rew0, mean_policy0 = agent0.train_on_batch([obs[0], Task.initMsgs[0]], r[0])
    loss1, rew1, mean_policy1 = agent1.train_on_batch([obs[1], Task.initMsgs[1]], r[1])
    obs = obs_
    downlink_msgs = downlink_msgs_
    if loss0 is not None:
        if epoch==2 and (mean_policy0.item()-0.5 == 0. or mean_policy1.item()-0.5 == 0.):
            epoch = 0
            agent0 = AriaAC(opt_params=opt_params, with_memory=True, aidi=0)
            agent1 = AriaAC(opt_params=opt_params, with_memory=True, aidi=1)
            print('Mean policies not updating, try restarting')
        wandb.log({"policy loss A0": loss0[0], "value loss A0": loss0[1], \
                   "entropy loss A0": loss0[2],"reward A0": np.mean(rew0), \
                   "policy loss A1": loss1[0], "value loss A1": loss1[1], \
                   "entropy loss A1": loss1[2], "reward A1": np.mean(rew1),\
                   "mean policy A0": mean_policy0, "mean policy A1": mean_policy1})
        

        losses.append([loss0, loss1])
        rewards1.append(rew0)
        rewards1.append(rew1)
        if epoch%50==0:
            im0, im1 = get_images(np.array(observations), np.array(predictions))
            table = wandb.Table(columns=["Epoch#", "batch_pred A0", "batch_pred A1"])
            print("Training epoch ", epoch)
            table.add_data(epoch,  wandb.Image(im0),  wandb.Image(im1))
            run.log({"Batch Predictions": table})
        epoch+=1
        observations = []
        predictions = []

