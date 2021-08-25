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


epochs = 1000
opt_params = {"lr":0.01, "training_loops":1, "batch_size":20, \
              "replay_size": 3, "gamma":0.9, "vocab_size":4, \
              "memory_size":8, "eps":0.001}
run = wandb.init(config=opt_params, project='EC-MARL TOY PB', entity='jjer125')

agent0 = AriaAC(opt_params=opt_params, with_memory=True, aidi=0)
np.random.seed(1)
field = OneDfield(speed=1)
Task = ToyTask(field=field, observationMappingFct=lambda x: (x>0.5).astype(int), comChannel=TwoWayComChannel())

wandb.watch((agent0.modT), log="all", log_freq=5)
table = wandb.Table(columns=["Epoch#", "batch_pred A0"])
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

    plt.grid(True)
    
    display.display(plt.gcf())
    return fig
def get_images(obsers, preds):
    fig0 = plt.figure(0)
    plt.plot(obsers[-opt_params["batch_size"]:])
    plt.plot(preds[-opt_params["batch_size"]:])
    plt.legend(['observations', 'predictions'])
    plt.title('Agent 0')
    plt.xlabel('Epoch')
    fig0.canvas.draw()
    im0 = PIL.Image.frombytes('RGB', fig0.canvas.get_width_height(),fig0.canvas.tostring_rgb())
    plt.close()
    
    return im0

# %%
obs, downlink_msgs = Task.reset()
losses = []
rewards1 = []
epoch = 0
observations = []
predictions = []
torch.autograd.set_detect_anomaly(True)
hid_state = torch.zeros(1, 2*agent0.memory_size)
while epoch<epochs:
    rs = torch.zeros(opt_params["batch_size"])
    a_lps = torch.zeros(opt_params["batch_size"])
    m_lps = torch.zeros(opt_params["batch_size"])
    vals = torch.zeros(opt_params["batch_size"])
    hid_states = [hid_state.detach()]
    init_state = hid_state.detach()
    agent0.optimizer.zero_grad()
    for bt in range(opt_params["batch_size"]):
        
        hs = hid_states[-1]
        a, m, a_lp, m_lp, val, hid_state, ent, a_ps, m_ps=  agent0.select_actionTraing(obs[0], Task.initMsgs[0], hs)
        
        hid_states.append(hid_state)
        
        predictions.append(a.item())
        (obs_, downlink_msgs_), r, done = Task.step([a.item(), None], [m, None])
        observations.append(obs_[0][1])
        rs[bt] = r[0]
        a_lps[bt] = a_lp
        m_lps[bt] = m_lp
        vals[bt] = val
    loss0 = agent0.train_online(rs, a_lps, m_lps, vals)

    obs = obs_
    downlink_msgs = downlink_msgs_
    if loss0 is not None:
        
        wandb.log({"policy loss A0": loss0[0], "value loss A0": loss0[1], \
                   "entropy loss A0": loss0[2],"reward A0": r[0], \
                   "mean policy A0": a_ps[0][0]})
        

        losses.append(loss0)
        rewards1.append(r[0])
        if epoch%50==0:
            im0 = get_images(np.array(observations), np.array(predictions))
            table = wandb.Table(columns=["Epoch#", "batch_pred A0"])
            print("Training epoch ", epoch)
            table.add_data(epoch,  wandb.Image(im0))
            run.log({"Batch Predictions": table})
        epoch+=1
        observations = []
        predictions = []

