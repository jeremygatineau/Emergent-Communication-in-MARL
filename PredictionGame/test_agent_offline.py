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
from agents.agent_AriaACs import AriaACs
from IPython import display
import wandb
import PIL
import matplotlib
matplotlib.use('Agg')

def train_function(opt_params):
    _log=True
    epochs = int(8e3)
    

    agent0 = AriaACs(opt_params=opt_params, split=True, with_memory=True, aidi=0)
    agent1 = AriaACs(opt_params=opt_params, split=True, with_memory=True, aidi=1)    
    np.random.seed(1)
    field = OneDfield(speed=1)
    Task = ToyTask(field=field,\
                observationMappingFct=lambda x: (x>0.5).astype(int), \
                comChannel=TwoWayComChannel(), \
                vocabulary_size=opt_params.vocab_size, \
                cross_r_coef=opt_params.cross_reward_coef)

    if _log: wandb.watch((agent0.modT, agent1.modT), log="all", log_freq=5)

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

    _is_converged = lambda rs: all(abs(rs - 1+opt_params.cross_reward_coef) < 0.2)
    _is_converged_streak = 0
    RMA_reward = np.array([0, 0])
    RMA_period = 1000
    RMA_coef = 1-1/RMA_period
    obs, downlink_msgs = Task.reset()
    epoch = 0
    observations = []
    predictions = []

    while epoch<epochs:
        rs0 = torch.zeros(opt_params.batch_size)
        rs1 = torch.zeros(opt_params.batch_size)
        a0, m0 = agent0.select_action(obs[0], downlink_msgs[0])
        a1, m1 = agent1.select_action(obs[1], downlink_msgs[1])
        mu_ = np.zeros((2, opt_params.vocab_size))
        mu_[0, m0.item()] = 1
        mu_[1, m1.item()] = 1

        predictions.append((a0.item(), a1.item()))
        (obs_, downlink_msgs_), r, done = Task.step([a0.item(), a1.item()], mu_)
        observations.append([obs_[0][1], obs_[1][0]])

        

        loss0, a_ps0_, rs0 = agent0.train_on_batch([obs[0], downlink_msgs[0]], r[0], (a0, m0))
        loss1, a_ps1_, rs1 = agent1.train_on_batch([obs[1], downlink_msgs[1]], r[1], (a1, m1))

        

        obs = obs_
        downlink_msgs = downlink_msgs_
        if loss0 is not None:
            RMA_reward = RMA_reward*RMA_coef + (1-RMA_coef)*np.array([rs0.mean().item(), rs1.mean().item()])
            _is_converged_streak = _is_converged_streak + 1 if _is_converged(RMA_reward) else 0
            if _is_converged_streak>200: pass
            if _log: wandb.log({"policy loss A0": loss0[0], "value loss A0": loss0[1], \
                    "entropy loss A0": loss0[2],"reward A0": rs0.mean().item(), \
                    "policy loss A1": loss1[0], "value loss A1": loss1[1], \
                    "entropy loss A1": loss1[2], "reward A1": rs1.mean().item(),\
                    "mean policy A0": np.mean(a_ps0_), "mean policy A1": np.mean(a_ps1_), "lr": agent0.scheduler.get_last_lr()[0]})
            

            if epoch%50==0:
                #im0, im1 = get_images(np.array(observations), np.array(predictions))
                print(f"Sweep with CR at {opt_params['cross_reward_coef']};  Training epoch {epoch}/{epochs}")
                #if _log: wandb.log({"batch_pred agent 0": wandb.Image(im0), "batch_pred agent 1": wandb.Image(im1)})
                
            epoch+=1
            observations = []
            predictions = []
    return epoch
# set up a sweep with wandb
def main():
    opt_params = {"lr":8e-3, "batch_size":32, \
                "gamma":0.99, "vocab_size":2, "training_loops":1, \
                "memory_size":20, "hidden_size": 30, "replay_size":32, \
                "eps":0.01, "grad_clamp":None, \
                "clip_c":0.2, "cross_reward_coef":0.3}
    run = wandb.init(config=opt_params, project='EC-MARL TOY PB', entity='jjer125')
    epochs = train_function(wandb.config)
    score = 5000/epochs - 1 # 0 if not converged otherwise positive
    wandb.log({"score": score})
if __name__=='__main__':
    sweep_configuration = {
        "method": "grid",
        "metric": {"goal": "maximize", "name":"score"},
        "parameters": {
            "cross_reward_coef": {"values": [0       , 0.16666667, 0.33333333, 0.5       , 0.66666667,
                                                0.83333333, 1.        , 1.16666667, 1.33333333, 1.5]}
        }
    }
    wandb.login()
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='EC-MARL TOY PB', entity='jjer125')
    #main()

    wandb.agent(sweep_id, function=main, count=10)

# %%
