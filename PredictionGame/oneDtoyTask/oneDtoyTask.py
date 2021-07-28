import numpy as np
import matplotlib.pyplot as plt


class ToyTask():
    def __init__(self, field, observationMappingFct, comChannel):
        self.hiddenStateField = field
        self.observationMappingFct = observationMappingFct
        self.comChannel = comChannel
        self.initMsgs = [np.zeros(4), np.zeros(4)]
    def step(self, predictions, messages):
        nxtObs =  self.hiddenStateField.step()
        observations = np.apply_along_axis(self.observationMappingFct, -1, nxtObs)
        downlinkMsgs = self.comChannel.setMessages(messages)
        rewards = self._getRewardsFromObsPred(predictions, observations)
        done = False
        return (observations, downlinkMsgs), rewards, done # s', r, done
        
    def _getRewardsFromObsPred(self, pred, obs):
        rewards = [0, 0]
        rewards[0] = int(pred[0] == obs[0][1])
        rewards[1] = int(pred[1] == obs[1][0])
        return rewards
    def reset(self):
        nxtObs = self.hiddenStateField.reset()
        self.comChannel.reset()
        observations = np.apply_along_axis(self.observationMappingFct, -1, nxtObs)
        downlinkMsgs = self.comChannel.setMessages(self.initMsgs)

        return (observations, downlinkMsgs)
    def render(self):
        """
        plot obs state and a visualization of the communication channel
        """
        fig, axs= plt.subplots()
        axs.plot(self.hiddenStateField.fields)
        return fig, axs


class TwoWayComChannel():
    def __init__(self):
        self.downlinkMsgs = []
    def reset(self):
        self.downlinkMsgs = []
        return self.downlinkMsgs
    def setMessages(self, messages):
        self.downlinkMsgs = [msg for msg in reversed(messages)]
        return self.downlinkMsgs

class OneDfield():
    def __init__(self, speed=1, agent_positions = (0.2, 0.8), n_channels=2, n_discrete=25, channels_propagation_dirs = (1, -1)):
        self.fields = np.zeros((n_discrete, n_channels))
        self.n_discrete = n_discrete
        self.n_channels = n_channels
        self.speed = speed
        self.gen_fct_state = [0, 4]
        self.onoff_dutyCyle = [(2, 20), 3]
        self.agent_positions = np.array(agent_positions)
        self.channels_propagation_dirs = channels_propagation_dirs
        self.agent_idxs = np.floor(self.n_discrete*self.agent_positions).astype(int)
    def _generatorFunction(self, hidden_var=None):
        if self.gen_fct_state[0]>0 and self.gen_fct_state[1]==0:
            self.gen_fct_state[0] -= 1
            return (1, 1)
        elif self.gen_fct_state[0] == 0 and self.gen_fct_state[1]==0:
            
            self.gen_fct_state[1] = np.random.randint(self.onoff_dutyCyle[0][0], self.onoff_dutyCyle[0][1])
            self.gen_fct_state[0] = self.onoff_dutyCyle[1]
            return (0, 0)
        elif self.gen_fct_state[1]>0:
            self.gen_fct_state[1] -= 1
            return (0, 0)
    
    def _propagate(self):
        new_values = self._generatorFunction(self.t)
        for i, channel in enumerate(self.fields.T):
            channel_ = channel.copy()
            if self.channels_propagation_dirs[i] == 1:
                channel_[1:] = channel_[:-1]
                channel_[0] = new_values[i]
            elif self.channels_propagation_dirs[i] == -1:
                channel_[:-1] = channel_[1:]
                channel_[-1] = new_values[i]
            self.fields[:, i] = channel_
    def _propagateNSteps(self, n):
        def prop(num):
            new_values_array = np.array([self._generatorFunction() for dt in range(num)])
            for i, channel in enumerate(self.fields.T):
                channel_ = channel.copy()
                if self.channels_propagation_dirs[i] == 1:
                    channel_[num:] = channel_[:-num]
                    channel_[:num] = list(reversed(new_values_array[:, i]))
                elif self.channels_propagation_dirs[i] == -1:
                    channel_[:-num] = channel_[num:]
                    channel_[-num:] = new_values_array[:, i]
                self.fields[:, i] = channel_
        k=n
        while k>=self.n_discrete:
            prop(self.n_discrete-1)
            k -= self.n_discrete-1
        prop(k)
    def reset(self):
        self.fields = np.zeros((self.n_discrete, self.n_channels))
        self.gen_fct_state = [0, 4]
        return self.fields[self.agent_idxs]
    def step(self):
        self._propagateNSteps(self.speed)
        return self.fields[self.agent_idxs]