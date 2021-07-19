import numpy as np
import matplotlib.pyplot as plt


class ToyTask():
    def __init__(self, field, observationMappingFct, comChannel):
        self.hiddenStateField = field
        self.observationMappingFct = observationMappingFct
        self.comChannel = comChannel

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
        self.hiddenStateField.reset()
        self.comChannel.reset()

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
    def setMessages(self, messages):
        self.downlinkMsgs = [msg for msg in reversed(messages)]
        return self.downlinkMsgs

class OneDfield():
    def __init__(self, speed=1, agent_positions = (0.2, 0.8), n_channels=2, n_discrete=100, channels_propagation_dirs = (1, -1)):
        self.fields = np.zeros((n_discrete, n_channels))
        self.n_discrete = n_discrete
        self.speed = speed
        self.t = 0
        self.agent_positions = np.array(agent_positions)
        self.channels_propagation_dirs = channels_propagation_dirs

    def _generatorFunction(self):
        return (0.6, 0.2)
    
    def _propagate(self):
        new_values = self._generatorFunction()
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
            new_values_array = np.array([self._generatorFunction() for _ in range(num)])
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
           
    def step(self):
        agent_idxs = np.floor(self.n_discrete*self.agent_positions).astype(int)

        self.t += 1
        self._propagateNSteps(self.speed)
        return self.fields[agent_idxs]