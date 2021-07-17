import numpy as np
import matplotlib.pyplot as plt


class ToyTask():
    def __init__(self):
        self.hiddenStateField = None
        self.observationMappingFct = None
        self.comChannel = None
    def step(self, predictions, messages):
        observations = self.hiddenStateField.step()
        downlinkMsgs = self.comChannel.setMessages(messages)
        reward = self._getRewardFromObsPred(predictions, observations)
        done = False
        return (observations, downlinkMsgs), reward, done # s', r, done

    def reset(self):
        self.hiddenStateField.reset()
        self.comChannel.reset()

    def render(self):
        """
        plot obs state and a visualization of the communication channel
        """
        pass

class oneDfield():
    def __init__(self, agent_positions = (0.2, 0.8), n_channels=2, n_discrete=100, channels_propagation_dirs = (1, -1)):
        self.fields = np.zeros(n_discrete, n_channels)
        self.n_discrete = n_discrete
        self.t = 0
        self.agent_positions = np.array(agent_positions)
        self.channels_propagation_dirs = channels_propagation_dirs

    def _generatorFunction(self):
        pass
    
    def _propagate(self):
        new_values = self._generatorFunction()
        for i, channel in enumerate(self.fields):
            channel_ = channel.copy()
            if self.channels_propagation_dirs[i] == 1:
                channel_[1:] = channel_[:-1]
                channel_[0] = new_values[i]
            elif self.channels_propagation_dirs[i] == -1:
                channel_[:-1] = channel_[1:]
                channel_[-1] = new_values[i]
            self.fields[i] = channel_
        
    def step(self):
        agent_idxs = np.floor(self.n_discrete*self.agent_positions)
        self.t += 1
        self._propagate()
        
        return self.fields[agent_idxs]