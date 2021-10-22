# imports for the gym, pytorch and numpy
import gym 
import torch
import numpy as np

# define custom environment class from gym
class CombinationGame(gym.Env):
    def __init__(self):
        # define the action space, 7 possible actions
        self.action_space = gym.spaces.Discrete(7)
        # define the state space, grid of size 7x10x10
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(7, 10, 10))
        # define the reward space, continuous
        self.reward_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        # define the state of the environment
        self.state = np.zeros((7, 10, 10))
        
    def place_objects(self, object_list):
        """
            places objects from object_list in the initial environment state
            inputs : 
                object_list : list of objects to be placed in the environment, each object is a list of size 3
                    [object_type, object_position_x, object_position_y]
            outputs :
                None
        """
        # iterate over the objects
        for obj in object_list:
            # get the object type and position
            obj_type = obj[0]
            obj_pos_x = obj[1]
            obj_pos_y = obj[2]
            # place the object in the environment state
            self.state[obj_type][obj_pos_x][obj_pos_y] = 1

    def reset(self):
        """
            resets the environment state
            inputs : None
            outputs :
                state : the initial state of the environment
        """
        # define the initial state of the environment
        self.state = np.zeros((7, 10, 10))
        # generate object list
        object_list = self.generate_object_list()
        # place objects in the environment
        self.place_objects(object_list)
        # return the initial state
        return self.state

    def generate_object_list(self):
        """
            generates a list of objects to be placed in the environment
            inputs : None
            outputs :
                object_list : list of objects to be placed in the environment, each object is a list of size 3
                    [object_type, object_position_x, object_position_y]
        """
        pass

    def step(self, action):
        """
            takes an action in the environment, returns the next state and reward. Actions 0-3 are for moving the agent. Action 4 staying in place. Action 5 and 6 are for picking up and placing objects respectively.
            inputs :
                action : the action taken by the agent
            outputs :
                next_state : the next state of the environment
                reward : the reward received by the agent
                done : boolean indicating whether the episode is over
        """
        # define the next state
        next_state = self.state
        # define the reward
        reward = 0
        # define the done flag
        done = False
        # check if the action is within the action space
        if action < 4:
            # if the action is within the action space, move the agent
            next_state = self.move_agent(action)
        elif action == 4:
            # if the action is 4, stay in place
            pass
        elif action == 5:
            # if the action is 5, pick up an object
            next_state = self.pick_up_object()
        elif action == 6:
            # if the action is 6, place an object
            next_state = self.place_object()
        # check if the episode is over
        if self.check_episode_over():
            # if the episode is over, set the done flag to true
            done = True
        # return the next state, reward and done flag
        return next_state, reward, done