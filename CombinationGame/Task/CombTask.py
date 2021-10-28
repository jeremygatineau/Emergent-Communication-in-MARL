# imports for the gym, pytorch and numpy
import gym 
import torch
import numpy as np

# define custom environment class from gym
class CombinationGame(gym.Env):
    def __init__(self, number_of_agents, grid_size=10, max_obj_per_type=5):
        self.max_obj_per_type = max_obj_per_type
        self.grid_size = grid_size
        self.number_of_agents = number_of_agents
        # define the action space, 7 possible actions
        self.action_space = gym.spaces.Discrete(7)
        # define the state space, grid of size 7x10x10
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(7, 10, 10))
        # define the reward space, continuous
        
        self.entity_list = []
        # define the grid as an empty np.array of size 10x10x(4+max_obj_per_type)
        self.grid = np.zeros((self.grid_size, self.grid_size, 7+max_obj_per_type))
        self._object_counter = np.zeros(7+self.max_obj_per_type, dtype=int)
        self.entity_string_doc = {
            0: "m+", # held movable object
            1: "m-", # movable object
            2: "a", # agent
            3: "l", # landmark
            4: "i+", # indicator on
            5: "i-", # indicator off
            6: "X", # wall
        }
        self.entity_string_doc_reverse = {
            "m+": 0,
            "m-": 1,
            "a": 2,
            "l": 3,
            "i+": 4,
            "i-": 5,
            "X": 6,
        }
        self.entity_string_doc_reverse_color = {
            "m+": "red",
            "m-": "green",
            "a": "blue",
            "l": "yellow",
            "i+": "white",
            "i-": "black",
            "X": "grey",
        }
        self.entity_string_doc_reverse_color_rgb = {
            "m+": (255, 0, 0),
            "m-": (0, 255, 0),
            "a": (0, 0, 255),
            "l": (255, 255, 0),
            "i+": (255, 255, 255),
            "i-": (0, 0, 0),
            "X": (128, 128, 128),
        }

    
    def _get_object_string_from_vector(self, vector):
        """
            gets the object string representation from a vector
            inputs :
                vector : the vector representation of the object
            outputs :
                object_string : the object string representation
        """
        # define the object string
        object_string = ""
        # if vector is all zeros object_string is "  "
        if np.any(vector):
            entity_type = self.entity_string_doc[np.argmax(vector[0:7])]
            object_string += entity_type + str(np.argmax(vector[7:self.max_obj_per_type+7]))
        else:
            object_string = "  "

        return object_string
    
    def _get_object_vector_from_string(self, object_string):
        """
            Returns the vector representation of the object
            inputs :   
                object_string : the object string representation
            outputs :
                vector : the vector representation of the object, of size 7+max_obj_per_type which includes the object type and number
        """
        # define the vector
        vector = np.zeros(7+self.max_obj_per_type)

        # get the object type
        object_type = self.entity_string_doc_reverse[object_string[0:2]] if len(object_string) > 2 else self.entity_string_doc_reverse[object_string[0]]
        object_id = int(object_string[2:]) if len(object_string) > 2 else int(object_string[1])
        
        # set the obeject type
        vector[object_type] = 1
        # set the object id
        vector[7+object_id] = 1
        return vector

    def initialize_entity_list(self, entity_list):
        """
            initializes the entity list
            inputs :
                entity_list : list of objects to be placed in the environment, each object is a list of size 3
                    [object_type, object_position_x, object_position_y]
            outputs : None
        """
        self.entity_list = entity_list

    def _remove_position_and_neighbors(self, position, indices):
        """
            removes the position and its neighbors from the list of available positions
            inputs :
                position : the position to be removed
                indices : the available positions as given by np.indices
            outputs :
                indices : the updated indices
        """
        object_pos_x = position[0]
        object_pos_y = position[1]
        
        np.delete(indices, (object_pos_x, object_pos_y))
        # try except blocks to handle the cases when the remove argument is not in the indices
        try:
            np.delete(indices, (object_pos_x-1, object_pos_y))
        except:
            pass
        try:
            np.delete(indices, (object_pos_x+1, object_pos_y))
        except:
            pass
        try:
            np.delete(indices, (object_pos_x, object_pos_y-1))
        except:
            pass
        try:
            np.delete(indices, (object_pos_x, object_pos_y+1))
        except:
            pass
        try:
            np.delete(indices, (object_pos_x-1, object_pos_y-1))
        except:
            pass
        try:
            np.delete(indices, (object_pos_x-1, object_pos_y+1))
        except:
            pass
        try:
            np.delete(indices, (object_pos_x+1, object_pos_y-1))
        except:
            pass
        try:
            np.delete(indices, (object_pos_x+1, object_pos_y+1))
        except:
            pass
        return indices
    def _generate_random_entity_list(self, total_number_of_objects):
        """
            generates a random entity list for the task
            inputs :
                total_number_of_objects : the total number of objects to be placed in the environment
            outputs :
                entity_list : list of objects to be placed in the environment, each object is an entity object
        """
        # define the entity list
        entity_list = []
        
        # initiate list of unoccupied indices
        indices = np.array(np.unravel_index(range(self.grid_size*self.grid_size), (self.grid_size, self.grid_size))).T
        # initialize object counters for each object type
        self._object_counter = np.zeros(7+self.max_obj_per_type, dtype=int)
        for i in range(total_number_of_objects):
            # generate a random object type that is not a wall (object type 6) nor an agent (object type 2)
            object_type = np.random.choice([1,3,4])
            # if number of objects of that type is already equal to max_obj_per_type, generate another object type
            while self._object_counter[object_type] == self.max_obj_per_type:
                object_type = np.random.choice([1,3,5])
            # increment corresponding object counter
            self._object_counter[object_type] += 1

            # if object is an indicator
            if object_type == 4 or object_type == 5:
                # choose a random position from the unoccupied indices list and make sure it is on the border
                object_pos_x, object_pos_y = indices[np.random.randint(0, indices.shape[0])]
                while object_pos_x == 0 or object_pos_x == self.grid_size-1 or object_pos_y == 0 or object_pos_y == self.grid_size-1:
                    object_pos_x, object_pos_y = indices[np.random.randint(0, indices.shape[0])]
                # remove the chosen position as well as the neighboring grid cells from the unoccupied indices list
                indices = self._remove_position_and_neighbors((object_pos_x, object_pos_y), indices)

            # if object is not a wall
            elif object_type != 6:
                # choose a random position from the unoccupied indices list
                object_pos_x, object_pos_y = indices[np.random.randint(0, indices.shape[0])]
                # remove the chosen position as well as the neighboring grid cells from the unoccupied indices list
                indices = self._remove_position_and_neighbors((object_pos_x, object_pos_y), indices)
            # if object is a wall
            else:
                pass # no walls yet
            # create the entity
            entity = Entity(object_type, (object_pos_x, object_pos_y), self._object_counter[object_type])
            # add the entity to the entity list
            entity_list.append(entity)
        # place one or two agents, depending on the self.number_of_agent parameter
        for i in range(self.number_of_agents):
            # choose a random position from the unoccupied indices list
            object_pos_x, object_pos_y = indices[np.random.randint(0, indices.shape[0])]
            # remove agent's position from the indices list
            np.delete(indices, (object_pos_x, object_pos_y))
            # create the entity
            entity = Entity(2, (object_pos_x, object_pos_y), i+1)
            # add the entity to the entity list
            entity_list.append(entity)
        return entity_list

    def place_entity_list_in_grid(self):
        """
            place each entity in the environment with the corresponding object string representation
            inputs : None
            outputs : None
        """
        # iterate over the entity list
        for entity in self.entity_list:
            # get object position
            object_pos_x = entity.get_object_pos_x()
            object_pos_y = entity.get_object_pos_y()            
            # set the object vector representation
            object_vector = self._get_object_vector_from_string(str(entity))
            
            self.grid[object_pos_x][object_pos_y] = object_vector

    def reset(self, total_number_of_objects=3):
        """
            resets the environment state
            inputs : None
            outputs :
                state : the initial state of the environment
        """
        # reset the grid
        self.grid = np.zeros((self.grid_size, self.grid_size, 7+self.max_obj_per_type))
        # generate entity list
        self.entity_list = self._generate_random_entity_list(total_number_of_objects)
        print(self.entity_list)
        # place the entities in the grid
        self.place_entity_list_in_grid()
        # return the initial state
        return self.grid
        

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

    def move_agent(self, action):
        """
            moves the agent in the environment
            inputs :
                action : the action taken by the agent
            outputs :
                next_state : the next state of the environment
        """
        # define the next state
        next_state = self.state
        # get the agent position
        agent_pos_x = next_state[0][0][0]
        agent_pos_y = next_state[0][0][1]
        # check if the agent is at the edge of the grid
        if agent_pos_x == 0:
            # if the agent is at the edge of the grid, move the agent to the right
            next_state[0][0][0] = next_state[0][0][0] + 1
        elif agent_pos_x == 9:
            # if the agent is at the edge of the grid, move the agent to the left
            next_state[0][0][0] = next_state[0][0][0] - 1
        elif agent_pos_y == 0:
            # if the agent is at the edge of the grid, move the agent down
            next_state[0][0][1] = next_state[0][0][1] + 1
        elif agent_pos_y == 9:
            # if the agent is at the edge of the grid, move the agent up
            next_state[0][0][1] = next_state[0][0][1] - 1
        
    def render(self):
        """
            renders the environment, returns string representing the current state
            inputs : None
            outputs : 
                s : string representing the current grid with numbers corresponding to the object type in the for each grid cell. 
        """
        # define the string to be returned
        s = ""
        # iterate over the grid, add object types, separate rows with "-" and columns with "|"
        for i in range(self.grid_size):
            # add a row of 4*self.grid_size+1 "-", 3 for the cells 1 for the separator 
            s += "-"*(1+4*self.grid_size) + "\n"
            for j in range(self.grid_size):
                # add separators to the left
                if j==0:
                    s += "|"

                object_string = self._get_object_string_from_vector(self.grid[i][j])
                s += object_string
                s += " " if len(object_string)==2 else ""
                s += "|"
            s += "\n"
        s += "-"*(1+4*self.grid_size) + "\n"
        # return the string
        return s
    
class Entity:
    def __init__(self, object_type, object_pos, object_id):
        """
            initializes the entity
            inputs :
                object_type : the object type of the entity
                object_pos : the object position of the entity
        """
        self.object_type = object_type
        self.object_pos = object_pos
        self.object_id = object_id
        self.entity_string_doc = {
            0: "m+", # held movable object
            1: "m-", # movable object
            2: "a", # agent
            3: "l", # landmark
            4: "i+", # indicator on
            5: "i-", # indicator off
            6: "X", # wall
        }
    
    def get_object_type(self):
        return self.object_type
    def get_object_pos_x(self):
        return self.object_pos[0]
    def get_object_pos_y(self):
        return self.object_pos[1]

    # string representation of the object
    def __str__(self):
        return self.entity_string_doc[self.object_type] + str(self.object_id)

    def __rep__(self):
        return self.entity_string_doc[self.object_type] + str(self.object_id)

# test the random generation and render the environment
if __name__ == "__main__":
    # create an environment
    env = CombinationGame(1)
    

    # reset and render the environment for 1 to 10 objects
    for i in range(1,10):
        # reset the environment
        env.reset(i)
        # render the environment
        print(env.render())