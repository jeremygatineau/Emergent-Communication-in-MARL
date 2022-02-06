# imports for the gym, pytorch and numpy
import gym 
import torch
import numpy as np
from ProgressionTree import ProgressionTree
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
        self.grid_indices = np.zeros((self.grid_size, self.grid_size))
        self._object_counter = np.zeros(7+self.max_obj_per_type, dtype=int)
        tree = ProgressionTree()
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
        self.possible_combinations = {"g":["ii", "lm", "mm", "ml"], "i": ["ii", "lm", "mm", "ml"], "l":["mm"], "m":["mm", "ml", "lm"]}

        self.tree = ProgressionTree()
        self.goal_state_vector = np.ones(7+self.max_obj_per_type)
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
        # if vector is all ones object represent te goal state
        if np.all(vector==1):
            return "g"
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
        # if object is goal state then return a vector with all ones
        if object_string == "g":
            return np.ones(7+self.max_obj_per_type)
        # define the vector
        vector = np.zeros(7+self.max_obj_per_type)

        # get the object type
        object_type = self.entity_string_doc_reverse[object_string[0:2]] if len(object_string) > 2 else self.entity_string_doc_reverse[object_string[0]]
        object_id = int(object_string[2:]) if len(object_string) > 2 else int(object_string[1])
        # set the obeject type
        vector[object_type] = 1
        # set the object id
        print(object_id)
        vector[7+object_id-1] = 1
        return vector
    def get_branching_ps_from_diff(self, difficulty):
        return [1, 1, 0.5, 0.5]
    def initialize_progression_tree(self, difficulty):
        branching_probabilities = self.get_branching_ps_from_diff( difficulty)
        self.tree.generate_tree(branching_probabilities)
        self.tree.set_node_ids()
        children_is_leaf_mask = self.tree._leaf_node_mask()
        object_counter = np.zeros(7+self.max_obj_per_type, dtype=int)
        for depth in range(self.tree.get_max_depth()):
            print("depth: ", depth, "nodes: ", self.tree.get_nodes_by_depth(depth))
            print("depth nodes strings: ", list(map(lambda l: self._get_object_string_from_vector(l), map(lambda n: n.value, self.tree.get_nodes_by_depth(depth)))))
            for node in self.tree.get_nodes_by_depth(depth):
                if node.value is None:
                    continue
                if node.parent is None:
                    self.tree.set_node_p(node.id, self.goal_state_vector, "value")
                print("value: ", node.value, "id: ", node.id)
                if self.tree._is_node_leaf(node):
                    continue
                node_type = self._get_object_string_from_vector(node.value)[0]
                combs = list(np.copy(self.possible_combinations[node_type]))
                if children_is_leaf_mask[node.id]: 
                    combs = [comb for comb in combs if comb!="ii"]
                print("\tNode: ", node_type, " Possible Combinations: ",combs, " Is_Leaf? ", children_is_leaf_mask[node.id])
                comb = np.random.choice(combs)
                found_comb=False
                potential_child_counter = np.copy(object_counter)
                while not found_comb:
                    comb_tuple = [comb[0], comb[1]]
                    for i, obj in enumerate(comb_tuple):                        
                        if obj == "m" or obj=="i":
                            comb_tuple[i] += "-"
                    print("YOOOOOOO 1 ", object_counter, potential_child_counter)

                    for i, obj in enumerate(comb_tuple): 
                        potential_child_counter[self.entity_string_doc_reverse[obj]] += 1
                    print("YOOOOOOO 2 ", object_counter, potential_child_counter)
                    if np.all(potential_child_counter <= self.max_obj_per_type):
                        print("\t\tFound Combination: ", comb_tuple)
                        break
                    else:
                        print("\t\tCombination ", comb_tuple," doesn't work, trying again")
                        combs.remove(comb)
                        if len(combs)==0:
                            print("\t\tNo more combinations, pruning node children")
                            self.tree.set_node_p(node.id, None, "Children")
                            print("This is supposed to be None: ", node.children)
                            break
                        comb = np.random.choice(combs)
                        potential_child_counter = np.copy(object_counter)
                    
                    
                            
                if self.tree._is_node_leaf(node):
                    continue
                print("not suppose to be printed") if node.children is None else print("\tFound Combination: ", comb)
                cobj = ["", ""]
                for i, obj in enumerate(comb): 
                    cobj[i]= obj
                    
                    if obj == "m" or obj=="i":
                        cobj[i] += "-"
                    object_type = self.entity_string_doc_reverse[cobj[i]]
                    # decide whether to reuse landmark or not
                    if obj == "l":
                        reuse = np.random.randint(1, object_counter[object_type]+2)
                        if reuse < object_counter[object_type]:
                            cobj[i] += str(reuse)
                        else:
                            cobj[i] += str(object_counter[object_type]+1)
                            object_counter[object_type] += 1
                    else: 
                        cobj[i] += str(object_counter[object_type]+1)
                        object_counter[object_type] += 1
                    node.children[i].value = self._get_object_vector_from_string(cobj[i])

    def _get_object_type_and_id_from_string(self, object_string):
        """
            Returns the object type and id from the object string
            inputs :   
                object_string : the object string representation
            outputs :
                object_type : the object type
                object_id : the object id
        """
        object_type = self.entity_string_doc_reverse[object_string[0:2]] if len(object_string) > 2 else self.entity_string_doc_reverse[object_string[0]]
        object_id = int(object_string[2:]) if len(object_string) > 2 else int(object_string[1])
        return object_type, object_id
    def initialize_entity_list(self, entity_list):
        """
            initializes the entity list
            inputs :
                entity_list : list of objects to be placed in the environment, each object is a list of size 3
                    [object_type, object_position_x, object_position_y]
            outputs : None
        """
        self.entity_list = entity_list
    def _delete_position(self, indices, pos):
        new_indices = []
        for i in range(len(indices)):
            if all(indices[i] == pos):
                continue
            new_indices.append(indices[i])
        return new_indices
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
        offsets = [[0, 1], [1, 0], [0, -1], [-1, 0], [1, 1], [-1, 1], [1, -1], [-1, -1]]
        indices = self._delete_position(indices, (object_pos_x, object_pos_y))
        for offset in offsets:
            # try except blocks to handle the cases when the remove argument is not in the indices

            try:
                indices = self._delete_position(indices, position + np.array(offset))
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
        assert total_number_of_objects<= self.max_obj_per_type*3, "total_number_of_objects should be less than 3*max_obj_per_type"
        # initiate list of unoccupied indices
        indices = np.array(np.unravel_index(range(self.grid_size*self.grid_size), (self.grid_size, self.grid_size))).T
        # initialize object counters for each object type
        self._object_counter = np.zeros(7, dtype=int)
        for i in range(total_number_of_objects):
            if len(indices) == 0:
                print(f"no more room to place objects, only {i} objects have been placed")
                break
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
                object_pos_x, object_pos_y = indices[np.random.randint(0, len(indices))]
                i = 0
                while object_pos_x == 0 or object_pos_x == self.grid_size-1 or object_pos_y == 0 or object_pos_y == self.grid_size-1:
                    object_pos_x, object_pos_y = indices[np.random.randint(0, len(indices))]
                    i+=1
                    if i>50:
                        print("could not place object")
                        break
                # remove the chosen position as well as the neighboring grid cells from the unoccupied indices list
                indices = self._remove_position_and_neighbors((object_pos_x, object_pos_y), indices)

            # if object is not a wall
            elif object_type != 6:
                
                # choose a random position from the unoccupied indices list
                object_pos_x, object_pos_y = indices[np.random.randint(0, len(indices))]
                # while loop to make sure position is not on the bottom row meaning that object_pos_x is not 0
                i = 0
                while object_pos_x == self.grid_size-1:
                    object_pos_x, object_pos_y = indices[np.random.randint(0, len(indices))]
                    i+=1
                    if i>50:
                        print("could not place object")
                        break
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
            object_pos_x, object_pos_y = indices[np.random.randint(0, len(indices))]
            # remove agent's position from the indices list
            self._delete_position(indices, (object_pos_x, object_pos_y))
            # create the entity
            entity = Entity(2, (object_pos_x, object_pos_y), i+1)
            # add the entity to the entity list
            entity_list.append(entity)
        return entity_list
    def _generate_entity_list_from_tree(self):
        """
            Goes thorugh the progression tree and generates the entity list based on the node objects
        """
        nodes_strings_distinct = list(set([self._get_object_string_from_vector(node.value) for node in self.tree.get_leaves()]))
        entity_list = []
        # initiate list of unoccupied indices
        indices = np.array(np.unravel_index(range(self.grid_size*self.grid_size), (self.grid_size, self.grid_size))).T
        print(f"{nodes_strings_distinct=}")
        for string_obj in nodes_strings_distinct:
            object_type, object_id = self._get_object_type_and_id_from_string(string_obj)
            if len(indices) == 0:
                print(f"no more room to place objects, only {i} objects have been placed")
                break
            # if object is an indicator
            if object_type == 4 or object_type == 5:
                # choose a random position from the unoccupied indices list and make sure it is on the border
                object_pos_x, object_pos_y = indices[np.random.randint(0, len(indices))]
                i = 0
                while object_pos_x == 0 or object_pos_x == self.grid_size-1 or object_pos_y == 0 or object_pos_y == self.grid_size-1:
                    object_pos_x, object_pos_y = indices[np.random.randint(0, len(indices))]
                    i+=1
                    if i>50:
                        print("could not place object")
                        break
                # remove the chosen position as well as the neighboring grid cells from the unoccupied indices list
                indices = self._remove_position_and_neighbors((object_pos_x, object_pos_y), indices)

            # if object is not a wall
            elif object_type != 6:
                
                # choose a random position from the unoccupied indices list
                object_pos_x, object_pos_y = indices[np.random.randint(0, len(indices))]
                # while loop to make sure position is not on the bottom row meaning that object_pos_x is not 0
                i = 0
                while object_pos_x == self.grid_size-1:
                    object_pos_x, object_pos_y = indices[np.random.randint(0, len(indices))]
                    i+=1
                    if i>50:
                        print("could not place object")
                        break
                # remove the chosen position as well as the neighboring grid cells from the unoccupied indices list
                indices = self._remove_position_and_neighbors((object_pos_x, object_pos_y), indices)
            # if object is a wall
            else:
                pass # no walls yet
            # create the entity
            entity = Entity(object_type, (object_pos_x, object_pos_y), object_id)
            # add the entity to the entity list
            entity_list.append(entity)
        # place one or two agents, depending on the self.number_of_agent parameter
        for i in range(self.number_of_agents):
            # choose a random position from the unoccupied indices list
            object_pos_x, object_pos_y = indices[np.random.randint(0, len(indices))]
            # remove agent's position from the indices list
            self._delete_position(indices, (object_pos_x, object_pos_y))
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
        for index, entity in enumerate(self.entity_list):
            # get object position
            object_pos_x = entity.get_object_pos_x()
            object_pos_y = entity.get_object_pos_y()            
            # set the object vector representation
            object_vector = self._get_object_vector_from_string(str(entity))
            
            self.grid[object_pos_x][object_pos_y] = object_vector
            self.grid_indices[object_pos_x][object_pos_y] = index

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
        print([str(entity) for entity in self.entity_list])
        # place the entities in the grid
        self.place_entity_list_in_grid()
        # return the initial state
        return self.grid
        

    def step(self, actions):
        pass
        

    def pick_up_object(self, agent_id):
        """
            pick up an object, modify the entity list and the grid
            inputs :
                agent_id : the id of the agent
            outputs :
                None
        """
        # get the agent's position
        agent_pos_x, agent_pos_y = self.get_agent_pos(agent_id)

        # check if the object on the above grid cell is movable
        object_above_string = self._get_object_string_from_vector(self.grid[agent_pos_x-1][agent_pos_y])
        if object_above_string[:2] == 'm-':
            # if movable, change object to held and assign it to the agent
            self.grid[agent_pos_x-1][agent_pos_y] = self._get_object_vector_from_string('m+'+object_above_string[2:])
            self.assign_object_to_agent((agent_pos_x, agent_pos_y), self.grid_indices[agent_pos_x-1][agent_pos_y])
            # modify entity_list
            self.entity_list[self.grid_indices[agent_pos_x-1][agent_pos_y]].object_type = 0
    
    def assign_object_to_agent(self, agent_pos, object_index):
        """
            assign an object to an agent in the entity_list
            inputs :
                agent_pos : the postion of the agent in the grid
                object_index : the index of the object in the entity_list
            outputs :
                None
        """
        # get the agent's entity_list index
        agent_index = self.grid_indices[agent_pos[0], agent_pos[1]]
        self.entity_list[agent_index].assign_object(object_index)
    def unassign_object_from_agent(self, agent_pos):
        """
            unassign an object from an agent in the entity_list
            inputs :
                agent_id : the id of the agent
            outputs :
                None
        """
        agent_index = self.grid_indices[agent_pos[0], agent_pos[1]]
        self.entity_list[agent_index].assign_object(None)
    def place_object(self, agent_id):
        """
            place an object, modify the entity list and the grid
            inputs :
                agent_id : the id of the agent
            outputs :
                None
        """
        # get the agent's position
        agent_pos_x, agent_pos_y = self.get_agent_pos(agent_id)
        # check if the above grid cell on is empty
        if self.grid_indices[agent_pos_x-1][agent_pos_y] is None:
            # get the held object's entity_list index by looking at the agent's assigned_object_id
            object_index = self.entity_list[self.grid_indices[agent_pos_x][agent_pos_y]].get_assigned_object_id()
            # get the entity object from the entity list
            pickup_position = self.entity_list[object_index].get_object_pos()
            # remove the object from it's pick up location
            self.grid[pickup_position[0]][pickup_position[1]] = np.zeros(7+self.max_obj_per_type)
            self.grid_indices[pickup_position[0]][pickup_position[1]] = None
            # if empty, place the object on the grid
            self.grid[agent_pos_x-1][agent_pos_y] = self._get_object_vector_from_string('m-'+self.entity_list[object_index].get_object_id())
            self.grid_indices[agent_pos_x-1][agent_pos_y] = object_index
            # modify entity_list
            self.entity_list[object_index].object_type = 1
            self.entity_list[object_index].object_pos = (agent_pos_x-1, agent_pos_y)
            # remove the object from the agent
            self.unassign_object_from_agent((agent_pos_x, agent_pos_y))

    def get_agent_pos(self, agent_id):
        """
            get the agent's position
            inputs :
                agent_id : the id of the agent
            outputs :
                agent_pos_x : the x position of the agent
                agent_pos_y : the y position of the agent
        """
        for entity in self.entity_list:
            if entity.get_object_type() == 2 and entity.get_object_id() == agent_id:
                agent_pos_x = entity.get_object_pos_x()
                agent_pos_y = entity.get_object_pos_y()
        return agent_pos_x, agent_pos_y
    def move_agent(self, agent_id, action):
        """
            moves the agent in the environment
            inputs :
                agent_id : the id of the agent
                action : the action taken by the agent
            outputs :
                None
        """
        # get the agent's position
        agent_pos_x, agent_pos_y = self.get_agent_pos(agent_id)
        # if action is 0, move up
        if action == 0:
            # check if the cell above is empty
            if self.grid_indices[agent_pos_x-1][agent_pos_y] is None:
                # change self.grid, self.grid_indices and self.entity_list so as to move the agent one cell up
                self.grid[agent_pos_x-1][agent_pos_y] = self.grid[agent_pos_x][agent_pos_y]
                self.grid[agent_pos_x][agent_pos_y] = np.zeros(7+self.max_obj_per_type)
                self.grid_indices[agent_pos_x-1][agent_pos_y] = self.grid_indices[agent_pos_x][agent_pos_y]
                self.grid_indices[agent_pos_x][agent_pos_y] = None
                self.entity_list[self.grid_indices[agent_pos_x-1][agent_pos_y]].object_pos = (agent_pos_x-1, agent_pos_y)
        # if action is 1, move right
        elif action == 1:
            # check if the cell to the right is empty
            if self.grid_indices[agent_pos_x][agent_pos_y+1] is None:
                # change self.grid, self.grid_indices and self.entity_list so as to move the agent one cell right
                self.grid[agent_pos_x][agent_pos_y+1] = self.grid[agent_pos_x][agent_pos_y]
                self.grid[agent_pos_x][agent_pos_y] = np.zeros(7+self.max_obj_per_type)
                self.grid_indices[agent_pos_x][agent_pos_y+1] = self.grid_indices[agent_pos_x][agent_pos_y]
                self.grid_indices[agent_pos_x][agent_pos_y] = None
                self.entity_list[self.grid_indices[agent_pos_x][agent_pos_y+1]].object_pos = (agent_pos_x, agent_pos_y+1)
        # if action is 2, move down
        elif action == 2:
            # check if the cell below is empty
            if self.grid_indices[agent_pos_x+1][agent_pos_y] is None:
                # change self.grid, self.grid_indices and self.entity_list so as to move the agent one cell down
                self.grid[agent_pos_x+1][agent_pos_y] = self.grid[agent_pos_x][agent_pos_y]
                self.grid[agent_pos_x][agent_pos_y] = np.zeros(7+self.max_obj_per_type)
                self.grid_indices[agent_pos_x+1][agent_pos_y] = self.grid_indices[agent_pos_x][agent_pos_y]
                self.grid_indices[agent_pos_x][agent_pos_y] = None
                self.entity_list[self.grid_indices[agent_pos_x+1][agent_pos_y]].object_pos = (agent_pos_x+1, agent_pos_y)
        # if action is 3, move left
        elif action == 3:
            # check if the cell to the left is empty
            if self.grid_indices[agent_pos_x][agent_pos_y-1] is None:
                # change self.grid, self.grid_indices and self.entity_list so as to move the agent one cell left
                self.grid[agent_pos_x][agent_pos_y-1] = self.grid[agent_pos_x][agent_pos_y]
                self.grid[agent_pos_x][agent_pos_y] = np.zeros(7+self.max_obj_per_type)
                self.grid_indices[agent_pos_x][agent_pos_y-1] = self.grid_indices[agent_pos_x][agent_pos_y]
                self.grid_indices[agent_pos_x][agent_pos_y] = None
                self.entity_list[self.grid_indices[agent_pos_x][agent_pos_y-1]].object_pos = (agent_pos_x, agent_pos_y-1)
        # if action is 4, do nothing
        elif action == 4:
            pass
        # if action is 5, pick up an object
        elif action == 5:
            self.pick_up_object(agent_id)
        # if action is 6, drop an object
        elif action == 6:
            self.place_object(agent_id)
    
    def _random_init(self, difficulty):
        """
            initializes the environment randomly
            inputs :
                difficulty : the difficulty of the environment
            outputs :
                None
        """
        # generate a tree
        self.initialize_progression_tree(difficulty)
        # get entity_list
        self.entity_list = self._generate_entity_list_from_tree()
        # place entity_list in the grid
        self.place_entity_list_in_grid()

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
        self.assigned_object_id = None
    
    def get_object_type(self):
        return self.object_type
    def get_object_id(self):
        return self.object_id
    def get_object_pos_x(self):
        return self.object_pos[0]
    def get_object_pos_y(self):
        return self.object_pos[1]
    def assign_object(self, object_id):
        self.assigned_object_id = object_id
    def get_assigned_object_id(self):
        return self.assigned_object_id
    # string representation of the object
    def __str__(self):
        return self.entity_string_doc[self.object_type] + str(self.object_id)

    def __rep__(self):
        return self.entity_string_doc[self.object_type] + str(self.object_id)


"""
TODO:
    - change the task generation to start from a progression tree
    - make the procedural progression tree procedure
    - don't place all objects on the grid, only the ones that aren't the product of combinations
    - when placing objects, add condition where you can place on an existing object if the combination is feasible according to the tree
    - define the reward from progression tree transition and time_steps since start of task
    - take care of object combinations according to the rules of the progression tree
    - define the step function
    - take care of how to handle half-steps and which agent goes first
    - make an interface to try the environment which should include a render of the task and the progression tree
"""