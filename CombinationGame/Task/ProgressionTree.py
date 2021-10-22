import numpy as np

class ProgressionTree:
    def __init__(self, number_of_objects, tree_depth):
        self.number_of_objects = number_of_objects
        self.tree_depth = tree_depth
        # Each node has a corresponding feature vector of size 2, the first component being the id of the object, the second component being the object type.
        # Objects can be of 3 different types: Movable (0), Ground Landmark (1), Indicator (2).
        # A Combination of objects is represented by a branch in the tree.
        # The root of the tree is the goal state. 

        # We initialize a binary tree with empty node features.
        self.tree = np.zeros((2**(tree_depth+1)-1), 2)
    
    def _get_possible_object_combinations(self):
        """
            Returns a list of all possible combinations of objects.
            Each combination is a tuple where the first element is the first object and the second element is the second object and the third is the combined object.
            input: None
            output: list of tuples
        """
        # We start with the root node, which is the goal state.
        # We then go through the tree and get all possible combinations of objects.
        
        object_combinations = []
        # We start with the root node.
        root_node = self.tree[0]
        # We then get all possible combinations of objects.
        object_combinations.append((root_node[0], root_node[1], root_node[0]))
        for i in range(1, self.tree.shape[0]):
            # for each node in the tree, we get the two children
            left_child = self.tree[2*i-1]
            right_child = self.tree[2*i]
            # a combination is then a tuple of the two children and the current node
            object_combinations.append((left_child, right_child, self.tree[i]))
        
        # We then reverse the object_combinations list so that the first tuples are the leaf combinations.
        
        return object_combinations[::-1]

    def instantiate_tree(self, objects):
        """
            Instantiates the tree with the given objects.
            input: list of tuples
            output: None
        """
        # We start with the root node, which is the goal state.
        # We then go through the tree and instantiate it with the given objects.
        root_node = self.tree[0]
        root_node[0] = objects[0][0]
        root_node[1] = objects[0][1]
        for i in range(1, self.tree.shape[0]):
            # We instantiate each node with the corresponding object.
            self.tree[i][0] = objects[i][0]
            self.tree[i][1] = objects[i][1]
        
    def _get_node_feature_vector(self, node_id):
        """
            Returns the feature vector of the node with the given id.
            input: int
            output: tuple
        """
        return self.tree[node_id]
    
    def _get_node_id(self, feature_vector):
        """
            Returns the id of the node with the given feature vector.
            input: tuple
            output: int
        """
        return np.where(self.tree == feature_vector)[0][0]
    
    def _get_node_children(self, node_id):
        """
            Returns the ids of the children of the node with the given id.
            input: int
            output: tuple
        """
        return (2*node_id+1, 2*node_id+2)
    
    def _get_node_parent(self, node_id):
        """
            Returns the id of the parent of the node with the given id.
            input: int
            output: int
        """
        return (node_id-1)//2
    
    def _get_node_siblings(self, node_id):
        """
            Returns the ids of the siblings of the node with the given id.
            input: int
            output: tuple
        """
        parent = self._get_node_parent(node_id)
        left_child = 2*parent+1
        right_child = 2*parent+2
        return (left_child, right_child)
