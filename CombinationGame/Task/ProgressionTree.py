import numpy as np


# a node in the tree
class Node:
    def __init__(self, value, depth, parent=None, children=None, ):
        self.value = value
        self.parent = parent
        self.children = children
        self.depth = depth
        self.id = None
    def __str__(self):
        return str(self.value)
    
    def __repr__(self):
        return str(self.value)
    
    def __eq__(self, other):
        
        if isinstance(other, Node):
            return self.value == other.value
        else:
            return False
    
    def __ne__(self, other):

        if isinstance(other, Node):
            return self.value != other.value
        else:
            return True

# a tree
class Tree:
    def __init__(self, root=None):
        self.root = root
    
    def __str__(self):
        return str(self.root)
    
    def __repr__(self):
        return str(self.root)
    
    def __eq__(self, other):
        
        if isinstance(other, Tree):
            return self.root == other.root
        else:
            return False
    
    def __ne__(self, other):

        if isinstance(other, Tree):
            return self.root != other.root
        else:
            return True
        
    def add_node(self, node, parent=None):
        
        if parent is None:
            self.root = node
        else:
            parent.children.append(node)
            node.parent = parent
    
    def get_leaves(self):
        leaves = []
        def get_leaves_rec(node):
            if node.children is None:
                leaves.append(node)
            else:
                for child in node.children:
                    get_leaves_rec(child)
        get_leaves_rec(self.root)
        return leaves
    
    def get_nodes(self):
        nodes = []
        def get_nodes_rec(node):
            nodes.append(node)
            if node.children is not None:
                for child in node.children:
                    get_nodes_rec(child)
        get_nodes_rec(self.root)
        return nodes
    
    def get_max_depth(self):
        def get_max_depth_rec(node):
            if node.children is None:
                return 1
            else:
                depths = []
                for child in node.children:
                    depths.append(get_max_depth_rec(child))
                return max(depths) + 1
        return get_max_depth_rec(self.root)
    
    def get_min_depth(self):
        def get_min_depth_rec(node):
            if node.children is None:
                return 1
            else:
                depths = []
                for child in node.children:
                    depths.append(get_min_depth_rec(child))
                return min(depths) + 1
        return get_min_depth_rec(self.root)
    
    def get_nodes_by_depth(self, depth):
        """Returns the list of nodes that have depth 'depth'"""
        def get_nodes_by_depth_rec(node, depth):
            if node.depth == depth:
                return [node]
            else:
                nodes = []
                if node.children is not None:
                    for child in node.children:
                        nodes.extend(get_nodes_by_depth_rec(child, depth))
                return nodes
        return get_nodes_by_depth_rec(self.root, depth)

# overload the tree class to define a progression tree 
class ProgressionTree(Tree):
    def __init__(self, root=None):
        super().__init__(root)
        self.number_of_nodes = 0
    def generate_tree(self, branching_probabilities):
        """
            Generates a random tree. Starts at the root and then for each depth branches with the corresponding probability. 
            For every branching node, generate 2 children node with unique values.
            Inputs: 
                branching_probabilities[k] is the probability that a node at depth k branches
            Outputs:
                tree: a tree object
        """
        # create the root node
        root = Node(0, 0)
        self.add_node(root)
        # create the tree
        def generate_tree_rec(node, branching_probabilities):
            if node.depth < len(branching_probabilities):
                if branching_probabilities[node.depth] > np.random.rand():
                    # generate the children nodes
                    children = []
                    for i in range(1, 3):
                        child = Node(None, depth=node.depth + 1, parent=node)
                        children.append(child)
                    # set the children nodes
                    node.children = children
                    # generate the children nodes
                    for child in node.children:
                        generate_tree_rec(child, branching_probabilities)
                else:
                    return
        generate_tree_rec(root, branching_probabilities)
        # set the number of nodes
        self.set_number_of_nodes()
        return self
    def set_number_of_nodes(self):
        # Computes the number of nodes in the tree and sets the number_f_nodes attribute
        self.number_of_nodes = 0
        def set_number_of_nodes_rec(node):
            self.number_of_nodes += 1
            if node.children is not None:
                for child in node.children:
                    set_number_of_nodes_rec(child)
        set_number_of_nodes_rec(self.root)

    def set_node_value(self, node_id, value):
        """
            Sets the value of a node.
            Inputs:
                node_id: the node id corresponding to the number of the node when traversing three from left to right
                value: the value to set
            Outputs:
                None
        """
        nodes = self.get_nodes()
        node = nodes[node_id]
        node.value = value
    
    def get_node_value(self, node_id):
        """
            Returns the value of a node.
            Inputs:
                node_id: the node id corresponding to the number of the node when traversing three from left to right
            Outputs:
                value: the value of the node
        """
        nodes = self.get_nodes()
        node = nodes[node_id]
        return node.value
    
    def set_node_values(self, values):
        """
            Sets the values of all nodes.
            Inputs:
                values: a list of values to set
            Outputs:
                None
        """
        nodes = self.get_nodes()
        for i in range(len(nodes)):
            nodes[i].value = values[i]
    def set_node_ids(self):
        """
            Sets the id of nodes following their id in from get_nodes().
            Inputs:
                values: a list of values to set
            Outputs:
                None
        """
        nodes = self.get_nodes()
        for i in range(len(nodes)):
            nodes[i].id = i
    def _is_node_leaf(self, node):
        return node.children is None
    def _leaf_node_mask(self):
        """
            Returns a mask of the leaf nodes over the node list.
        """
        nodes = self.get_nodes()
        leaf_nodes_mask = np.zeros(len(nodes), dtype=bool)
        for i in range(len(nodes)):
            if self._is_node_leaf(nodes[i]):
                leaf_nodes_mask[i] = True
            else:  
                leaf_nodes_mask[i] = False
        return leaf_nodes_mask
    def children_is_leaf_node_mask(self):
        """
            Returns a mask of the leaf nodes over the node list.
        """
        nodes = self.get_nodes()
        leaf_nodes_mask = np.zeros(len(nodes), dtype=bool)
        for i in range(len(nodes)):
            for child in nodes[i].children():
                if self._is_node_leaf(child):
                    leaf_nodes_mask[i] = True
                    break
        return leaf_nodes_mask
    
    def _get_node_id_by_depth(self):
        """
        returns a list of node ids (as accessible in the get_nodes() list) for each depth 
        """
        nodes = self.get_nodes()
        node_ids_by_depth = []
        for depth in range(self.get_max_depth()):
            node_ids_by_depth.append([])
            for node in nodes:
                if node.depth == depth:
                    node_ids_by_depth[depth].append(node.id)
        return node_ids_by_depth

    def set_node_ids_by_depth(self):
        nodes = self.get_nodes()
        count = 0
        for depth in range(self.get_max_depth()):
            for node in nodes:
                if node.depth == depth:
                    node.id = count
                    count += 1
    def render_tree(self, representation='values'):
        """
            Renders the tree in a human readable format.
            Inputs:
                tree: a tree object
                representation: the representation of the nodes to render. Can be 'values' or a function that takes a node as input and returns a string
            Outputs:
                tree_str: a string representing the tree
        """
        
        def str_tree_from_node(node, level=0, stt=""):
            if representation == 'values':
                stt += '\t' * level + repr(node.value) + '\n'
            else:
                stt += '\t' * level + representation(node) + '\n'

            if node.children is not None:
                for child in node.children:
                    stt = str_tree_from_node(child, level+1, stt)
                return stt
            else:
                return stt
        return str_tree_from_node(self.root)
if __name__ == "__main__":
    # create a tree
    tree = ProgressionTree()
    # generate a tree
    tree = tree.generate_tree([1, 1, 1, 0.5])
    # render the tree
    tree.set_node_ids_by_depth()
    print(tree.render_tree(representation=lambda node: str(node.id)))