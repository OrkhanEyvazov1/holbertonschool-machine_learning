#!/usr/bin/env python3
"""isolation random tree implementation for anomaly detection"""
import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree():
    """isolation random tree class for anomaly"""
    def __init__(self, max_depth=10, seed=0, root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """
        Returns a string representation of the decision tree.

        Returns:
        str
            The string representation of the decision tree.
        """
        return self.root.__str__() + "\n"

    def depth(self):
        """
        Returns the maximum depth of the tree.

        Returns:
        int
            The maximum depth of the tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Counts the number of nodes in the decision tree.

        Parameters:
        only_leaves : bool, optional
            If True, count only the leaf nodes (default is False).

        Returns:
        int
            The number of nodes in the tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """update the bounds"""
        pass

    def get_leaves(self):
        """get the leaves"""
        pass

    def update_predict(self):
        """update"""
        pass  # <--- same as in Decision_Tree (but not implemented there yet)

    def np_extrema(self, arr):
        """np   dd"""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """random split criterion"""
        pass  # <--- same as in Decision_Tree (but not implemented there yet)

    def get_leaf_child(self, node, sub_population):
        """get a leaf child"""
        leaf_child = Leaf(value=None)  # Value logic to be implemented
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """get a node child"""
        pass  # <--- same as in Decision_Tree

    def fit_node(self, node):
        """fit node"""
        node.feature, node.threshold = self.random_split_criterion(node)

        left_population = None  # <--- to be filled
        right_population = None  # <--- to be filled

        is_left_leaf = (node.depth + 1 >= self.max_depth) or \
                       (np.sum(left_population) < self.min_pop)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = (node.depth + 1 >= self.max_depth) or \
                        (np.sum(right_population) < self.min_pop)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """fitting the isolation random tree to the data"""
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones(
            explanatory.shape[0], dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")
