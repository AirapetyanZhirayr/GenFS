from libs import *

class Node:
    """
    class for representing taxonomy nodes with their attributes
    """
    def __init__(self, index, name, parent, children=None):
        self.index = index
        self.name = name
        self.children = children
        if self.children is None:
            self.children = []
        self.parent = parent
        if self.parent is not None:
            # adding node to parent's children list
            self.parent.children.append(self)


    def __contains__(self, item):
        return item in self.children

    def __iter__(self):
        for item in self.children:
            yield item

    def __len__(self):
        return len(self.children)

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __getattr__(self, name):
        if name not in self.__dict__:
            return None
        return self.__dict__[name]

    @property
    def is_leaf(self):
        return not self.children

    @property
    def is_internal(self):
        return not self.is_leaf

    @property
    def is_root(self):
        return self.parent is None

    @property
    def leaf_cluster(self):
        """
        :return: list of leaves covered by node
        BFS search
        """
        queue = []  # queue of nodes to be checked
        leaves = []  # leaves from queue
        queue += self.children
        while queue:
            node = queue.pop(0)
            if node.is_leaf:
                leaves.append(node)
            else:
                queue += node.children
        return leaves

    @property
    def descendants(self):
        """
        :return: list of nodes of subtree with root in self
        Does not include root itself, only descendants
        """
        queue = []
        nodes = []
        queue += self.children
        while queue:
            node = queue.pop(0)
            nodes.append(node)
            queue += node.children
        return nodes


class Taxonomy:
    """
    class for Taxonomy tree
    2 ways of initialization:
     - by taxonomy_df with nodes in dfs order
     - by root - entity of Node class with already encoded tree structure
    """

    def __init__(self, initializer):
        if type(initializer) == pd.core.frame.DataFrame:
            self.df = initializer
            self.root = self.get_tree_df()
        elif type(initializer) == Node:
            self.root = initializer
        else:
            print('Wrong input')
            raise ValueError

    def get_tree_df(self):
        """

        :return: root node with built up tree
        Assumes that nodes in the df are arranged in DFS order
        """
        root  = Node(index='0', name='root',
                     parent=None, children=None)
        nodes = [root]
        prev_depth = 0
        parent = root

        for i,  node_data in self.df.iterrows():
            index, name, depth = node_data[['level', 'label', 'depth']]
            if depth > prev_depth:
                if depth - prev_depth > 1:
                    '''
                    Not ok situation for DFS nodes.
                    '''
                    print('Nodes are not arranged in DFS order!')
                    raise ValueError
                else:
                    parent = nodes[-1]
            elif depth < prev_depth:
                '''
                This may happen if previous node was leaf and 
                current node is much higher in the tree
                '''
                # going up in tree
                for _ in range(prev_depth - depth):
                    parent = parent.parent
            elif depth == prev_depth:
                '''
                This elif brunch is created for reader convenience.
                It points out that if depth has not changed, 
                then parent has not changed too. 
                '''
                pass

            node = Node(index, name, parent, None)
            nodes.append(node)
            prev_depth = depth

        return root

    def copy(self):
        copy_taxonomy = Taxonomy(self.df)
        for copy_node, node in zip(copy_taxonomy.nodes, self.nodes):
            for atr in node.__dict__:
                if atr not in ['children', 'parent']:
                    copy_node.__dict__[atr] = node.__dict__[atr]
        return copy_taxonomy

    @property
    def leaves(self):
        return self.root.leaf_cluster

    @property
    def nodes(self):
        """
        :return: list of taxonomy nodes in BFS order, including root
        """
        queue = [self.root]
        nodes = []
        while queue:
            node = queue.pop(0)
            nodes.append(node)
            queue += node.children
        return nodes