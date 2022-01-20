""" Functions for dealing with ete3 for taxonomy representations
"""
from tree_structures import Node, Taxonomy
from typing import Union

# from ParGenN import Node
# from imp import reload
# import ParGen
# reload(ParGen)
# class Node(Collection):
#
#     def __init__(self, index, name, parent, children=None):
#         self.index=index
#         self.name=name
#         self.parent = parent
#         self.children = children
#         if self.children is None: self.children = []
#
#     def __contains__(self, item):
#         return item in self.children
#
#     def __iter__(self):
#         for item in self.children:
#             yield item
#
#     def __len__(self):
#         return len(self.children)
#
#     def __setattr__(self, name, value):
#         self.__dict__[name] = value
#
#     def __getattr__(self, name):
#         if name not in self.__dict__:
#             return None
#         return self.__dict__[name]
#
#     @property
#     def leaf_cluster(self):
#         leaves = []
#
#         def find_leaves(self):
#             if self.is_internal:
#                 for child in self.children:
#                     find_leaves(child)
#             else:
#                 leaves.append(self)
#         find_leaves(self)
#         return leaves
#
#     @property
#     def is_leaf(self):
#         return not self.children
#
#     @property
#     def is_internal(self):
#         return bool(self.children)
#
#     @property
#     def is_root(self):
#         return self.parent is None
#
# class Taxonomy:
#     # def __init__(self, taxonomy_df):
#     #     self.df = taxonomy_df
#     #     self._root = self.get_taxonomy_tree(taxonomy_df)
#     #     self.root = self._root
#     #     self._all_nodes = self.all_nodes(self.root)
#     #     self.leaves = self._root.leaf_cluster
#
#     def __init__(self, taxonomy):
#         if type(taxonomy) == pd.core.frame.DataFrame:
#             self.df = taxonomy_df
#             self._root = self.get_taxonomy_tree(taxonomy_df)
#         else:
#             self._root = taxonomy
#         self.root = self._root
#         self._all_nodes = self.all_nodes(self.root)
#         self.leaves = self._root.leaf_cluster
#
#     def get_taxonomy_tree(self, taxonomy_df):
#         nodes=[]
#         for i, entity in self.df.iterrows():
#             index = entity['level']
#             name = entity['label']
#             depth = entity['depth']
#             nodes.append([index, name, depth])
#             nodes = sorted(nodes, key=lambda x: x[2])
#
#
#         tree = Node(index='0', name='root', parent=None, children=None)         # creating root node
#
#         curr_parent = tree
#         self.processed = []
#         for node in nodes:
#             index, name, depth = node
#             if depth != 1:
#                 curr_parent = self.find_parent(index)
#             curr_node = Node(index, name, curr_parent)
#             curr_parent.children.append(curr_node)
#             self.processed.append(curr_node)
#         return tree
#
#     def find_parent(self, index):
#         for parent_c in self.processed:
#             if parent_c.index == '.'.join(index.split('.')[:-1]):
#                 return parent_c
#
#     def all_nodes(self, node):
#         nodes = []
#         for child in node:
#             nodes += self.all_nodes(child)
#         nodes.append(node)
#         return nodes
#
#
#     def copy(self):
#         C = Taxonomy(self.df)
#         for node1, node2 in zip(self.all_nodes_bfs, C.all_nodes_bfs):
#             for atr in node1.__dict__:
#                 if atr not in ['children', 'parent']:
#                     node2.__dict__[atr] = node1.__dict__[atr]
#         return C
#
#     @property
#     def all_nodes_bfs(self):
#         result=[]
#         queue = [self.root]
#         while queue:
#             node=queue.pop(0)
#             result.append(node)
#             queue+=[child for child in node]
#         return result

def make_ete3_lifted(taxonomy_tree: Union[Node, Taxonomy], print_all: bool = True) -> str:
    """Returns ete3 representation of a taxonomy tree
       after lifting procedure completed

    Parameters
    ----------
    taxonomy_tree : Union[Node, tree_structures]
        the root of the taxonomy tree / sub-tree or taxonomy
    print_all : bool, default=True
        label for printing all the parameters

    Returns
    -------
    str
        resulting ete3 representation
    """
    if isinstance(taxonomy_tree, Taxonomy):
        taxonomy_tree = taxonomy_tree.root

    head_subjects = set(t.index for t in taxonomy_tree.H)
    gaps = set(t.index for t in taxonomy_tree.L)
    print('Gaps idx: ', gaps)
    hs = set(t.index for t in taxonomy_tree.H if not t.is_leaf)
 

    def rec_ete3(node, head_subject=0):
        output = []

        if node.index in head_subjects and not head_subject:
            head_subject = 1

        if node.is_internal:
            output.append("(")
            sorted_children = sorted(node.children, key=lambda x: x.u)
            j = 0
            while not sorted_children[j].u:
                j += 1

            last_sorted_name = sorted_children[j - 1].name
            if j == 2:
                sorted_children[j - 1].name = sorted_children[0].name + ". " \
                                                 + sorted_children[j - 1].name
            if j > 2:
                sorted_children[j - 1].name = sorted_children[0].name + "..." \
                                                 + sorted_children[j - 1].name + \
                                                 " " +  str(j) + " items"
            if j:
                output.extend(rec_ete3(sorted_children[j - 1], head_subject=head_subject))
                output.append(",")

            sorted_children[j - 1].name = last_sorted_name

            children_len = len(sorted_children[j:])
            for k, child in enumerate(sorted_children[j:]):
                output.extend(rec_ete3(child, head_subject=head_subject))
                if k < children_len - 1:
                    output.append(",")
            output.append(")")

        if node.u > 0 or print_all:
            output.append(node.name)
            output.extend(["[&&NHX:", "p=", str(round(node.p, 3)) if node.p else '0', ":", "e=", str(node.e), \
                           ":", "H={", ";".join([s.name for s in ((node.H or []) if \
                                                                  len(node.H or []) < 3 \
                                                                  else [node.H[0], \
                                                                        Node(None, "...", None), \
                                                                        node.H[-1]])]), \
                           "}:u=", str(round(node.u, 3)), ":", "v=", str(round(node.v, 3)), \
                           ":G={", ";".join([s.name for s in ((node.G or []) \
                                                              if len(node.G or []) < 3 \
                                                              else [node.G[0],
                                                                    Node(None, "...", None), \
                                                                    node.G[-1]])]), \
                           "}:L={", ";".join([s.name for s in ((node.L or []) \
                                                               if len(node.L or []) < 3 \
                                                               else [node.L[0], \
                                                                     Node(None, "...", None), \
                                                                     node.L[-1]])]), \
                           # "}:Hd=", ("1" if node.index in head_subjects else "0"), ":Ch=", \
                           "}:Hd=", ("1" if node.index in hs else "0"), ":Ch=", \
                           ("1" if node.is_internal else "0"), ":Sq=", ("1" if head_subject \
                                                                        else "0"),\
                           ":Gap=", ("1" if node.index in gaps else "0"),\
                           ":index=", node.index, \
                           "]"])

        return output

    output = rec_ete3(taxonomy_tree)
    output.append(";")
    return "".join(output)


def make_ete3_raw(taxonomy_tree: Union[Node, Taxonomy]) -> str:
    """Returns ete3 representation of a taxonomy tree
       for raw taxonomy

    Parameters
    ----------
    taxonomy_tree : Union[Node, tree_structures]
        the root of the taxonomy tree / sub-tree or taxonomy

    Returns
    -------
    str
        resulting ete3 representation
    """
    if isinstance(taxonomy_tree, Taxonomy):
        taxonomy_tree = taxonomy_tree.root

    def rec_ete3(node):
        output = []

        if node.is_internal:
            output.append("(")

            children_len = len(node.children)

            for k, child in enumerate(node.children):
                output.extend(rec_ete3(child))
                if k < children_len - 1:
                    output.append(",")

            output.append(")")
        output.append(node.name)

        return output

    output = rec_ete3(taxonomy_tree)
    output.append(";")
    return "".join(output)


def save_ete3(ete3_desc: str, filename: str = "taxonomy_tree_lifted.ete") -> None:
    """Writes resulting ete3 in a file

    Parameters
    ----------
    ete3_desc : str
        ete3 representation in a string
    filename : str, default="taxonomy_tree_lifted.ete"
        name of the file for writing

    Returns
    -------
    None
    """

    with open(filename, 'w') as file_opened:
        file_opened.write(ete3_desc)

    print(f"ete representation saved in the file: {filename}")

