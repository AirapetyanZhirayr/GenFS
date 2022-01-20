from libs import *
from tree_structures import Node, Taxonomy
from ete3_functions import make_ete3_lifted, save_ete3
from visualize import *
from numpy import log, sqrt

TRUNCATING_THRESHOLD = 0.001
EPS = 1e-15


class ParGen:

    def __init__(self, taxonomy,
                 truncating_threshold=TRUNCATING_THRESHOLD):
        """

        :param taxonomy: global taxonomy for keeping statistics
        of node scenarios
        :param truncating_threshold: ADD DESCRIPTION
        """
        self.taxonomy = taxonomy
        self.threshold = TRUNCATING_THRESHOLD
        # self.root: global root
        self.root = self.taxonomy.root
        self.initialize_scenarios(self.root)
        self.n_clusters = 0

    def initialize_scenarios(self, node):
        """
        node.n_G: counter for scenarios not_inherited but Gained
        node.n_not_G: counter for scenarios not_inherited nor Gained
        node.i_L: counter for scenarios inherited but Lost
        node.i_not_L : counter for scenarios inherited not Lost
        """
        for node in self.taxonomy.nodes:
            node.n_G= 0
            node.n_not_G = 0
            node.i_L = 0
            node.i_not_L = 0

            node.n_gains = 0
            node.n_losses = 0

    def fit(self, cluster, curr_taxonomy, cl_num,
            gamma, _lambda, save_results,
            verbose=False):
        self.gamma = gamma; self._lambda  = _lambda
        self.curr_taxonomy = curr_taxonomy
        self.curr_root = curr_taxonomy.root

        if verbose:
            print('Enumerating layers of current taxonomy...')
        self.enumerate_tree_layers(self.curr_root)

        _sum = self.annotate_leaf_memberships(cluster)
        if verbose:
            print(f'Overall sum of squares of leaf memberships: {_sum}')
        leaf_memberships = self.normalize_leaf_memberships(_sum)
        if verbose:
            print(f"Number of leaves: {len(leaf_memberships)}")
            print("All positive weights:")
            for node_name, node_u in sorted(leaf_memberships, key=lambda x: x[1], reverse=True):
                if not node_u:
                    break
                print(f"{node_name:<60} {node_u:.2f}")
        sum_after_trunc = self.truncate_memberships(self.threshold)
        if sum_after_trunc < .1:
            print('Truncating threshold is too high.')
            raise RuntimeError
        updated_leaf_memberships = self.normalize_leaf_memberships(sum_after_trunc)
        if verbose:
            print("After transformation:")
            for node_name, node_u in sorted(updated_leaf_memberships, key=lambda x: x[1], reverse=True):
                if not node_u:
                    break
                print(f"{node_name:<60} {node_u:.3f}")
        root_membership = self.set_internal_memberships(self.curr_root)
        if abs(root_membership - 1.) > 0.01:
            print(f'Something wrong with propagation of memberships, root membership is {root_membership}.')
            print(f'It should be equal to 1 after quadratic propagation')
            raise RuntimeError
        self.prune_tree()
        # self.set_gaps()
        self.set_params(self.curr_root)
        self.reduce_edges(self.curr_root)

        self.make_init_step(self.curr_root, self.gamma)
        self.make_recursive_step(self.curr_root, self.gamma, self._lambda)

        self.indicate_offshoots()

        if save_results:
            result_table = self.make_result_table(self.curr_root)
            ete3_desc = make_ete3_lifted(self.curr_root)
            if type(cl_num) == int:
                # save_ete3(ete3_desc, filename=f'lifts/taxonomy_tree_lifted_PG.{cl_num}.ete')
                save_ete3(ete3_desc, filename=f'/Users/jiji/Desktop/GenFS/CODE/experiments/TFIDF/keywords/lifts/taxonomy_tree_lifted_PG.{cl_num}.ete')
                self.save_result_table(result_table,
                f'/Users/jiji/Desktop/GenFS/CODE/experiments/TFIDF/keywords/lift_tables/        table_PG.{cl_num}.csv')

                # self.save_result_table(result_table, f'tables/table_PG.{cl_num}.csv')
            else:
                save_ete3(ete3_desc, filename='lifts/taxonomy_tree_lifted_PG.ete')
                self.save_result_table(result_table, f'tables/table_PG.csv')

        self.count_scenarios()
        self.count_events()
        self.n_clusters += 1

    def fit_ml(self, cluster, cl_num=None, verbose=False):
        self.curr_taxonomy = self.taxonomy.copy()
        self.curr_root = self.curr_taxonomy.root
        self.enumerate_tree_layers(self.curr_root)
        if verbose:
            print('Enumerating layers of current taxonomy...')
        self.enumerate_tree_layers(self.curr_root)

        _sum = self.annotate_leaf_memberships(cluster)
        if verbose:
            print(f'Overall sum of squares of leaf memberships: {_sum}')

        leaf_memberships = self.normalize_leaf_memberships(_sum)
        if verbose:
            print(f"Number of leaves: {len(leaf_memberships)}")
            print("All positive weights:")
            for node_name, node_u in sorted(leaf_memberships, key=lambda x: x[1], reverse=True):
                if not node_u:
                    break
                print(f"{node_name:<60} {node_u:.2f}")
        sum_after_trunc = self.truncate_memberships(self.threshold)
        if sum_after_trunc < .1:
            print('Truncating threshold is too high.')
            raise RuntimeError
        updated_leaf_memberships = self.normalize_leaf_memberships(sum_after_trunc)
        if verbose:
            print("After transformation:")
            for node_name, node_u in sorted(updated_leaf_memberships, key=lambda x: x[1], reverse=True):
                if not node_u:
                    break
                print(f"{node_name:<60} {node_u:.3f}")
        root_membership = self.set_internal_memberships(self.curr_root)
        if abs(root_membership - 1.) > 0.01:
            print(f'Something wrong with propagation of memberships, root membership is {root_membership}.')
            print(f'It should be equal to 1 after quadratic propagation')
            raise RuntimeError
        self.prune_tree()
        # self.set_gaps()
        self.set_params(self.curr_root)
        self.reduce_edges(self.curr_root)
        ReversedLevelOrderTraversal = reversed(self.curr_taxonomy.nodes)
        for node in ReversedLevelOrderTraversal:
            if node.is_leaf:
                if node.u > 0:
                    node.i = log(1 - node.p_loss + EPS)
                    node.n = log(node.p_gain + EPS)
                    node.ev_i = [(node, 'Pass')]
                    node.ev_n = [(node, 'Gain')]
                else:
                    node.i = log(node.p_loss + EPS)
                    node.n = log(1 - node.p_gain + EPS)
                    node.ev_i = [(node, 'Loss')]
                    node.ev_n = [(node, 'Pass')]
            else:

                sum_i = sum(c.i for c in node.children)
                sum_n = sum(c.n for c in node.children)

                # scenarios for inheritance
                sc_lost = log(node.p_loss + EPS) + sum_n
                sc_not_lost = log(1 - node.p_loss + EPS) + sum_i

                if sc_lost > sc_not_lost:
                    node.i = sc_lost
                    node.ev_i = sum((c.ev_n for c in node), []) + [(node, 'Loss')]
                else:
                    node.i = sc_not_lost
                    node.ev_i = sum((c.ev_i for c in node), []) + [(node, 'Pass')]

                sc_gain = log(node.p_gain + EPS) + sum_i
                sc_no_gain = log(1 - node.p_gain + EPS) + sum_n

                if sc_gain > sc_no_gain:
                    node.n = sc_gain
                    node.ev_n = sum((c.ev_i for c in node), []) + [(node, 'Gain')]
                else:
                    node.n = sc_no_gain
                    node.ev_n = sum((c.ev_n for c in node), []) + [(node, 'Pass')]
        scenario = node.ev_n
        self.curr_root.H = []
        self.curr_root.L = []
        for node, event in scenario:
            if event == 'Gain':
                self.curr_root.H.append(node)
            elif node.is_leaf and event == 'Loss':
                self.curr_root.L.append(node)

        ete3_desc = make_ete3_lifted(self.curr_root)
        if type(cl_num) == int:
            save_ete3(ete3_desc, filename= f"lifts/taxonomy_tree_lifted_ML.{cl_num}.ete")
        else:
            save_ete3(ete3_desc, filename="lifts/taxonomy_tree_lifted_ML.ete")

    def enumerate_tree_layers(self, node, curr_layer=0):
        node.e = curr_layer
        for child in node:
            self.enumerate_tree_layers(child, curr_layer=curr_layer+1)

    def annotate_leaf_memberships(self, cluster):
        """
        Annotates leaf nodes of current taxonomy with their memberships (node.u)
        :param cluster: dict of type leaf.name: leaf.membership
        :return: _sum: sum of squares of leaf memberships
        """
        _sum = 0
        for leaf in self.curr_taxonomy.leaves:
            membership = cluster.get(leaf.name, 0)
            leaf.u = membership
            # leaf.score = membership
            _sum += membership**2
        return _sum

    def normalize_leaf_memberships(self, _sum):
        """
        Normalizes memberships of leaves in accordance
        with quadratic normalization
        :param _sum: sum of squares of leaf memberships
        :return: list of tuples of type [(leaf.name, leaf.membership)]
        """
        leaf_memberships = []
        for leaf in self.curr_taxonomy.leaves:
            leaf.u = leaf.u/np.sqrt(_sum)
            leaf_memberships.append((leaf.name, leaf.u))
        return leaf_memberships

    def truncate_memberships(self, threshold):
        """
        :param threshold: if memberships is less then threshold, then it is reset to 0
        :return: _sum: overall sum of squares of leaf memberships
        """
        _sum = .0
        for leaf in self.curr_taxonomy.leaves:
            if leaf.u < threshold:
                leaf.u = .0
            _sum += leaf.u**2
        return _sum

    def set_internal_memberships(self, node):
        """
        Propagates membership values from leaf nodes to internal nodes.
        :param node:
        :return: _sum: membership in root.
        Membership in root should be equal to 1. This return is used to check correctness
        of membership propagation
        """
        if node.is_leaf:
            return node.u**2
        _sum = .0
        for child in node:
            _sum += self.set_internal_memberships(child)
        node.u = np.sqrt(_sum)
        return _sum

    def prune_tree(self):
        """
        Deleting all non-maximal u-irrelevant nodes
        """
        queue = [self.curr_root]
        while queue:
            node = queue.pop(0)
            if node.is_internal and node.u == 0.0:
                node.children = []
                # node.G = [node]
            else:
                queue += node.children
    # def set_gaps(self):
    #     for node in self.curr_taxonomy.nodes:
    #         gaps = [child for child in node if child.u == 0.0]
    #         if not node.G:
    #             node.G = gaps
    def set_params(self, node):
        """
        Setting Gaps -- node.G, gap importance values -- node.v
        and the summary gap importance values -- node.V for all nodes
        Note: Actually we do not need gap importance values -- node.v at
        interior nodes, but we set it equal to node.parent membership for convenience.
        (It is needed for drawing tree)
        """
        node.G = []
        node.v = node.parent.u if node.parent else 1
        node.V = 0

        if node.is_leaf and not node.u:
            return [node], node.v

        for child in node:
            g, V =self.set_params(child)
            node.G+=g; node.V+=V
        return node.G, node.V
    # def set_params(self):
    #     """
    #     nodes are processed in reversed order. (leaves, then leaves parents, .. etc... root)
    #
    #     NOTE: try to change from lists to sets (If you do soo, do not forget to change lists to sets
    #     in prune_tree [node.G] --> {node.G})
    #     """
    #     for node in reversed(self.curr_taxonomy.nodes):
    #         gaps = node.G
    #         for child in node:
    #             gaps+=child.G
    #         gaps = list(set(gaps))
    #         # gaps
    #         # assert  len(set(gaps)) == len(gaps)
    #         node.G = gaps
    #         node.v = node.parent.u if node.parent else 1.
    #         node.V = sum(gap.v for gap in node.G)
    def reduce_edges(self, node):
        """
        Delete edges from node to child if there is only one child
        """
        if len(node) == 1:
            temp = node.children[0].children
            node.children = temp
            for _child in node:
                _child.parent = node

            def update_layer_number(t_node):
                t_node.e -= 1
                for child in t_node:
                    update_layer_number(child)

            for child in node:
                update_layer_number(child)

        for child in node:
            self.reduce_edges(child)

    def make_init_step(self, node, gamma_v):
        """
        Init step of ParGenFS
        """
        if node.is_internal:
            for child in node:
                self.make_init_step(child, gamma_v)
        else:
            if node.u > 0:
                node.H = [node]
                node.L = []
                node.p = gamma_v * node.u
            else:
                node.H = []
                node.L = []
                node.p = 0
            # node.o = True

    def make_recursive_step(self, node, gamma_v, lambda_v):
        """
        Recursive step of ParGenFS
        """
        if node.is_internal:
            for child in node:
                self.make_recursive_step(child, gamma_v, lambda_v)

            sum_penalty = sum([child.p for child in node])

            if node.u + lambda_v * node.V < sum_penalty:
                node.H = [node]
                node.L = node.G
                node.p = node.u + lambda_v * node.V
            else:
                node.H = sum((child.H for child in node), [])
                node.L = sum((child.L for child in node), [])
                node.p = sum_penalty

    def indicate_offshoots(self):
        for node in self.curr_root.H:
            if node.is_leaf:
                node.of = 1

    def make_result_table(self, node):

        table = []

        if node.is_internal:
            for child in node:
                table.extend(self.make_result_table(child))

        table.append([node.index.rstrip(".") or "", node.name, str(round(node.u, 3)),
                      str(round(node.p, 3)), str(round(node.V, 3)),
                      "; ".join([" ".join([s.index, s.name]) for s in (node.G or [])]),
                      "; ".join([" ".join([s.index, s.name]) for s in (node.H or [])]),
                      "; ".join([" ".join([s.index, s.name]) for s in (node.L or [])])])

        return table

    def save_result_table(self, result_table, filename):

        result_table = sorted(result_table, key=lambda x: (len(x), x))
        result_table = [["index", "name", "u", "p", "V", "G", "H", "L"]] + result_table

        with open(filename, 'w') as file_opened:
            for table_row in result_table:
                file_opened.write('\t'.join(table_row) + '\n')

        # print(f"Table saved in the file: {filename}")

    def count_scenarios(self):
        """
        Counting node scenarios for new vers. of MaLGen
        Note: Nodes are added to Absent list just for completeness.
        Actually we do not use the added nodes.
        """
        queue = [self.curr_root]
        Present = self.curr_root.H.copy()
        Absent = self.curr_root.L.copy()
        while queue:
            node = queue.pop(0)
            global_node = self.get_global_node(node.index)
            if node in Present:
                global_node.n_G += 1
            elif node in Absent:
                global_node.i_L += 1
            else:
                if node.parent and node.parent in Present:
                    global_node.i_not_L += 1
                    Present.append(node)
                else:
                    global_node.n_not_G += 1
                    Absent.append(node)
            queue += node.children
    #
    # def count_scenarios(self):
    #     """
    #     Counting scenarios for further calculation of probabilities for Max. Likelihood approach
    #     """
    #     for loc_node in self.curr_root.L:
    #         # all gaps are from inherited but Lost (i_L)
    #         glob_node = self.get_global_node(loc_node.index)
    #         glob_node.i_L += 1
    #         loc_node.labeled = True
    #
    #     for loc_node in self.curr_root.H:
    #         # all h_s or offshoots are not_inherited but Gained (n_G)
    #         glob_node = self.get_global_node(loc_node.index)
    #         glob_node.n_G += 1
    #         loc_node.labeled = True
    #
    #         if loc_node.is_internal:
    #             # all descendants of h_s (except gaps) are inherited not Lost (i_not_L)
    #             for loc_node_desc in loc_node.descendants:
    #                     if not loc_node_desc.labeled:
    #                         glob_node = self.get_global_node(loc_node_desc.index)
    #                         glob_node.i_not_L += 1
    #                         loc_node_desc.labeled = True
    #
    #     for loc_node in self.curr_taxonomy.nodes:
    #         # all other nodes are from nor inherited nor gained
    #         if not loc_node.labeled:
    #             glob_node = self.get_global_node(loc_node.index)
    #             glob_node.n_not_G += 1
    #         loc_node.labeled = True
    def count_events(self):
        """
        Counting loss and gain events for old vers. of MaLGen
        """
        for node in self.curr_root.H:
            global_node = self.get_global_node(node.index)
            global_node.n_gains += 1

        for node in self.curr_root.L:
            global_node = self.get_global_node(node.index)
            global_node.n_losses += 1

    def get_global_node(self, index):
        """
        Auxillary func to map nodes from local taxonomy to nodes from global taxonomy
        :param index: index of node to be found un global taxonomy
        :return: node from global taxonomy with index [index]
        """
        for node in self.taxonomy.nodes:
            if node.index == index:
                return node

    def estimate_probs_scenarios(self):
        """
        Estimating probabilities of loss and gain for new version of MaLGen
        """
        for node in self.taxonomy.nodes:
            if node.n_G > 0:
                node.p_gain = node.n_G/(node.n_G + node.n_not_G)
            else:
                node.p_gain = 0
            if node.i_L > 0:
                node.p_loss = node.i_L/(node.i_L + node.i_not_L)
            else:
                node.p_loss = 0

    def estimate_probs_events(self):
        """
        Estimating probabilities of gain and loss for old version of MaLGen
        """
        for node in self.taxonomy.nodes:
            node.p_gain = node.n_gains / self.n_clusters
            node.p_loss = node.n_losses / self.n_clusters


if __name__ == '__main__':
    taxonomy_df = pd.read_csv('data/taxonomy_df.csv')
    u = load_obj('clusters_full')
    topics_unique = load_obj('topics_unique')
    cl_idx = 0
    cl_dict = dict(zip(topics_unique, u[:,cl_idx]))

    taxonomy1 = Taxonomy(taxonomy_df)
    taxonomy2 = Taxonomy(taxonomy_df)

    pargen1 = ParGen(taxonomy1)
    pargen2 = ParGen(taxonomy2)
    pargen1.curr_taxonomy = taxonomy1
    pargen2.curr_taxonomy = taxonomy2
    pargen1.curr_root = pargen1.curr_taxonomy.root
    pargen2.curr_root = pargen2.curr_taxonomy.root

    for pargen in pargen1, pargen2:
        pargen.gamma = 0.8; pargen._lambda = 0.1
        pargen.enumerate_tree_layers(pargen.curr_root)
        _sum = pargen.annotate_leaf_memberships(cl_dict)
        leaf_memberships = pargen.normalize_leaf_memberships(_sum)
        sum_after_trunc = pargen.truncate_memberships(0.001)
        updated_leaf_memberships = pargen.normalize_leaf_memberships(sum_after_trunc)
        root_membership = pargen.set_internal_memberships(pargen.taxonomy.root)
        pargen.prune_tree()
        pargen.set_params(pargen.curr_root)
        pargen.make_init_step(pargen.curr_root, pargen.gamma)
        pargen.make_recursive_step(pargen.curr_root, pargen.gamma, pargen._lambda)


    pargen1.indicate_offshoots(pargen1.curr_root)
    pargen2.indicate_offshoots_alt()

    # pargen2.set_params_alt(pargen2.taxonomy.root)
    assert len(pargen1.curr_taxonomy.nodes) == len(pargen2.curr_taxonomy.nodes)
    c = 0

    for node1, node2 in zip(pargen1.curr_taxonomy.nodes, pargen2.curr_taxonomy.nodes):
        node1G = {node.index + '-' + node.name for node in node1.G}
        node2G = {node.index + '-' + node.name for node in node2.G}
        # if node1G != node2G or np.abs(node1.V - node2.V) > 0.001:
        #     c+=1
        #     print(node1.V)
        #     print(node1G)
        #     print(node2.V)
        #     print(node2G)
        #     print('===')
        if node1.of == 1:
            print(node1.index)
            print(node1.of, node2.of)
    print('N times:', c)








