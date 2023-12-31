import torch
import torch_geometric
import networkx as nx
from torch_geometric.data import Data
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.utils.convert import to_networkx, from_networkx
from tasks.braess_rewire import braess_rewire
class TreeDataset(object):
    def __init__(self, depth):
        super(TreeDataset, self).__init__()
        self.depth = depth
        self.num_nodes, self.edges, self.leaf_indices = self._create_blank_tree()
        self.criterion = F.cross_entropy

    def add_child_edges(self, cur_node, max_node):
        edges = []
        leaf_indices = []
        stack = [(cur_node, max_node)]
        while len(stack) > 0:
            cur_node, max_node = stack.pop()
            if cur_node == max_node:
                leaf_indices.append(cur_node)
                continue
            left_child = cur_node + 1
            right_child = cur_node + 1 + ((max_node - cur_node) // 2)
            edges.append([left_child, cur_node])
            edges.append([right_child, cur_node])
            stack.append((right_child, max_node))
            stack.append((left_child, right_child - 1))
        return edges, leaf_indices

    def _create_blank_tree(self):
        max_node_id = 2 ** (self.depth + 1) - 2
        edges, leaf_indices = self.add_child_edges(cur_node=0, max_node=max_node_id)
        return max_node_id + 1, edges, leaf_indices

    def create_blank_tree(self, add_self_loops=True):
        edge_index = torch.tensor(self.edges).t()
        if add_self_loops:
            edge_index, _ = torch_geometric.utils.add_remaining_self_loops(edge_index=edge_index, )
        return edge_index

    def generate_data(self, train_fraction):
        data_list = []
        print("Generating Tree Dataset...")
        for comb in self.get_combinations():
            edge_index = self.create_blank_tree(add_self_loops=True)
            nodes = torch.tensor(self.get_nodes_features(comb), dtype=torch.long)
            root_mask = torch.tensor([True] + [False] * (len(nodes) - 1))
            label = self.label(comb)
            tree_data = Data(x=nodes, edge_index=edge_index, root_mask=root_mask, y=label)
            print(tree_data)
            print("Performing Braess Addition...")
            nxtree = to_networkx(tree_data)
            adj = nx.adjacency_matrix(nxtree)
            newadj = braess_rewire(adj,50)
            new_data = from_networkx(nx.from_scipy_sparse_array(newadj))
            tree_data.edge_index = new_data.edge_index
            print("Done!")
            data_list.append(tree_data)

        dim0, out_dim = self.get_dims()
        X_train, X_test = train_test_split(
            data_list, train_size=train_fraction, shuffle=True, stratify=[data.y for data in data_list])


        return X_train, X_test, dim0, out_dim, self.criterion

    # Every sub-class should implement the following methods:
    def get_combinations(self):
        raise NotImplementedError

    def get_nodes_features(self, combination):
        raise NotImplementedError

    def label(self, combination):
        raise NotImplementedError

    def get_dims(self):
        raise NotImplementedError

