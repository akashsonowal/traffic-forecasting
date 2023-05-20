import os
import torch
import numpy as np
import pandas as pd
from shutil import copyfile

from torch_geometric.data import InMemoryDataset, Data

from ..utils import z_score


def distance_to_weight(W, sigma2=0.1, epsilon=0.5, gat_version=False):
    """
    Given distances between all nodes, convert into a weight matrix
    :param  W distances
    :param sigma2 user configurable parameter to adjust sparsity of matrix
    :param epsilon user configurable parameter to adjust sparsity of matrix
    :param gat_version If true, use 0/1 weights with self loops. Otherwise use float
    """
    n = W.shape[0]
    W = W / 1000.0
    W2, W_mask = W * W, np.ones([n, n]) - np.identity(n)
    W = (
        np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask
    )  # thresholded gaussian kernel method

    # If using the gat version of this, round to 0/1 and include self loops
    if gat_version:
        W[W > 0] = 1
        W += np.identity(n)

    return W


class TrafficDataset(InMemoryDataset):
    """
    Dataset for Graph Neural Networks.
    """

    def __init__(self, config, W, root="data/raw/", transform=None, pre_transform=None):
        self.config = config
        self.W = W
        super().__init__(root, transform, pre_transform)
        self.data, self.slices, self.n_node, self.mean, self.std_dev = torch.load(
            self.processed_paths[0]
        )

    @property
    def raw_file_names(self):
        return [os.path.join(self.raw_dir, "PeMSD7_V_228.csv")]

    @property
    def processed_file_names(self):
        return ["./data.pt"]

    def download(self):  # velocity dataset
        copyfile(
            "./data/raw/PeMSD7_V_228.csv",
            os.path.join(self.raw_dir, "PeMSD7_V_228.csv"),
        )

    def process(self):
        """
        Process the raw datasets into saved .pt dataset for later use.
        Note that any self.fields won't exist if loading straight from .pt file
        """
        # Data preprocessing and loading
        data = pd.read_csv(self.raw_file_names[0], header=None).values
        # Technically using the validation and test datasets here, but it is fine, would normally
        # get the mean and std_dev from a large dataset
        mean = np.mean(data)
        std_dev = np.std(data)
        data = z_score(data, np.mean(data), np.std(data))

        _, n_node = data.shape  # (12672, 228)
        n_window = (
            self.config["N_PRED"] + self.config["N_HIST"]
        )  # window size = 9 + 12 = 21

        # manipulate n x n matrix into 2 x num_edges
        edge_index = torch.zeros((2, n_node**2), dtype=torch.long)  # COO format
        # create an edge_attr matrix with our weights (num_edges x 1) --> our edge features are dim 1
        edge_attr = torch.zeros((n_node**2, 1))
        num_edges = 0
        for i in range(n_node):
            for j in range(n_node):
                if self.W[i, j] != 0:
                    edge_index[0, num_edges] = i
                    edge_index[1, num_edges] = j
                    edge_attr[num_edges] = self.W[i, j]
                    num_edges += 1
        # using resize_ to just keep the first num_edges entries (where W is non zero)
        edge_index = edge_index.resize_(2, num_edges)
        edge_attr = edge_attr.resize_(num_edges, 1)

        sequences = []
        # T x F x N
        for i in range(self.config["N_DAYS"]):  # 44 days
            for j in range(self.config["N_SLOT"]):  # No. of windows in a day
                # for each time point construct a different graph with data object
                # Docs here: https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data
                g = Data()
                g.__num_nodes__ = n_node

                g.edge_index = edge_index
                g.edge_attr = edge_attr

                sta = i * self.config["N_DAY_SLOT"] + j
                end = sta + n_window  # n_window is window size
                # (F, N) switched to (N, F)
                # [21, 228] -> [228, 21]
                full_window = np.swapaxes(
                    data[sta:end, :], 0, 1
                )  # rows becomes cols and vice versa
                g.x = torch.FloatTensor(
                    full_window[:, 0 : self.config["N_HIST"]]
                )  # (228, 12)
                g.y = torch.FloatTensor(
                    full_window[:, self.config["N_HIST"] : :]
                )  # (228, 9)
                sequences += [g]

        # make the actual dataset
        data, slices = self.collate(sequences)  # concatenate graph data objects
        # slices is a list of number of nodes of each graph. For our case it is [228, 228, ...44 x N_SLOT]
        # data is tuple of (x,  edge_index, edge_attr) where
        # x is the concatenation of node feature tensors of both the graphs along the first dimension of shape
        # edge_index is the concatenation of edge index tensors of both the graphs along the last dimension of shape
        # edge_attr is the concatenation of edge attribute tensors of both the graphs along the first dimension of shape
        torch.save((data, slices, n_node, mean, std_dev), self.processed_paths[0])


def get_splits(dataset: TrafficDataset, n_slot, splits):
    """
    Given the data, split it into random subsets of train, val and test as given by splits
    :param dataset: TrafficDataset object to split
    :param n_slot: Number of possible sliding windows in a day
    :param splits: (train, val, test) ratios
    """
    split_train, split_val, _ = splits
    i = n_slot * split_train
    j = n_slot * split_val
    train = dataset[:i]
    val = dataset[i : i + j]
    test = dataset[i + j :]

    return train, val, test
