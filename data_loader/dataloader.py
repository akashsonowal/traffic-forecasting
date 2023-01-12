import torch
import numpy as np
import pandas as pd
import os
from torch_geometric.data import InMemoryDataset, Data
from shutil import copyfile 

from utils.math_utils import *

def distance_to_weight(W, sigma2=0.1, epsilon=0.5, gat_version=False):
    """
    Given distances between all nodes, convert into a weight matrix
    :param  W distances
    :param sigma2 user configurable parameter to adjust sparsity of matrix
    :param epsilon user configurable paramter to adjust sparsity of matrix
    :param gat_version If true, use 0/1 weights with self loops. Otherwise use float
    """ 
    n = W.shape[0]
    W = W / 1000.
    W2, W_mask = W*W, np.ones([n, n]) - np.identity(n)
    W = np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask #thresholded gaussian kernel method

    #If using the gat version of this, round to 0/1 and include self loops
    if gat_version:
        W[W>0] = 1
        W += np.identity(n)

    return W

class TrafficDataset(InMemoryDataset):
    """
    Dataset for Graph Neural Networks.
    """
    def __init__(self, config, W, root='', transform=None, pre_transform=None):
        self.config = config
        self.W = W
        super().__init__(root, transform, pre_transform)
        self.data, self.slices, self.n_node, self.mean, self.std_dev = torch.load(self.processed_paths[0])

    @property 
    def raw_file_names(self):
        return [os.path.join(self.raw_dir, 'PeMSD7_V_228.csv')]

    @property 
    def processed_file_names(self):
        return ['./data.pt']
    
    def download(self):
        copyfile('./dataset/PeMSD7_V_228.csv', os.path.join(self.raw_dir, 'PeMSD7_V_228.csv'))
    
    def process(self):
        """
        Process the raw datasets into saved .pt dataset for later use.
        Note that any self.fiels won't exist if loading straight from .pt file
        """
        #Data preprocessing and loading
        data = pd.read_csv(self.raw_file_names[0], header=None).values
        #Technically using the validation and test datasets here, but it is fine, would normally
        #get the mean and std_dev from a large dataset
        mean = np.mean(data)
        std_dev = np.std(data)
        data = z_score(data, np.mean(data), np.std(data))

        _, n_node = data.shape
        n_window = self.config['N_PRED'] + self.config['N_HIST']

        #manipulate n x n matrix into 2 x num_edges






