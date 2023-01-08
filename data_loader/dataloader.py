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
    def 





