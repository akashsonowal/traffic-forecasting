import torch 
import torch.optim as optim
from tqdm import tqdm
import time 
import os
import matplotlib.pyplot as plt

from models.st_gat import STGAT
from utils.math_utils import *
from torch.utils.tensorboard import SummaryWriter 

# Make a tensorboard writer
writer = SummaryWriter()


def model_train(train_dataloader, val_dataloader, config, device):
    """
    Train the ST-GAT model. Evaluate on validation dataset as you go.
    :param train_dataloader Data loader of training dataset
    :param val_dataloader Data loader of validation dataset
    :param config configuration to use
    :param device Device to evaluate on
    """
    
    # Make the model. Each dataset in the graph is 228 x 12: N x F (N = #nodes, F = time window)
    model = STGAT(in_channels=config['N_HIST'], out_channels=config['N_PRED'], n_nodes=config['N_NODE'], dropout=config['DROPOUT'])
    

def model_test():
    pass

