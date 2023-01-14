#!usr/bin/python3

import torch 
import pandas as pd 

from models.trainer import load_checkpoint, model_train, model_test 
from torch_geometric.loader import DataLoader
from data_loader.dataloader import TrafficDataset, get_splits, distance_to_weight


def main():
    """
    Main function to train and test a model
    """
    #constant config to use throughout
    config = {
        'BATCH_SIZE': 50,
        'EPOCHS': 200,
        'WEIGHT_DECAY': 5e-5,
        'INITIAL_LR': 3e-4,
        'CHECKPOINT_DIR': './runs',
        'N_PRED': 9,
        'N_HIST': 12,
        'DROPOUT': 0.2,
        #number of possible 5 mins measurements per day
        'N_DAY_SLOT': 288, #(24*60)/5
        #number of days worth of data in the dataset
        'N_DAYS': 44,
        #If False, use GCN paper weight matrix, if true GAT paper weight matrix
        'USE_GAT_WEIGHTS': True,
        'N_NODE': 228
    }

    #number of possible windows in a day
    config['N_SLOT'] = config['N_DAY_SLOT'] - (config['N_PRED'] + config['N_HIST']) + 1

    #Load the weight matrix
    distances = pd.read_csv('./dataset/PeMSD7_W_228.csv', header=None).values
    W = distance_to_weight(distances, gat_version=config['USE_GAT_WEIGHTS'])
    #Load the dataset
    dataset = TrafficDataset(config, W)

    # Or load from a saved checkpoint
    # model = 


if __name__ == '__main__':
    main()