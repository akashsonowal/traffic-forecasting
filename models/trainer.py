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

@torch.no_grad()
def eval(model, device, dataloader, type=''):
    """
    Evaluation function to evaluate model on data
    :param model Model to evaluate 
    :param device Device to evaluate on
    :param dataloader Dataloader
    :param type Name of evaluation type, e.g. Train/Val/Test
    """
    model.eval()
    model.to(device)
    
    mae = 0
    rmse = 0
    mape = 0
    n = 0

    #Evaluate model on all the data
    for i, batch in enumerate(dataloader):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch, device)
            truth = batch.y.view(pred.shape)
            if i == 0:
                y_pred = torch.zeros(len(dataloader), pred.shape[0], pred.shape[1])
                y_truth = torch.zeros(len(dataloader), pred.shape[0], pred.shape[1])
            truth = un_z_score(truth, dataloader.dataset.mean(), dataloader.dataset.std_dev())
            pred = un_z_score(pred, dataloader.dataset.mean(), dataloader.dataset.std_dev())
            y_pred[i, :pred.shape[0], :] = pred
            y_truth[i, :pred_shape[0], :] = truth 
            rmse += RMSE(truth, pred)
            mae += MAE(truth, pred)
            mape += MAPE(truth, pred)
            n += 1
    rmse, mae, mape = rmse / n, mae / n, mape / n 
    print(f'{type}, MAE: {mae}, RMSE: {RMSE}, MAPE: {mape}')

    #get the average score for each metric in each batch
    return rmse, mae, mape, y_pred, y_truth 


def train(model, device, dataloader, optimizer, loss_fn, epoch):
    """
    Evaluation function to evaluate model on data
    :param model Model to evaluate
    :param device Device to evaluate on
    :param dataloader Dataloader
    :optimizer Optimizer to use
    :param loss_func Loss function
    :param epoch current epoch
    """
    model.train()
    for _, batch in enumerate(tqdm(dataloader, desc=f'Epochs {epoch}')):
        batch.to(device)
        optimizer.zero_grad()
        y_pred = torch.squeeze(model(batch, device))
        loss = loss_fn()(y_pred.float(), torch.squeeze(batch.y).float())
        writer.add_scaler('Loss/train', loss, epoch)
        loss.backward()
        optimizer.step()
    return loss

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
    optimizer = optim.Adam(model_parameters(), lr=config['INITIAL_LR'], weight_decay=config['WEIGHT_DECAY'])
    loss_fn = torch.nn.MSELoss

    model.to(device)

    # For every epoch train the model on training dataset. Evaluate the model on evaluation dataset
    for epoch in range(config['EPOCHS']):
        loss = train(model, device, train_dataloader, optimizer, loss_fn, epoch)
        print(f'Loss: {loss:.3f}')
        if epoch % 5 == 0:
            train_mae, train_rmse, train_mape, _, _ = eval(model, device, train_dataloader, 'Train')


def model_test():
    pass

