import os
import time
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .utils import RMSE, MAE, MAPE, plot_predictions

# Make a tensorboard writer
writer = SummaryWriter()


@torch.no_grad()
def eval(model, device, dataloader, type=""):
    model.eval()
    model.to(device)

    mae = 0
    rmse = 0
    mape = 0
    n = 0

    for i, batch in enumerate(dataloader):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch, device)
            truth = batch.y.view(pred.shape) # (11400, 9)
            if i == 0:
                # (183, 11400, 9) for train_loader, 183 = 9112 / 50
                # (27, 11400, 9) for val_loader and test_loader, 27 = 1340 / 50
                y_pred = torch.zeros(len(dataloader), pred.shape[0], pred.shape[1])
                y_truth = torch.zeros(len(dataloader), pred.shape[0], pred.shape[1])
            print("y pred shape", y_pred.shape)
            truth = truth * dataloader.dataset.std_dev + dataloader.dataset.mean
            pred = pred * dataloader.dataset.std_dev + dataloader.dataset.mean
            y_pred[i, : pred.shape[0], :] = pred
            y_truth[i, : truth.shape[0], :] = truth
            rmse += RMSE(truth, pred)
            mae += MAE(truth, pred)
            mape += MAPE(truth, pred)
            n += 1
    rmse, mae, mape = rmse / n, mae / n, mape / n
    print(f"{type}, MAE: {mae}, RMSE: {RMSE}, MAPE: {mape}")

    # get the average score for each metric in each batch
    return rmse, mae, mape, y_pred, y_truth


def train_one_epoch(model, device, dataloader, optimizer, loss_fn, epoch):
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
    for _, batch in enumerate(tqdm(dataloader, desc=f"Epochs {epoch}")):
        batch.to(device)
        y_pred = torch.squeeze(model(batch, device))
        loss = loss_fn()(y_pred.float(), torch.squeeze(batch.y).float())
        writer.add_scalar("Loss/train", loss, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss


def model_train(model, train_dataloader, val_dataloader, config, device):
    """
    Train the ST-GAT model. Evaluate on validation dataset as you go.
    :param train_dataloader Data loader of training dataset
    :param val_dataloader Data loader of validation dataset
    :param config configuration to use
    :param device Device to evaluate on
    """
    # Make the model. Each dataset in the graph is 228 x 12: N x F (N = #nodes, F = time window)
    optimizer = optim.Adam(
        model.parameters(), lr=config["INITIAL_LR"], weight_decay=config["WEIGHT_DECAY"]
    )
    loss_fn = torch.nn.MSELoss

    model.to(device)

    # For every epoch train the model on training dataset. Evaluate the model on evaluation dataset
    for epoch in tqdm(range(config["EPOCHS"])):
        loss = train_one_epoch(
            model, device, train_dataloader, optimizer, loss_fn, epoch
        )
        print(f"Loss: {loss:.3f}")
        if epoch % 5 == 0:
            train_mae, train_rmse, train_mape, _, _ = eval(
                model, device, train_dataloader, "Train"
            )
            val_mae, val_rmse, val_mape, _, _ = eval(
                model, device, val_dataloader, "Valid"
            )
            writer.add_scalar(f"MAE/train", train_mae, epoch)
            writer.add_scalar(f"RMSE/train", train_rmse, epoch)
            writer.add_scalar(f"MAPE/train", train_mape, epoch)
            writer.add_scalar(f"MAE/val", val_mae, epoch)
            writer.add_scalar(f"RMSE/val", val_rmse, epoch)
            writer.add_scalar(f"MAPE/val", val_mape, epoch)

    writer.flush()
    # Save the model
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        os.path.join(config["CHECKPOINT_DIR"], f"stgat_checkpoint.pt"),
    )


def model_test(model, test_dataloader, config, device):
    """
    Test the ST-GAT model
    :param test_dataloader Data loader of test dataset
    :param device Device to evaluate on
    """
    _, _, _, y_pred, y_test = eval(model, device, test_dataloader, "Test")
    plot_predictions(y_pred, y_test, 0, config)
