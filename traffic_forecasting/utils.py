import numpy as np
import matplotlib.pyplot as plt
import torch

from .data import TrafficDataset
from .model import ST_GAT


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


def un_z_score(x_normed, mean, std):
    """
    Undo the z-score calculation
    :param x_normed torch array, input_array to be un-normalized
    :param mean: float, the value of mean
    :param std: float, the value of standard deviation
    """
    return x_normed * std + mean


def RMSE(v, v_):
    """
    Mean squared error.
    :param v: torch array, ground truth
    :param v_: torch array, prediction
    :return: torch scaler, RMSE averages on all elements of input
    """
    return torch.sqrt(torch.mean(v_ - v) ** 2)


def MAE(v, v_):
    """
    Mean Absolute Error
    :param v: torch array, ground truth.
    :param v_: torch array, prediction.
    :return: torch scalar, MAE averages on all elements of input.
    """
    return torch.mean(torch.abs(v_ - v))


def MAPE(v, v_):
    """
    Mean absolute percentage error, given as a % (e.g. 99 -> 99%)
    :param v: torch array, ground truth.
    :param v_: torch array, prediction.
    :return: torch scalar, MAPE averages on all elements of input.
    """
    return torch.mean(torch.abs(v_ - v) / (v + 1e-15) * 100)


def load_from_checkpoint(checkpoint_path, config):
    """
    Load a model from the checkpoint
    :param checkpoint_path Path to checkpoint
    :param config Configuration to load model with
    """
    model = ST_GAT(
        in_channels=config["N_HIST"],
        out_channels=config["N_PRED"],
        n_nodes=config["N_NODE"],
    )
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def plot_predictions(y_pred, y_truth, node, config):
    s = y_truth.shape  #
    print("************* s shape", s)
    y_truth = y_truth.reshape(s[0], config["BATCH_SIZE"], config["N_NODE"], s[-1]) # (, batch_size, nodes, days)
    # just get the first prediction out for the nth node
    y_truth = y_truth[:, :, node, 0]
    # Flatten to get the predictions for entire test dataset
    y_truth = torch.flatten(y_truth)
    day0_truth = y_truth[: config["N_SLOT"]]

    # Calculate the predicted
    s = y_pred.shape
    y_pred = y_pred.reshape(s[0], config["BATCH_SIZE"], config["N_NODE"], s[-1])
    # just get the first prediction out for the nth node
    y_pred = y_pred[:, :, node, 0]
    # Flatten to get the predictions for entire test dataset
    y_pred = torch.flatten(y_pred)
    # Just grab the first day
    day0_pred = y_pred[: config["N_SLOT"]]

    t = [t for t in range(0, config["N_SLOT"], 5, 5)]
    plt.plot(t, day0_pred, label="ST-GAT")
    plt.plot(t, day0_truth, label="truth")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Speed prediction")
    plt.title("Predictions of traffic over time")
    plt.legend()
    plt.savefig(f"../assets/traffic_on_node{node}_day0.png")
    plt.show()
