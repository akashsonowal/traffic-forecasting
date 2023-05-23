#!usr/bin/python3

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from traffic_forecasting.data import TrafficDataset
from traffic_forecasting.utils import distance_to_weight, get_splits, load_from_checkpoint
from traffic_forecasting.model import ST_GAT
from traffic_forecasting.trainer import model_train, model_test


def main():
    config = {
        "BATCH_SIZE": 50,
        "EPOCHS": 200,
        "WEIGHT_DECAY": 5e-5,
        "INITIAL_LR": 3e-4,
        "CHECKPOINT_DIR": "./runs",
        "N_PRED": 9,
        "N_HIST": 12,
        "DROPOUT": 0.2,
        "N_DAY_SLOT": 288,  # (24 * 60)/5 number of possible 5 mins measurements per day
        "N_DAYS": 44,  # number of days worth of data in the dataset
        "USE_GAT_WEIGHTS": True,  # If True, use GAT weight matrix, else GCN weight matrix
        "N_NODE": 228,
    }
    config["N_SLOT"] = (
        config["N_DAY_SLOT"] - (config["N_PRED"] + config["N_HIST"]) + 1
    )  # number of possible windows in a day
    distances = pd.read_csv(
        "./data/raw/PeMSD7_W_228.csv", header=None
    ).values  # (228, 228)
    W = distance_to_weight(distances, gat_version=config["USE_GAT_WEIGHTS"])
    dataset = TrafficDataset(config, W)

    # total of 44 days in the dataset, use 34 for training, 5 for val, 5 for test
    train, val, test = get_splits(dataset, config["N_SLOT"], (34, 5, 5))
    train_dataloader = DataLoader(train, batch_size=config["BATCH_SIZE"], shuffle=True)
    val_dataloader = DataLoader(val, batch_size=config["BATCH_SIZE"], shuffle=True)
    test_dataloader = DataLoader(test, batch_size=config["BATCH_SIZE"], shuffle=False)

    # Get gpu if you can
    device = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    model = ST_GAT(
        in_channels=config["N_HIST"],
        out_channels=config["N_PRED"],
        n_nodes=config["N_NODE"],
        dropout=config["DROPOUT"],
    )

    # Configure and train model
    model_train(model, train_dataloader, val_dataloader, config, device)
    model = load_from_checkpoint("./runs/stgat_checkpoint.pt", config)
    # Test model
    model_test(model, test_dataloader, config, device)


if __name__ == "__main__":
    main()
