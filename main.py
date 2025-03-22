import pandas as pd
import numpy as np
import torch
import torch.optim as optim

from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from Datasets.train_val_cell_dataset import TrainValCellDataset
from Datasets.test_cell_dataset import TestCellDataset
from Datasets.sampler import CellSampler
from utils import prepare_folders, get_device
from model import OrderPredictionModel
from train import Trainer
from test import Tester


if __name__ == "__main__":
    data_dir = "./Data/"
    model_name = "microsoft/codebert-base"
    
    save_dir = prepare_folders()
    device = get_device()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)

    print("*" * 80)
    print("Reading data")
    info = pd.read_csv(f"{data_dir}train_orders.csv", index_col="id")
    info["cell_order"] = info["cell_order"].apply(lambda x: x.split())
    indeces = list(info.index)

    rng = np.random.default_rng(42)
    rng.shuffle(indeces)

    train_size = 0.7
    valid_size = 0.2
    test_size = 0.1

    train_border = int(train_size * len(indeces))
    valid_border = int((train_size + valid_size) * len(indeces))

    train_data = info.loc[indeces[:train_border]]
    valid_data = info.loc[indeces[train_border:valid_border]]
    test_data = info.loc[indeces[valid_border:]]

    train_data_short = train_data.iloc[:10000]
    valid_data_short = valid_data.iloc[:100]
    test_data_short = test_data.iloc[:100]

    train_dataset = TrainValCellDataset(f"{data_dir}train/", train_data_short, tokenizer, 128)
    train_sampler = CellSampler(train_data_short)
    train_dataloader = DataLoader(train_dataset, 64, drop_last=True, sampler=train_sampler)

    valid_dataset = TrainValCellDataset(f"{data_dir}train/", valid_data_short, tokenizer, 128)
    valid_sampler = CellSampler(valid_data_short, 42)
    valid_dataloader = DataLoader(valid_dataset, 64, drop_last=True, sampler=valid_sampler)

    test_dataset = TestCellDataset(f"{data_dir}train/", test_data_short, tokenizer, 128)
    test_dataloader = DataLoader(test_dataset, 1, shuffle=False)

    model = OrderPredictionModel(model_name, 256, 0.2)
    optimizer = optim.NAdam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        save_dir=save_dir,
        device=device,
        epochs=10,
        early_stopping=5,
        saving_freq=5,
    )

    trainer.train()

    best_model_weights = torch.load(f"{save_dir}best_model.pt", weights_only=True)
    model.load_state_dict(best_model_weights)

    print("*" * 80)
    print("Test model")
    tester = Tester(model, device)
    result = tester.test(test_dataloader)
    print(f"Kendall Tau score: {result}")
