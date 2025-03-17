import pandas as pd
import numpy as np

from transformers import BertTokenizer
from torch.utils.data import DataLoader

from Datasets.cell_dataset import CellDataset
from Datasets.sampler import CellSampler
from utils import prepare_folders, get_device
from model import OrderPredictionModel
from train import Trainer


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    print("*" * 80)
    print("Reading data")
    info = pd.read_csv('./Data/train_orders.csv', index_col='id')
    info['cell_order'] = info['cell_order'].apply(lambda x: x.split())
    indeces = list(info.index)
    np.random.shuffle(indeces)

    train_size = 0.7
    valid_size = 0.2
    test_size = 0.1

    train_border = int(train_size * len(indeces))
    valid_border = int((train_size + valid_size) * len(indeces))

    train_data = info.loc[indeces[:train_border]]
    valid_data = info.loc[indeces[train_border:valid_border]]
    test_data = info.loc[indeces[valid_border:]]

    train_data_short = train_data.iloc[:100]
    valid_data_short = valid_data.iloc[:100]

    train_dataset = CellDataset('./Data/train/', train_data_short, tokenizer, 128)
    train_sampler = CellSampler(train_data_short)
    train_dataloader = DataLoader(train_dataset, 64, drop_last=True, sampler=train_sampler)

    valid_dataset = CellDataset('./Data/train/', valid_data_short, tokenizer, 128)
    valid_sampler = CellSampler(valid_data_short, 42)
    valid_dataloader = DataLoader(valid_dataset, 64, drop_last=True, sampler=valid_sampler)

    model = OrderPredictionModel(128)
    savedir = prepare_folders()
    device = get_device()
    
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        savedir=savedir,
        device=device,
        epochs=10,
        saving_freq=5,
        lr=1e-4
    )

    trainer.train()