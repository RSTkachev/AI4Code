import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, BertTokenizer

import os

from Datasets.notebook_dataset import ListWiseCellDataset
from utils import prepare_folders, get_device
from model import ListWiseOrderPredictionModel
from train import ListWiseTrainer
from test import ListWiseTester


if __name__ == "__main__":

    code_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    text_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")

    print("*"*80)
    print("Reading data")

    info = pd.read_csv("../AI4Code_data/train_orders.csv", index_col="id")
    info["cell_order"] = info["cell_order"].apply(lambda x: x.split())
    indeces = list(info.index)

    rng = np.random.default_rng(42)
    rng.shuffle(indeces)

    train_size = 0.7
    valid_size = 0.2
    test_size  = 0.1

    train_border = int(train_size * len(indeces))
    valid_border = int((train_size + valid_size) * len(indeces))

    train_data = info.loc[indeces[:train_border]]
    valid_data = info.loc[indeces[train_border:valid_border]]
    test_data  = info.loc[indeces[valid_border:]]

    train_data_short = train_data.iloc[:40]
    valid_data_short = valid_data.iloc[:10]
    test_data_short  = test_data.iloc[:10]

    train_dataset = ListWiseCellDataset(
        path="../AI4Code_data/train/",
        data=train_data_short,
        code_tokenizer=code_tokenizer,
        text_tokenizer=text_tokenizer,
        max_length=128
    )

    valid_dataset = ListWiseCellDataset(
        path="../AI4Code_data/train/",
        data=valid_data_short,
        code_tokenizer=code_tokenizer,
        text_tokenizer=text_tokenizer,
        max_length=128
    )

    test_dataset = ListWiseCellDataset(
        path="../AI4Code_data/train/",
        data=test_data_short,
        code_tokenizer=code_tokenizer,
        text_tokenizer=text_tokenizer,
        max_length=128
    )

    model = ListWiseOrderPredictionModel(hidden_dim=128, dropout_prob=0.1)

    savedir = prepare_folders(savedir_name="checkpoints_listwise")
    device = get_device()

    trainer = ListWiseTrainer(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        device=device,
        save_dir=savedir,
        lr=1e-4,
        epochs=5 
    )
    trainer.train()

    best_model_path = os.path.join(savedir, "best_model.pt")
    best_weights = torch.load(best_model_path, map_location="cpu")
    model.load_state_dict(best_weights)
    model.to(device)

    tester = ListWiseTester(model, device)
    result = tester.test(test_dataset)
    print("*"*80)
    print(f"Test Kendall Tau score: {result:.4f}")
