import json
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def prepare_data(cell_type, cell_content, max_length=128):
    tokens = tokenizer(
        cell_content,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )

    cell_type = 1 if cell_type == "code" else 0
    type_tensor = torch.tensor([cell_type], dtype=torch.long)

    return tokens["input_ids"], tokens["attention_mask"], type_tensor

def load_data(csv_path, train_size=0.7, valid_size=0.2, test_size=0.1):
    info = pd.read_csv(csv_path)
    info["cell_order"] = info["cell_order"].apply(lambda x: x.split())
    
    indices = list(info.index)
    np.random.shuffle(indices)

    train_border = int(train_size * len(indices))
    valid_border = int((train_size + valid_size) * len(indices))

    train_data = info.loc[indices[:train_border]]
    valid_data = info.loc[indices[train_border:valid_border]]
    test_data = info.loc[indices[valid_border:]]

    return train_data, valid_data, test_data