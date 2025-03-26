import torch
import json

from tqdm import tqdm
from torch.utils.data import Dataset

from Datasets.cell import Cell


class CellDataset(Dataset):
    def __init__(self, path, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.files = {}

        for filename in tqdm(self.data.index):
            cells_dict = {}
            cells = self.data.loc[filename, "cell_order"]
            with open(f"{path}{filename}.json") as file:
                json_code = json.load(file)
            for cell in cells:
                input_ids, att_mask, cell_type = self.prepare_data(
                    json_code["cell_type"][cell], json_code["source"][cell]
                )
                cells_dict[cell] = Cell(input_ids, att_mask, cell_type)
            self.files[filename] = cells_dict

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def prepare_data(self, cell_type, cell_content):

        tokens = self.tokenizer(
            cell_content,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        cell_type = 1 if cell_type == "code" else 0

        type_tensor = torch.tensor([cell_type], dtype=torch.long)

        return (tokens["input_ids"], tokens["attention_mask"], type_tensor)
