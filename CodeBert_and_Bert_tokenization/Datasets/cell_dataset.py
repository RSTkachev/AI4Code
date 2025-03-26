import torch
import json

from tqdm import tqdm
from torch.utils.data import Dataset

from Datasets.cell import Cell


class CellDataset(Dataset):
    def __init__(self, path, data, code_tokenizer, text_tokenizer, max_length):
        self.data = data
        self.code_tokenizer = code_tokenizer
        self.text_tokenizer = text_tokenizer
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

        if cell_type == "code":
            tokenizer = self.code_tokenizer
            type_label = 1
        else:
            tokenizer = self.text_tokenizer
            type_label = 0

        tokens = tokenizer(
            cell_content,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
    
        type_tensor = torch.tensor([type_label], dtype=torch.long)

        return (tokens["input_ids"], tokens["attention_mask"], type_tensor)
