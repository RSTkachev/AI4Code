import torch
from torch.utils.data import Dataset

class ListWiseCellDataset(Dataset):
    def __init__(self, path, data, code_tokenizer, text_tokenizer, max_length=128):
        super().__init__()
        self.path = path
        self.data = data
        self.notebook_ids = list(data.index)
        self.code_tokenizer = code_tokenizer
        self.text_tokenizer = text_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.notebook_ids)

    def __getitem__(self, idx):
        notebook_id = self.notebook_ids[idx]
        cell_order = self.data.loc[notebook_id, "cell_order"]
        
        if isinstance(cell_order, str):
            cell_order = cell_order.split()
            
        import json
        with open(f"{self.path}{notebook_id}.json", "r") as f:
            nb_json = json.load(f)

        input_ids_list = []
        attn_mask_list = []
        cell_type_list = []

        for cell_id in cell_order:
            ctype = nb_json["cell_type"][cell_id]
            csource = nb_json["source"][cell_id]

            if ctype == "code":
                tok = self.code_tokenizer(
                    csource,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                cell_type_list.append(1)
            else:  # markdown
                tok = self.text_tokenizer(
                    csource,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                cell_type_list.append(0)

            input_ids_list.append(tok["input_ids"].squeeze(0))
            attn_mask_list.append(tok["attention_mask"].squeeze(0))

        input_ids_tensor = torch.stack(input_ids_list, dim=0)
        attn_mask_tensor = torch.stack(attn_mask_list, dim=0)
        cell_type_tensor = torch.tensor(cell_type_list, dtype=torch.long)

        return {
            "id_notebook": notebook_id,
            "cell_ids": cell_order,
            "input_ids": input_ids_tensor,
            "attention_mask": attn_mask_tensor,
            "cell_types": cell_type_tensor
        }
