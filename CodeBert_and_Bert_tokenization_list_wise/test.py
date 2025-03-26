
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import kendall_tau

def custom_collate_fn(batch):
    ids_notebook = [item["id_notebook"] for item in batch]
    
    cell_ids_lists = [item["cell_ids"] for item in batch]
    
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    cell_types = [item["cell_types"] for item in batch]
    
    return {
        "id_notebook": ids_notebook,
        "cell_ids": cell_ids_lists,
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "cell_types": cell_types
    }


class ListWiseTester:

    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def test(self, test_dataset):
        loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
        all_gt_orders = []
        all_pred_orders = []

        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch["cell_ids"], list):
                    cell_ids = batch["cell_ids"][0]
                else:
                    cell_ids = list(batch["cell_ids"][0])
                    
                input_ids = batch["input_ids"][0].to(self.device)
                att_mask = batch["attention_mask"][0].to(self.device)
                cell_types = batch["cell_types"][0].to(self.device)

                scores = self.model(input_ids, att_mask, cell_types)
                scores_cpu = scores.cpu().numpy()
                
                idx_sorted = np.argsort(-scores_cpu)
                idx_sorted = idx_sorted[:len(cell_ids)]
                
                predicted_ids = [cell_ids[i] for i in idx_sorted]

                all_gt_orders.append(list(cell_ids))
                all_pred_orders.append(predicted_ids)

        ktau = kendall_tau(all_gt_orders, all_pred_orders)
        return ktau


