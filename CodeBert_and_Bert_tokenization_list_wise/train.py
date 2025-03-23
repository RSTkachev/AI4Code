import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import kendall_tau

import os

def listnet_loss(scores, n_items):
    rank_tensor = torch.arange(n_items, device=scores.device, dtype=torch.float32)
    q_unnorm = torch.exp(-rank_tensor)
    q = q_unnorm / q_unnorm.sum()

    p = F.softmax(scores, dim=0)
    eps = 1e-10
    loss = - (q * torch.log(p + eps)).sum()
    return loss

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


class ListWiseTrainer:
    def __init__(self, model, train_dataset, valid_dataset, device, save_dir, lr=1e-4, epochs=5):
        self.model = model
        self.device = device
        self.epochs = epochs
        self.save_dir = save_dir

        self.train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
        self.valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.best_kendall = -999.0
        self.best_model_state = None

    def train(self):
        self.model.to(self.device)
        for epoch in range(1, self.epochs+1):
            print(f"\nEpoch [{epoch}/{self.epochs}]")
            train_loss = self._train_one_epoch()
            val_kendall = self._validate()
            print(f"Train loss: {train_loss:.4f}, Valid Kendall Tau: {val_kendall:.4f}")

            if val_kendall > self.best_kendall:
                self.best_kendall = val_kendall

                self.best_model_state = {
                    k: v.cpu() for k, v in self.model.state_dict().items()
                }
                print("New best model saved.")

        if self.best_model_state is not None:
            torch.save(self.best_model_state, os.path.join(self.save_dir, "best_model.pt"))
            print(f"Best model with Kendall Tau={self.best_kendall:.4f} saved.")

    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(self.train_loader, desc="Training"):
            self.optimizer.zero_grad()
            input_ids = batch["input_ids"][0].to(self.device)
            att_mask  = batch["attention_mask"][0].to(self.device)
            cell_types= batch["cell_types"][0].to(self.device)
            N = input_ids.size(0)

            scores = self.model(input_ids, att_mask, cell_types)
            loss = listnet_loss(scores, N)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(1, n_batches)

    def _validate(self):
        self.model.eval()
        all_gt_orders = []
        all_pred_orders = []

        with torch.no_grad():
            for batch in self.valid_loader:
                notebook_id = batch["id_notebook"][0]
                
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
