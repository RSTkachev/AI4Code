import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from functools import cmp_to_key
from time import time
from utils import kendall_tau

class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        valid_dataloader,
        savedir,
        device,
        epochs=10,
        saving_freq=5,
        lr=1e-4
    ):
        
        self.device = device

        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.criterion = nn.BCELoss()
        self.optimizer = optim.NAdam(self.model.parameters(), lr=lr)
        self.epochs = epochs

        self.best_score = -float('inf')
        self.best_model = None

        self.savedir = savedir
        self.saving_freq = saving_freq

    def train(self):
        for epoch in range(1, self.epochs + 1):
            print('*' * 80)
            print(f"Epoch {epoch}/{self.epochs}")
            start_time = time()
            train_loss = self._train_one_epoch()
            kendall_score = self._validate()

            print(f"Train loss: {train_loss:.4f}, Valid Kendall Tau correlation: {kendall_score:.4f}")
            print(f"Epoch execution time: {time() - start_time:.2f} seconds")

            if kendall_score > self.best_score:
                self.best_score = kendall_score
                self.best_model = {k: v.cpu() for k, v in self.model.state_dict().items()}
                print(f"New best model saved with Kendall Tau: {kendall_score:.4f}")

            if epoch % self.saving_freq == 0:
                self._save_checkpoint(epoch, train_loss)
    
        if self.best_model:
            torch.save(self.best_model, f'{self.savedir}best_model.pt')
            print("Best model saved as 'best_model.pt'.")

    def _train_one_epoch(self):
        self.model.train()
        train_loss = 0
        n_batches = 0
        
        for (first_cell, second_cell), train_label in tqdm(self.train_dataloader):
            self.optimizer.zero_grad()
            output = self.model(
                first_cell[0].squeeze(1).to(self.device),
                first_cell[1].squeeze(1).to(self.device),
                first_cell[2].squeeze(1).to(self.device),
                second_cell[0].squeeze(1).to(self.device),
                second_cell[1].squeeze(1).to(self.device),
                second_cell[2].squeeze(1).to(self.device)
            )
            loss = self.criterion(output, train_label.float().to(self.device))
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            n_batches += 1
        
        return train_loss / n_batches

    def _validate(self):
        self.model.eval()
        true_order = []
        predicted_order = []
        
        with torch.no_grad():
            for cells, correct_order in tqdm(self.valid_dataloader):
                sorted_cells = sorted(cells, key=cmp_to_key(self._custom_compare))
                sorted_order = [cell[0] for cell in sorted_cells]
                true_order.append(correct_order)
                predicted_order.append(sorted_order)
        
        return kendall_tau(true_order, predicted_order)

    def _save_checkpoint(self, epoch, train_loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': {k: v.cpu() for k, v in self.model.state_dict().items()},
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss
        }
        checkpoint_path = f'{self.savedir}checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}.")

    def _custom_compare(self, cell1, cell2):
        with torch.no_grad():
            result = self.model(
                cell1[1].squeeze(0).to(self.device), cell1[2].squeeze(0).to(self.device), cell1[3].squeeze(0).to(self.device),
                cell2[1].squeeze(0).to(self.device), cell2[2].squeeze(0).to(self.device), cell2[3].squeeze(0).to(self.device)
            )
            
            if result.item() <= 0.5:
                return -1
            else:
                return 1
            