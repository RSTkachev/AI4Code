import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from time import time


class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        valid_dataloader,
        savedir,
        device,
        epochs=10,
        early_stopping=5,
        saving_freq=5,
        lr=1e-4,
    ):

        self.device = device

        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.criterion = nn.BCELoss()
        self.optimizer = optim.NAdam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.early_stopping = early_stopping

        self.best_score = -float("inf")
        self.best_model = None

        self.savedir = savedir
        self.saving_freq = saving_freq

    def train(self):
        early_stopping_remaining = self.early_stopping
        print("*" * 80)
        print(f"Train model")

        for epoch in range(1, self.epochs + 1):
            print("*" * 80)
            print(f"Epoch {epoch}/{self.epochs}")
            start_time = time()
            train_loss = self._train_one_epoch()
            valid_score = self._validate()

            print(f"Train loss: {train_loss:.4f}, Valid accuracy: {valid_score:.4f}")
            print(f"Epoch execution time: {time() - start_time:.2f} seconds")

            if valid_score > self.best_score:
                early_stopping_remaining = self.early_stopping
                self.best_score = valid_score
                self.best_model = {k: v.cpu() for k, v in self.model.state_dict().items()}
                print(f"New best model saved with valid accuracy: {valid_score:.4f}")
            else:
                early_stopping_remaining -= 1

            if epoch % self.saving_freq == 0:
                self._save_checkpoint(epoch, train_loss)

            if not early_stopping_remaining:
                print(f"Training stopped at {epoch} epoch")
                break

        if self.best_model:
            torch.save(self.best_model, f"{self.savedir}best_model.pt")
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
                second_cell[2].squeeze(1).to(self.device),
            )
            loss = self.criterion(output, train_label.float().to(self.device))
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        return train_loss / n_batches

    def _validate(self):
        self.model.eval()
        score = 0
        n_batches = 0

        with torch.no_grad():
            for (first_cell, second_cell), correct_order in tqdm(self.valid_dataloader):
                n_batches += 1
                output = self.model(
                    first_cell[0].squeeze(1).to(self.device),
                    first_cell[1].squeeze(1).to(self.device),
                    first_cell[2].squeeze(1).to(self.device),
                    second_cell[0].squeeze(1).to(self.device),
                    second_cell[1].squeeze(1).to(self.device),
                    second_cell[2].squeeze(1).to(self.device),
                )

                output += 0.5
                order = output.to(dtype=torch.int32).cpu()
                score += sum(order == correct_order).sum() / correct_order.shape[0]

        score /= n_batches

        return score

    def _save_checkpoint(self, epoch, train_loss):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": train_loss,
        }
        checkpoint_path = f"{self.savedir}checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}.")
