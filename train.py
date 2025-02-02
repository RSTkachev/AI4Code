import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import json
import numpy as np
from preprocess import prepare_data, load_data
from model import SiameseNetwork
from metrics import kendall_tau
from functools import cmp_to_key

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(train_data, valid_data, model, optimizer, criterion, n_epochs, path, saving_freq=10):
    model.to(device)

    best_score = -np.inf
    best = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0
        n_train = 0

        for index in tqdm(train_data.index):
            train_filename = train_data.loc[index, "id"]
            correct_order = train_data.loc[index, "cell_order"]
            random_order = correct_order.copy()
            np.random.shuffle(random_order)

            with open(f"{path}{train_filename}.json", "r") as file:
                json_code = json.load(file)

            for first, second in zip(random_order[:-1], random_order[1:]):
                equality = 0 if correct_order.index(first) < correct_order.index(second) else 1

                input_ids1, att_mask1, cell_type1 = prepare_data(json_code["cell_type"][first], json_code["source"][first])
                input_ids2, att_mask2, cell_type2 = prepare_data(json_code["cell_type"][second], json_code["source"][second])

                input_ids1, att_mask1, cell_type1 = input_ids1.to(device), att_mask1.to(device), cell_type1.to(device)
                input_ids2, att_mask2, cell_type2 = input_ids2.to(device), att_mask2.to(device), cell_type2.to(device)

                optimizer.zero_grad()
                output = model(input_ids1, att_mask1, cell_type1, input_ids2, att_mask2, cell_type2)
                loss = criterion(output, torch.tensor([[equality]], dtype=torch.float32).to(device))
                train_loss += loss.item()
                loss.backward()
                optimizer.step()

                n_train += 1

        train_loss /= n_train

        model.eval()
        true_order, predicted_order = [], []
        with torch.no_grad():
            for index in tqdm(valid_data.index):
                valid_filename = valid_data.loc[index, "id"]
                correct_order = valid_data.loc[index, "cell_order"]
                random_order = correct_order.copy()
                np.random.shuffle(random_order)

                with open(f"{path}{valid_filename}.json", "r") as file:
                    json_code = json.load(file)

                sorted_cells = sorted(random_order, key=cmp_to_key(lambda a, b: -1 if model(a, b) <= 0.5 else 1))
                sorted_order = sorted_cells

                true_order.append(correct_order)
                predicted_order.append(sorted_order)

        kendall_score = kendall_tau(true_order, predicted_order)

        if kendall_score > best_score:
            best = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss
            }

        if epoch % saving_freq == 0:
            torch.save(best, 'models/best.pt')

        print(f"Epoch {epoch}/{n_epochs}, Train loss: {train_loss}, Valid Kendall Tau: {kendall_score}")

torch.save(best, 'models/best.pt')