import torch
import torch.optim as optim
import torch.nn as nn
from preprocess import load_data
from model import SiameseNetwork
from train import train

train_data, valid_data, test_data = load_data('./data/train_orders.csv')

model = SiameseNetwork(128)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCELoss()

train(train_data, valid_data, model, optimizer, criterion, n_epochs=1, path="./data/")