import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader

from AMPredictor import *
from utils import *
from emetrics import *

# Hyperparameters
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
LR = 0.001
NUM_EPOCHS = 100

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

models_dir = 'models'
results_dir = 'results'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

result_str = ''
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')
model = GNNPredictor()
model.to(device)
model_st = GNNPredictor.__name__

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train_data, valid_data = load_dataset()
train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate)

best_mse = 1000
best_test_mse = 1000
best_epoch = -1
model_file_name = 'models/model_' + model_st + '_' + '.model'

for epoch in range(NUM_EPOCHS):
    train(model, device, train_loader, optimizer, epoch + 1, TRAIN_BATCH_SIZE)
    print('predicting for valid data')
    G, P = predicting(model, device, valid_loader)
    val = get_mse(G, P)
    print('valid result:', val, best_mse)
    if val < best_mse:
        best_mse = val
        best_epoch = epoch + 1
        torch.save(model.state_dict(), model_file_name)
        print('rmse improved at epoch ', best_epoch, '; best_test_mse', best_mse, model_st)
    else:
        print('No improvement since epoch ', best_epoch, '; best_test_mse', best_mse, model_st)