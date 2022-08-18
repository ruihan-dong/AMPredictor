import matplotlib
matplotlib.use("Agg")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import numpy as np

from AMPredictor import *
from utils import *

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            # data = data.to(device)
            output = model(data_mol, data_pro)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

def load_model(model_path):
    model = torch.load(model_path)
    return model

if __name__ == '__main__':
    model_st = GNNPredictor.__name__

    TEST_BATCH_SIZE = 128
    models_dir = 'models'
    results_dir = 'results'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_file_name = 'models/model_' + model_st + '_.model'
    result_file_name = 'results/result_' + model_st + '.txt'

    model = GNNPredictor()
    model.to(device)
    model.load_state_dict(torch.load(model_file_name, map_location=device))
    test_data = load_dataset(test_flag=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate)

    Y, P = predicting(model, device, test_loader)
    print("Predicted logMIC values:", P)
    output = np.array([P]).transpose()
    np.savetxt('output.csv', output, delimiter=',')

