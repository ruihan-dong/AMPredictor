import matplotlib
matplotlib.use("Agg")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import matplotlib.pyplot as plt

from AMPredictor import *
from emetrics import get_cindex, get_rm2, get_ci, get_mse, get_rmse, get_pearson, get_spearman
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

def calculate_metrics(Y, P):
    cindex = get_cindex(Y, P)
    cindex2 = get_ci(Y, P)
    rm2 = get_rm2(Y, P) 
    mse = get_mse(Y, P)
    pearson = get_pearson(Y, P)
    spearman = get_spearman(Y, P)
    rmse = get_rmse(Y, P)

    print('cindex:', cindex)
    print('cindex2', cindex2)
    print('rm2:', rm2)
    print('mse:', mse)
    print('pearson', pearson)

    result_file_name = 'results/result_' + model_st + '.txt'
    result_str = ''
    result_str += '\r\n'
    result_str += 'rmse:' + str(rmse) + ' ' + ' mse:' + str(mse) + ' ' + ' pearson:' + str(
        pearson) + ' ' + 'spearman:' + str(spearman) + ' ' + 'ci:' + str(cindex) + ' ' + 'rm2:' + str(rm2)
    print(result_str)
    open(result_file_name, 'w').writelines(result_str)


def plot_density(Y, P):
    plt.figure(figsize=(5, 5))
    plt.grid(linestyle='--')
    ax = plt.gca()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    plt.scatter(P, Y, color='blue', s=10)
    # plt.title('density', fontsize=15, fontweight='bold')
    plt.xlabel('predicted', fontsize=10, fontweight='bold')
    plt.ylabel('measured', fontsize=10, fontweight='bold')
    plt.plot([-1, 4], [-1, 4], color='black')
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=12, fontweight='bold')
    plt.savefig(os.path.join('results.png'), dpi=75, bbox_inches='tight')


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
    calculate_metrics(Y, P)
    plot_density(Y, P)
