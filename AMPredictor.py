# Antimicrobial Activity (MIC) Predictor
# Input: peptide sequence; Output: MIC regression value
# 22-01-12, 22-02-22

import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from rdkit.Chem import AllChem as Chem

from utils import *

# peptide graph encoding
# to judge whether the required files exist
def valid_peptide(name):
    contact_dir = './data/contact_65/'
    emb_dir = './data/embedding_65/'
    contact_file = os.path.join(contact_dir, name + '.npy')
    emb_file = os.path.join(emb_dir, name + '.pt')
    if os.path.exists(contact_file) and os.path.exists(emb_file):
        return True
    else:
        return False

# node feature from esm embedding
def peptide_to_feature(peptide_name):
    embedding_path = './data/embedding_65/'
    embedding_file = os.path.join(embedding_path, peptide_name + '.pt')
    embedding = torch.load(embedding_file)
    feature = embedding['feature']
    size = embedding['size']
    # feature = torch.mean(feature, dim=1, keepdim=False) # 有无在此处压缩的必要？
    # print("peptide feature: ", feature.shape)
    return feature, size

# with contact as edge, convert a peptide to a graph
def peptide_to_graph(peptide_name):
    edge_index = []
    contact_path = './data/contact_65/'
    contact_file = os.path.join(contact_path, peptide_name + '.npy')
    contact_map = np.load(contact_file)
    contact_map += np.matrix(np.eye(contact_map.shape[0]))
    index_row, index_col = np.where(contact_map >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
    node_feature, peptide_size = peptide_to_feature(peptide_name)
    edge_index = np.array(edge_index)
    return peptide_size, node_feature, edge_index

# RDkit encoding amino acid fingerprint
# input: sequences
def fingerprint_feature(seqs):
    seq_max = 65
    fp_list = []
    for seq in seqs:
        smiles_file = './data/smiles_file.json'
        with open(smiles_file) as json_file:
            smiles = json.load(json_file)
        lookupfps = {}

        for key, value in smiles.items():
            mol = Chem.MolFromSmiles(value)
            fp = np.array(Chem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)) # 需要参考指定此二参数
            lookupfps[key] = fp
        lookupfps[' '] = np.zeros(lookupfps['A'].shape, dtype=int)
        lookupfps['0'] = np.zeros(lookupfps['A'].shape, dtype=int)

        fp = np.asarray([lookupfps[seq[i]] for i in range(len(seq))])
        n_rows = seq_max - len(seq)
        shape_padding = (n_rows, 2048) # nbits from Morgan fp
        padding_array = np.zeros(shape_padding)
        fingerprint = np.concatenate((fp, padding_array), axis = 0)
        fingerprint = np.mean(fingerprint, axis=1) # pool to 1d vector
        fp_list.append(fingerprint)
    return np.array(fp_list)

# GCN layer + full-connected regressor model
class GNNPredictor(torch.nn.Module):
    def __init__(self, n_output=1, num_features_pro=1280, output_dim=128, dropout=0.1):
        super(GNNPredictor, self).__init__()
        self.n_output = n_output

        self.pro_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro_conv2 = GCNConv(num_features_pro, 640)
        self.pro_conv3 = GCNConv(640, 320)

        self.pro_fc_g1 = torch.nn.Linear(320, 1024)
        self.pro_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(193, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data_pro, data_fp):
        # get fingerprint input
        # fp_x, fp_batch = data_fp.x, data_fp.batch
        fp_x = data_fp
        # get protein input
        target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch

        # xf = global_mean_pool(fp_x, fp_batch)  # global pooling for fp
        xf = fp_x.float()

        xt = self.pro_conv1(target_x, target_edge_index)
        xt = self.relu(xt)

        xt = self.pro_conv2(xt, target_edge_index)
        xt = self.relu(xt)

        xt = self.pro_conv3(xt, target_edge_index)
        xt = self.relu(xt)
        xt = global_mean_pool(xt, target_batch)  # global pooling

        # flatten
        xt = self.relu(self.pro_fc_g1(xt))
        xt = self.dropout(xt)
        xt = self.pro_fc_g2(xt)
        xt = self.dropout(xt)

        # concat
        xc = torch.cat((xf, xt), 1)
        # fully connected layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out

# data.csv: No./seq/MIC
def read_dataset(path):
    try:
        file = open(path, "r")
    except:
        print('Data path Not Found')
    X_seq = []
    target_keys = []
    y = []
    for line in file:
        values = line.split()
        target_keys.append(values[0])
        X_seq.append(values[1])
        y.append(float(values[2]))
    file.close()
    return np.array(target_keys), X_seq, np.array(y)

def load_dataset(test_flag=False):
    # create target graph
    def create_graph(target_key):
        target_graph = {}
        for key in target_key:
            if not valid_peptide(key):
                print('cannot find peptide files of' + key)
                continue
            graph = peptide_to_graph(key)
            target_graph[key] = graph
        return target_graph

    if test_flag == False:
        # train dataset
        train_prot_keys, train_prot_seqs, train_y = read_dataset('./data/train.txt')
        train_graph = create_graph(train_prot_keys)
        train_fp = fingerprint_feature(train_prot_seqs)
        train_dataset = DTADataset(root='data', dataset='train', target_key=train_prot_keys,
                                y=train_y, target_graph=train_graph, fingerprint=train_fp)
        
        # valid dataset
        valid_prot_keys, valid_prot_seqs, valid_y = read_dataset('./data/valid.txt')
        valid_graph = create_graph(valid_prot_keys)
        valid_fp = fingerprint_feature(valid_prot_seqs)
        valid_dataset = DTADataset(root='data', dataset='valid', target_key=valid_prot_keys,
                                y=valid_y, target_graph=valid_graph, fingerprint=valid_fp)
        return train_dataset, valid_dataset
    else:
        # test dataset
        test_prot_keys, test_prot_seqs, test_y = read_dataset('./data/test.txt')
        test_graph = create_graph(test_prot_keys)
        test_fp = fingerprint_feature(test_prot_seqs)
        test_dataset = DTADataset(root='data', dataset='test', target_key=test_prot_keys,
                                y=test_y, target_graph=test_graph, fingerprint=test_fp)
        return test_dataset
