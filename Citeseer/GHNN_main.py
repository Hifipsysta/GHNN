
# coding: utf-8


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import dgl
import os.path as osp
from data import read_data,build_adjacency,normalization
from model import GHNN_Net

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

import torch.optim as optim
from torch_sparse import spspmm, spmm
import matplotlib.pyplot as plt




learning_rate = 0.1
weight_decay = 5e-4
epochs = 200
ncount=3327



device = "cuda" if torch.cuda.is_available() else "cpu"
model = GHNN_Net().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)





dataset = dgl.data.CiteseerGraphDataset()
graph = dataset[0]
tensor_x = graph.ndata['feat'].to(device)
tensor_y = graph.ndata['label'].to(device)
tensor_train_mask = graph.ndata['train_mask'].to(device)
tensor_val_mask = graph.ndata['val_mask'].to(device)
tensor_test_mask = graph.ndata['test_mask'].to(device)
graph = read_data(osp.join("citeseer", "raw", "ind.citeseer.graph"))
normalized_Laplacian = normalization(build_adjacency(graph))

indices = torch.from_numpy(np.asarray([normalized_Laplacian.row,
                                       normalized_Laplacian.col]).astype('int64')).long()
values = torch.from_numpy(normalized_Laplacian.data.astype(np.float32))
Laplacian_tensor_ = torch.sparse.FloatTensor(indices, values,
                                            (ncount, ncount)).to(device)

identity_list_ = [1 for i in range(ncount)]
identity_coo_ = sp.spdiags(identity_list_, diags=[0], m=ncount, n=ncount, format="coo")
indices = torch.from_numpy(np.asarray([identity_coo_.row,
                                       identity_coo_.col]).astype('int64')).long()
values = torch.from_numpy(identity_coo_.data.astype(np.float32))
identity_tensor_ = torch.sparse.FloatTensor(indices, values,
                                            (ncount, ncount)).to(device)
poly_item1 = 1 * identity_tensor_
poly_item2 = 1 * Laplacian_tensor_
print(type(Laplacian_tensor_.to_dense()))
inx3,val3 = spspmm(indices,values, indices,values,ncount,ncount,ncount)
poly_item3 = torch.sparse.FloatTensor(inx3, val3,(ncount, ncount)).to(device)
inx4,val4 = spspmm(inx3, val3, indices, values, ncount, ncount, ncount)
poly_item4 = torch.sparse.FloatTensor(inx4, val4,(ncount, ncount)).to(device)
sparse_poly = 0*poly_item1 + 1.08* poly_item2 + 0 * poly_item3 + 0 * poly_item4


def train():
    loss_history = []
    val_acc_history = []
    model.train()
    train_y = tensor_y[tensor_train_mask]
    for epoch in range(epochs):
        logits = model(sparse_poly, tensor_x)
        train_mask_logits = logits[tensor_train_mask]
        loss = criterion(train_mask_logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc, _, _ = test(tensor_train_mask)
        val_acc, _, _ = test(tensor_val_mask)

        loss_history.append(loss.item())
        val_acc_history.append(val_acc.item())
        print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}, ValAcc {:.4f}".format(
            epoch+1, loss.item(), train_acc.item(), val_acc.item()))

    return loss_history, val_acc_history


def test(mask):
    model.eval()
    with torch.no_grad():
        logits = model(sparse_poly, tensor_x)
        test_mask_logits = logits[mask]
        predict_y = test_mask_logits.max(1)[1]
        accuarcy = torch.eq(predict_y, tensor_y[mask]).float().mean()
    return accuarcy, test_mask_logits.cpu().numpy(), tensor_y[mask].cpu().numpy()




loss, val_acc = train()
test_acc, test_logits, test_label = test(tensor_test_mask)
print("Test accuarcy: ", test_acc.item())









