# coding: utf-8

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


from data import CoraData
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


device = "cuda" if torch.cuda.is_available() else "cpu"
model = GHNN_Net().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


dataset = CoraData().data
x = dataset.x / dataset.x.sum(1, keepdims=True) 
tensor_x = torch.from_numpy(x).to(device)
tensor_y = torch.from_numpy(dataset.y).to(device)
tensor_train_mask = torch.from_numpy(dataset.train_mask).to(device)
tensor_val_mask = torch.from_numpy(dataset.val_mask).to(device)
tensor_test_mask = torch.from_numpy(dataset.test_mask).to(device)
normalized_Laplacian = CoraData.normalization(dataset.adjacency)  
indices = torch.from_numpy(np.asarray([normalized_Laplacian.row,
                                       normalized_Laplacian.col]).astype('int64')).long()
values = torch.from_numpy(normalized_Laplacian.data.astype(np.float32))
Laplacian_tensor_ = torch.sparse.FloatTensor(indices, values,
                                            (2708, 2708)).to(device)

identity_list_ = [1 for i in range(2708)]
identity_coo_ = sp.spdiags(identity_list_, diags=[0], m=2708, n=2708, format="coo")
indices = torch.from_numpy(np.asarray([identity_coo_.row,
                                       identity_coo_.col]).astype('int64')).long()
values = torch.from_numpy(identity_coo_.data.astype(np.float32))
identity_tensor_ = torch.sparse.FloatTensor(indices, values,
                                            (2708, 2708)).to(device)
poly_item1 = identity_tensor_
poly_item2 = 1.08 * Laplacian_tensor_
print(type(Laplacian_tensor_.to_dense()))
inx3,val3 = spspmm(indices,values, indices,values,2708,2708,2708)
poly_item3 = torch.sparse.FloatTensor(inx3, val3,(2708, 2708)).to(device)
inx4,val4 = spspmm(inx3, val3, indices, values, 2708, 2708, 2708)
poly_item4 = torch.sparse.FloatTensor(inx4, val4,(2708, 2708)).to(device)
sparse_poly = 0*poly_item1 +  poly_item2 +  0* poly_item3 + 0 * poly_item4   

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


def plot_loss_with_acc(loss_history, val_acc_history):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(range(len(loss_history)), loss_history,
             c=np.array([255, 71, 90]) / 255.)
    plt.ylabel('Loss')
    
    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    ax2.plot(range(len(val_acc_history)), val_acc_history,
             c=np.array([79, 179, 255]) / 255.)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    plt.ylabel('ValAcc')
    
    plt.xlabel('Epoch')
    plt.title('Training Loss & Validation Accuracy')
    plt.show()


loss, val_acc = train()
test_acc, test_logits, test_label = test(tensor_test_mask)
print("Test accuarcy: ", test_acc.item())


plot_loss_with_acc(loss, val_acc)






