#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#基于Cora数据集的GCN节点分类" data-toc-modified-id="基于Cora数据集的GCN节点分类-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>基于Cora数据集的GCN节点分类</a></span><ul class="toc-item"><li><span><a href="#数据准备" data-toc-modified-id="数据准备-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>数据准备</a></span></li><li><span><a href="#图卷积层定义" data-toc-modified-id="图卷积层定义-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>图卷积层定义</a></span></li><li><span><a href="#模型定义" data-toc-modified-id="模型定义-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>模型定义</a></span></li><li><span><a href="#模型训练" data-toc-modified-id="模型训练-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>模型训练</a></span></li></ul></li></ul></div>

# # 基于Cora数据集的GCN节点分类

# In[1]:
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



# 超参数定义
learning_rate = 0.1
weight_decay = 5e-4
epochs = 200


# 模型定义：Model, Loss, Optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GHNN_Net().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)



# 加载数据，并转换为torch.Tensor
dataset = CoraData().data
x = dataset.x / dataset.x.sum(1, keepdims=True)  # 归一化数据，使得每一行和为1
tensor_x = torch.from_numpy(x).to(device)
tensor_y = torch.from_numpy(dataset.y).to(device)
tensor_train_mask = torch.from_numpy(dataset.train_mask).to(device)
tensor_val_mask = torch.from_numpy(dataset.val_mask).to(device)
tensor_test_mask = torch.from_numpy(dataset.test_mask).to(device)
normalized_Laplacian = CoraData.normalization(dataset.adjacency)   # 规范化邻接矩阵
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
poly_item2 = 1.05 * Laplacian_tensor_
print(type(Laplacian_tensor_.to_dense()))
inx3,val3 = spspmm(indices,values, indices,values,2708,2708,2708)
poly_item3 = torch.sparse.FloatTensor(inx3, val3,(2708, 2708)).to(device)
inx4,val4 = spspmm(inx3, val3, indices, values, 2708, 2708, 2708)
poly_item4 = torch.sparse.FloatTensor(inx4, val4,(2708, 2708)).to(device)
sparse_poly = 0*poly_item1 +  poly_item2 +  0* poly_item3 + 0 * poly_item4   # 2708*2708

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


# In[13]:


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


# In[14]:
plot_loss_with_acc(loss, val_acc)






