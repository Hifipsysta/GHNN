#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#基于Cora数据集的GCN节点分类" data-toc-modified-id="基于Cora数据集的GCN节点分类-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>基于Cora数据集的GCN节点分类</a></span><ul class="toc-item"><li><span><a href="#数据准备" data-toc-modified-id="数据准备-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>数据准备</a></span></li><li><span><a href="#图卷积层定义" data-toc-modified-id="图卷积层定义-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>图卷积层定义</a></span></li><li><span><a href="#模型定义" data-toc-modified-id="模型定义-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>模型定义</a></span></li><li><span><a href="#模型训练" data-toc-modified-id="模型训练-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>模型训练</a></span></li></ul></li></ul></div>

# # 基于Cora数据集的GCN节点分类

# In[1]:
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


from data import CoraData
from model import GraphHeat_Net

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

import torch.optim as optim
import matplotlib.pyplot as plt

from utils import chebyshev_polynomials







# 超参数定义
learning_rate = 0.1
weight_decay = 5e-4
epochs = 200


# 模型定义：Model, Loss, Optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GraphHeat_Net().to(device)
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
normalize_adjacency = CoraData.normalization(dataset.adjacency)   # 规范化邻接矩阵
indices = torch.from_numpy(np.asarray([normalize_adjacency.row, 
                                       normalize_adjacency.col]).astype('int64')).long()
values = torch.from_numpy(normalize_adjacency.data.astype(np.float32))
tensor_adjacency = torch.sparse.FloatTensor(indices, values, 
                                            (2708, 2708)).to(device)


heatkernel = chebyshev_polynomials(dataset.adjacency,3)




# In[8]:
# 训练主体函数
def train():
    loss_history = []
    val_acc_history = []
    model.train()
    train_y = tensor_y[tensor_train_mask]
    for epoch in range(epochs):
        logits = model(tensor_x, heatkernel)  # 前向传播
        train_mask_logits = logits[tensor_train_mask]   # 只选择训练节点进行监督
        loss = criterion(train_mask_logits, train_y)    # 计算损失值
        optimizer.zero_grad()
        loss.backward()     # 反向传播计算参数的梯度
        optimizer.step()    # 使用优化方法进行梯度更新
        train_acc, _, _ = test(tensor_train_mask)     # 计算当前模型训练集上的准确率
        val_acc, _, _ = test(tensor_val_mask)     # 计算当前模型在验证集上的准确率
        # 记录训练过程中损失值和准确率的变化，用于画图
        loss_history.append(loss.item())
        val_acc_history.append(val_acc.item())
        print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}, ValAcc {:.4f}".format(
            epoch, loss.item(), train_acc.item(), val_acc.item()))
    return loss_history, val_acc_history

# 测试函数
def test(mask):
    model.eval()
    with torch.no_grad():
        logits = model(tensor_x, heatkernel)
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






