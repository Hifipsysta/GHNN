#!/usr/bin/env python
# coding: utf-8



import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import dgl
import os.path as osp
from data import read_data,build_adjacency,normalization
from model import GCN_Net

import numpy as np
import torch
import torch.nn as nn

import torch.optim as optim
import matplotlib.pyplot as plt


# 超参数定义
learning_rate = 0.1
weight_decay = 5e-4
epochs = 200






# 模型定义：Model, Loss, Optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GCN_Net().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)



# 加载数据，并转换为torch.Tensor
dataset = dgl.data.PubmedGraphDataset()
graph = dataset[0]
tensor_x = graph.ndata['feat'].to(device)
tensor_y = graph.ndata['label'].to(device)
tensor_train_mask = graph.ndata['train_mask'].to(device)
tensor_val_mask = graph.ndata['val_mask'].to(device)
tensor_test_mask = graph.ndata['test_mask'].to(device)
graph = read_data(osp.join("pubmed", "raw", "ind.pubmed.graph"))
normalize_adjacency = normalization(build_adjacency(graph))


indices = torch.from_numpy(np.asarray([normalize_adjacency.row,
                                       normalize_adjacency.col]).astype('int64')).long()
values = torch.from_numpy(normalize_adjacency.data.astype(np.float32))
tensor_adjacency = torch.sparse.FloatTensor(indices, values,
                                            (19717, 19717)).to(device)





def train():
    loss_history = []
    val_acc_history = []
    model.train()
    train_y = tensor_y[tensor_train_mask]
    for epoch in range(epochs):
        logits = model(tensor_adjacency, tensor_x)  # 前向传播
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
        logits = model(tensor_adjacency, tensor_x)
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






