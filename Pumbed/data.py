

import itertools
import os
import os.path as osp
import pickle
import urllib
import numpy as np
import scipy.sparse as sp
from collections import namedtuple

def read_data(path):
    """使用不同的方式读取原始数据以进一步处理"""
    name = osp.basename(path)
    if name == "ind.cora.test.index":
        out = np.genfromtxt(path, dtype="int64")
        return out
    else:
        out = pickle.load(open(path, "rb"), encoding="latin1")
        out = out.toarray() if hasattr(out, "toarray") else out
        return out


def build_adjacency(adj_dict):
    """根据邻接表创建邻接矩阵"""
    edge_index = []
    num_nodes = len(adj_dict)
    for src, dst in adj_dict.items():
        edge_index.extend([src, v] for v in dst)
        edge_index.extend([v, src] for v in dst)
    # 去除重复的边
    edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))

    edge_index = np.asarray(edge_index)
    adjacency = sp.coo_matrix((np.ones(len(edge_index)),
                               (edge_index[:, 0], edge_index[:, 1])),
                              shape=(num_nodes, num_nodes), dtype="float32")
    return adjacency

def normalization(adjacency):
    """计算 L=D^-0.5 * (A+I) * D^-0.5"""
    adjacency += sp.eye(adjacency.shape[0])  # 增加自连接
    degree = np.array(adjacency.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    return d_hat.dot(adjacency).dot(d_hat).tocoo()