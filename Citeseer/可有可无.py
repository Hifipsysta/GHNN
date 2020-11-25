
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.sparse as sp



e=torch.tensor([[1,2,3],[4,5,6],[7,8,9]])


identity_array = [1 for i in range(5)]
print(identity_array)
identity_matrix = sp.spdiags(identity_array, diags=[0], m=5, n=5, format="coo")
print(identity_matrix.row)


from torch_sparse import spspmm, spmm
help(spspmm)