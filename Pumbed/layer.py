import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import scipy.sparse as sp


class GCN_layer(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):

        super(GCN_layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, Laplacian, input_feature):
        """
            adjacency: torch.sparse.FloatTensor
            input_feature: torch.Tensor
        """
        support = torch.mm(input_feature, self.weight)
        output = torch.sparse.mm(Laplacian, support)
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' ('             + str(self.in_features) + ' -> '             + str(self.out_features) + ')'




class GHNN_layer(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):

        super(GHNN_layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, sparse_poly, input_feature):
        support = torch.mm(input_feature, self.weight)   #2708*1433  1433*16 = 2708*16
        output = torch.sparse.mm(sparse_poly, support)   # 2708*2708  2708*16 =2708*16

        if self.use_bias:
            output += self.bias
        return output


'''
class GraphHeat_layer(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GraphHeat_layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.identity_array = torch.FloatTensor(np.eye(2708))

        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))

        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, input_feature, heatkernel):

        poly_item1 = 1 * self.identity_array
        print(heatkernel[0][0])
        poly_item2 = 1 * heatkernel[0]
        poly = poly_item1 + torch.tensor(poly_item2)  #2708*2708

        support = torch.mm(poly, input_feature)
        output = torch.mm(support,self.weight)

        if self.use_bias:
            output += self.bias
        return output
'''


