
import torch.nn as nn
import torch.nn.functional as F
from layer import GCN_layer,GHNN_layer

class GCN_Net(nn.Module):
    """
    定义一个包含两层GraphConvolution的模型
    """
    def __init__(self, input_dim=3703):
        super(GCN_Net, self).__init__()
        self.gcn1 = GCN_layer(input_dim, 16)
        self.gcn2 = GCN_layer(16, 7)

    def forward(self, Laplacian, feature):
        output = F.relu(self.gcn1(Laplacian, feature))
        logits = self.gcn2(Laplacian, output)
        return logits

class GHNN_Net(nn.Module):
    def __init__(self, input_dim=3703):
        super(GHNN_Net, self).__init__()
        self.ghnn1 = GHNN_layer(input_dim, 32)
        self.ghnn2 = GHNN_layer(32, 7)

    def forward(self, sparse_poly, input_feature):
        output = F.relu(self.ghnn1(sparse_poly, input_feature))
        logits = self.ghnn2(sparse_poly, output)
        return logits

class GraphHeat_Net(nn.Module):
    def __init__(self, input_dim=3703):    #2, 1433
        super(GraphHeat_Net, self).__init__()
        self.graphheat1 = GraphHeat_layer(input_dim, 16)
        self.graphheat2 = GraphHeat_layer(16, 7)

    def forward(self, input_feature, heat):
        output = F.relu(self.graphheat1(input_feature, heat))
        logits = self.graphheat2(output, heat)
        return logits
