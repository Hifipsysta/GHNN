from typing import List

import torch
from torch.nn import ModuleList, Dropout, ReLU, init, Linear, functional
from torch_geometric.nn import GCNConv, GATConv, FAConv, SAGEConv, ARMAConv, SGConv, ChebConv, GINConv
from torch_geometric.data import Data, InMemoryDataset
from ghnn import GHNNConv




class GCN(torch.nn.Module):
    def __init__(self,
                 dataset: InMemoryDataset,
                 hidden: List[int] = [64],
                 dropout: float = 0.5):
        super(GCN, self).__init__()

        num_features = [dataset.data.x.shape[1]] + hidden + [dataset.num_classes]
        layers = []
        for in_features, out_features in zip(num_features[:-1], num_features[1:]):
            layers.append(GCNConv(in_features, out_features))
        self.layers = ModuleList(layers)

        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for i, layer in enumerate(self.layers):
            x = layer(x = x, edge_index = edge_index, edge_weight=edge_attr)

            if i == len(self.layers) - 1:
                break

            x = self.act_fn(x)
            x = self.dropout(x)

        return functional.log_softmax(x, dim=1)


class GraphSAGE(torch.nn.Module):
    def __init__(self,
                 dataset: InMemoryDataset,
                 hidden: List[int] = [64],
                 dropout: float = 0.5):
        super(GraphSAGE, self).__init__()

        num_features = [dataset.data.x.shape[1]] + hidden + [dataset.num_classes]
        layers = []
        for in_features, out_features in zip(num_features[:-1], num_features[1:]):
            layers.append(SAGEConv(in_features, out_features))
        self.layers = ModuleList(layers)

        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for i, layer in enumerate(self.layers):
            x = layer(x = x, edge_index = edge_index, size = None)

            if i == len(self.layers) - 1:
                break

            x = self.act_fn(x)
            x = self.dropout(x)

        return functional.log_softmax(x, dim=1)



class GAT(torch.nn.Module):
    def __init__(self,
                 dataset: InMemoryDataset,
                 hidden: List[int] = 64,
                 dropout: float = 0.5):
        super(GAT, self).__init__()

        num_features = [dataset.data.x.shape[1]] + hidden + [dataset.num_classes]
        layers = []
        for in_features, out_features in zip(num_features[:-1], num_features[1:]):
            #print(in_features, out_features)
            layers.append(GATConv(in_features, out_features))
        self.layers = ModuleList(layers)

        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for i, layer in enumerate(self.layers):
            x = layer(x = x, edge_index = edge_index, edge_attr = edge_attr)

            if i == len(self.layers) - 1:
                break

            x = self.act_fn(x)
            x = self.dropout(x)

        return functional.log_softmax(x, dim=1)



class GAT_NET(torch.nn.Module):
    def __init__(self, 
                 dataset: InMemoryDataset, 
                 hidden: int = 64, 
                 dropout: float = 0.5,
                 heads=8):
        super(GAT_NET, self).__init__()
        
        self.dropout = dropout
        self.layers = ModuleList()
        #print(dataset.num_features)
        self.layers.append(GATConv(dataset.num_features, 
                                   hidden, 
                                   heads=heads,
                                   dropout=dropout))
        self.layers.append(GATConv(hidden*heads, 
                                   dataset.num_classes, 
                                   heads=1, 
                                   concat=False,
                                   dropout=dropout)) 

        self.reg_params = list(self.layers[0].parameters())
        self.non_reg_params = list([p for l in self.layers[1:] for p in l.parameters()])

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = functional.dropout(x, p=self.dropout, training = self.training)
        x = functional.elu(self.layers[0](x, edge_index))
        x = functional.dropout(x, p=self.dropout, training = self.training)
        x = self.layers[1](x, edge_index)

        return functional.log_softmax(x, dim=1)



class ARMA(torch.nn.Module):
    def __init__(self, 
                 dataset: InMemoryDataset, 
                 hidden: int = 64, 
                 dropout: float = 0.5,
                 num_stacks: int = 1,
                 num_layers: int = 1):
        super(ARMA, self).__init__()

        self.dropout = dropout
        self.layers = ModuleList()
        self.layers.append(ARMAConv(dataset.num_features, 
                                    hidden, 
                                    num_stacks=num_stacks,
                                    num_layers=num_layers,
                                    shared_weights=False,
                                    dropout=dropout))
        self.layers.append(ARMAConv(hidden, 
                                    dataset.num_classes, 
                                    num_stacks=num_stacks, 
                                    num_layers=num_layers,
                                    shared_weights=False,
                                    dropout=dropout)) 

        self.reg_params = list(self.layers[0].parameters())
        self.non_reg_params = list([p for l in self.layers[1:] for p in l.parameters()])

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        #x = functional.dropout(x, p=self.dropout, training = self.training)
        x = functional.relu(self.layers[0](x, edge_index))
        x = functional.dropout(x, p=self.dropout, training = self.training)
        x = self.layers[1](x, edge_index)

        return functional.log_softmax(x, dim=1)



class GHNN(torch.nn.Module):
    def __init__(self, 
                 dataset: InMemoryDataset, 
                 hidden: int = 64, 
                 dropout: float = 0.5):
        super(GHNN, self).__init__()

        self.layers = ModuleList()
        self.layers.append(Linear(dataset.num_features, hidden))
        self.layers.append(Linear(hidden, dataset.num_classes))
        self.layers.append(GHNNConv(iterations=10, alpha=0.1))

        self.reg_params = list(self.layers[0].parameters())
        self.non_reg_params = list([p for l in self.layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.dropout(x)
        x = self.layers[0](x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.layers[1](x)
        x = self.layers[2](x = x, edge_index = edge_index) 

        return functional.log_softmax(x, dim=1)



class SGC(torch.nn.Module):
    def __init__(self, 
                 dataset: InMemoryDataset, 
                 hidden: int = 64, 
                 dropout: float = 0.5):
        super(SGC, self).__init__()

        self.layers = ModuleList()
        self.layers.append(SGConv(dataset.num_features, 
                                  dataset.num_classes, 
                                  K=2, 
                                  cached=True))

        self.reg_params = list(self.layers[0].parameters())
        self.non_reg_params = list([p for l in self.layers[1:] for p in l.parameters()])

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.layers[0](x = x, edge_index = edge_index) 

        return functional.log_softmax(x, dim=1)



class ChebyNet(torch.nn.Module):
    def __init__(self, 
                 dataset: InMemoryDataset, 
                 hidden: int = 64, 
                 dropout: float = 0.5,
                 num_hops: int = 3):
        super(ChebyNet, self).__init__()

        self.dropout = dropout
        self.layers = ModuleList()
        self.layers.append(ChebConv(dataset.num_features, 
                                    hidden, 
                                    K=num_hops))
        self.layers.append(ChebConv(hidden,
                                    dataset.num_classes,
                                    K=num_hops))

        self.reg_params = list(self.layers[0].parameters())
        self.non_reg_params = list([p for l in self.layers[1:] for p in l.parameters()])

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = functional.relu(self.layers[0](x, edge_index))
        x = functional.dropout(x, p=self.dropout,training=self.training)
        x = self.layers[1](x, edge_index)

        return functional.log_softmax(x, dim=1)


class GIN(torch.nn.Module):
    def __init__(self, 
                 dataset: InMemoryDataset, 
                 hidden: int = 64, 
                 dropout: float = 0.5,
                 num_hops: int = 3):
        super(GIN, self).__init__()

        self.layers = ModuleList()
        self.layers.append(GINConv(Seq, 
                                    hidden, 
                                    K=num_hops))
        self.layers.append(GINConv(hidden,
                                    dataset.num_classes,
                                    K=num_hops))

        self.reg_params = list(self.layers[0].parameters())
        self.non_reg_params = list([p for l in self.layers[1:] for p in l.parameters()])

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = functional.relu(self.layers[0](x, edge_index))
        x = functional.dropout(x, p=self.dropout,training=self.training)
        x = self.layers[1](x, edge_index)

        return functional.log_softmax(x, dim=1)





class FAGCN(torch.nn.Module):
    def __init__(self,
                 dataset: InMemoryDataset,
                 hidden: int = 64,
                 dropout: float = 0.5,
                 layer_num: int = 2):
        super(FAGCN, self).__init__()

        self.layer_num = layer_num
        self.layers = ModuleList()
        self.layers.append(Linear(dataset.num_features, hidden))
        for i in range(self.layer_num):
            self.layers.append(FAConv(hidden))
        self.layers.append(Linear(hidden, dataset.num_classes))

        self.reg_params = list(self.layers[0].parameters())
        self.non_reg_params = list([p for l in self.layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.dropout(x)
        x = self.layers[0](x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x_0 = x

        for i in range(self.layer_num):
            x = self.layers[i+1](x = x, x_0 = x_0, edge_index = edge_index, edge_weight=edge_attr) 

        x = self.layers[-1](x)
        #x = self.act_fn(x)
        #x = self.dropout(x)

        return functional.log_softmax(x, dim=1)
