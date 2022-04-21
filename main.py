import time
import yaml
import torch
import argparse

import scipy.sparse as sp
import numpy as np
import seaborn as sns
import torch.nn.functional as F

from tqdm import tqdm
from torch.optim import Adam, Optimizer
from collections import defaultdict

from torch_geometric.data import Data, InMemoryDataset

from data import get_dataset, HeatDataset, PPRDataset, set_train_val_test_split
from models import GHNN, GCN, GAT, GraphSAGE, FAGCN, GAT_NET, ARMA, SGC, ChebyNet
from seeds import val_seeds, test_seeds




parser = argparse.ArgumentParser(description = "Graph Hilbert Neural Network")

#parser.add_argument('--path', type=str, default = 'data')
parser.add_argument('--device', type=str, default = 'cuda')
parser.add_argument('--dataset', type=str, default = 'Cora')
parser.add_argument('--model', type=str, default='GCN')
parser.add_argument('--preprocessing', type=str, default='none')

args = parser.parse_args()

device = args.device


with open('config.yaml', 'r') as c:
    config = yaml.safe_load(c)

config['dataset_name'] = args.dataset


datasets = {}

preprocessing=args.preprocessing

if preprocessing == 'none':
    dataset = get_dataset(       #func
        name=config['dataset_name'],
        use_lcc=config['use_lcc']
    )
    dataset.data = dataset.data.to(device)
    datasets[preprocessing] = dataset
    #print(dataset.data.train_mask)

elif preprocessing == 'heat':
    dataset = HeatDataset(       #func
        name=config['dataset_name'],
        use_lcc=config['use_lcc'],
        t=config[preprocessing]['t'],
        k=config[preprocessing]['k'],
        eps=config[preprocessing]['eps']
    )
    dataset.data = dataset.data.to(device)
    datasets[preprocessing] = dataset
elif preprocessing == 'ppr':
    dataset = PPRDataset(        #func`
        name=config['dataset_name'],
        use_lcc=config['use_lcc'],
        alpha=config[preprocessing]['alpha'],
        k=config[preprocessing]['k'],
        eps=config[preprocessing]['eps']
    )
    dataset.data = dataset.data.to(device)
    datasets[preprocessing] = dataset



print("===========Data has been downloaded============")



models = {}

if args.model == 'GraphSAGE':
    for preprocessing, dataset in datasets.items():
        models[preprocessing] = GraphSAGE(
            dataset,
            hidden = config[preprocessing]['hidden_layers'] * [config[preprocessing]['hidden_units']],
            dropout = config[preprocessing]['dropout']
        ).to(device)

elif args.model == 'GCN':
    for preprocessing, dataset in datasets.items():
        models[preprocessing] = GCN(
            dataset,
            hidden = config[preprocessing]['hidden_layers'] * [config[preprocessing]['hidden_units']],
            dropout = config[preprocessing]['dropout']
        )

elif args.model == 'FAGCN':
    for preprocessing, dataset in datasets.items():
        models[preprocessing] = FAGCN(
            dataset,
            dropout = config[preprocessing]['dropout']
        ).to(device)

elif args.model == 'GAT':
	for preprocessing, dataset in datasets.items():
		models[preprocessing] = GAT(
			dataset,
			hidden = config[preprocessing]['hidden_layers'] * [config[preprocessing]['hidden_units']],
			dropout = config[preprocessing]['dropout']
		).to(device)

elif args.model == 'GAT_NET':
    for preprocessing, dataset in datasets.items():
        models[preprocessing] = GAT_NET(
            dataset,
            dropout = config[preprocessing]['dropout']
        ).to(device)

elif args.model == 'GHNN':
    for preprocessing, dataset in datasets.items():
        models[preprocessing] = GHNN(
            dataset,
            dropout = config[preprocessing]['dropout']
        ).to(device)

elif args.model == 'ARMA':
    for preprocessing, dataset in datasets.items():
        models[preprocessing] = ARMA(
            dataset,
            dropout = config[preprocessing]['dropout']
        ).to(device)

elif args.model == 'SGC':
    for preprocessing, dataset in datasets.items():
        models[preprocessing] = SGC(
            dataset,
            dropout = config[preprocessing]['dropout']
        ).to(device)

elif args.model == 'ChebyNet':
    for preprocessing, dataset in datasets.items():
        models[preprocessing] = ChebyNet(
            dataset,
            dropout = config[preprocessing]['dropout']
        ).to(device)



def train(model: torch.nn.Module, optimizer: Optimizer, data: Data):
    model.train()
    optimizer.zero_grad()
    logits = model(data)
    loss = F.nll_loss(logits[data.train_mask], data.y[data.train_mask])
    loss.requires_grad_(True)
    loss.backward()
    optimizer.step()


def evaluate(model: torch.nn.Module, data: Data, test: bool):
    model.eval()
    with torch.no_grad():
        logits = model(data)
    eval_dict = {}
    keys = ['val', 'test'] if test else ['val']
    for key in keys:
        mask = data[f'{key}_mask']
        # loss = F.nll_loss(logits[mask], data.y[mask]).item()
        # eval_dict[f'{key}_loss'] = loss
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        eval_dict[f'{key}_acc'] = acc
    return eval_dict


def run(dataset: InMemoryDataset,
        model: torch.nn.Module,
        seeds: np.ndarray,
        test: bool = False,
        max_epochs: int = 10000,
        patience: int = 100,
        lr: float = 0.01,
        weight_decay: float = 0.01,
        num_development: int = 1500,
        device: str = 'cuda'):
    start_time = time.perf_counter()

    best_dict = defaultdict(list)

    cnt = 0
    for seed in tqdm(seeds):
        dataset.data = set_train_val_test_split(
            seed,
            dataset.data,
            num_development=num_development,
        ).to(device)
        model.to(device).reset_parameters()
        optimizer = Adam(
            [
                {'params': model.non_reg_params, 'weight_decay': 0},
                {'params': model.reg_params, 'weight_decay': weight_decay}
            ],
            lr=lr
        )

        patience_counter = 0
        tmp_dict = {'val_acc': 0}

        for epoch in range(1, max_epochs + 1):
            if patience_counter == patience:
                break

            train(model, optimizer, dataset.data)
            eval_dict = evaluate(model, dataset.data, test)

            if eval_dict['val_acc'] < tmp_dict['val_acc']:
                patience_counter += 1
            else:
                patience_counter = 0
                tmp_dict['epoch'] = epoch
                for k, v in eval_dict.items():
                    tmp_dict[k] = v

        for k, v in tmp_dict.items():
            best_dict[k].append(v)
            
    best_dict['duration'] = time.perf_counter() - start_time
    return dict(best_dict)


results = {}


results[preprocessing] = run(
    datasets[preprocessing],
    models[preprocessing],
    seeds=test_seeds if config['test'] else val_seeds,
    lr=config[preprocessing]['lr'],
    weight_decay=config[preprocessing]['weight_decay'],
    test=config['test'],
    num_development=config['num_development'],
    device=device
)


for _, best_dict in results.items():
    boots_series = sns.algorithms.bootstrap(best_dict['val_acc'], func=np.mean, n_boot=1000)
    best_dict['val_acc_ci'] = np.max(np.abs(sns.utils.ci(boots_series, 95) - np.mean(best_dict['val_acc'])))
    if 'test_acc' in best_dict:
        boots_series = sns.algorithms.bootstrap(best_dict['test_acc'], func=np.mean, n_boot=1000)
        best_dict['test_acc_ci'] = np.max(
            np.abs(sns.utils.ci(boots_series, 95) - np.mean(best_dict['test_acc']))
        )

    for k, v in best_dict.items():
        if 'acc_ci' not in k and k != 'duration':
            best_dict[k] = np.mean(best_dict[k])



mean_acc = results[preprocessing]['test_acc']
uncertainty = results[preprocessing]['test_acc_ci']
print(f"{preprocessing}: Mean accuracy: {100 * mean_acc:.2f} +- {100 * uncertainty:.2f}%")




 
