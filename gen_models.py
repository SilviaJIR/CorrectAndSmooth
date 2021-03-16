import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm


from copy import deepcopy
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import numpy as np


from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from outcome_correlation import prepare_folder
from diffusion_feature import preprocess
import glob
import os
import shutil

from logger import Logger

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, relu_first = True):
        super(MLP, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.relu_first = relu_first

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):    
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.relu_first:
                x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            if not self.relu_first:
                x = F.relu(x, inplace=True)


            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return F.log_softmax(x, dim=-1)



class MLPLinear(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLPLinear, self).__init__()
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x):
        return F.log_softmax(self.lin(x), dim=-1)


class SGC(torch.nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()

        self.W = torch.nn.Linear(nfeat, nclass)

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, x):
        return self.W(x)


class Breadth(torch.nn.Module):
    def __init__(self, in_dim, out_dim, headnum):
        super(Breadth, self).__init__()
        self.gatconv = GATConv(in_dim, out_dim, heads=headnum)
    def forward(self, x, edge_index):
        x = torch.tanh(self.gatconv(x, edge_index))
        return x

class Depth(torch.nn.Module):
    def __init__(self, in_dim, hidden):
        super(Depth, self).__init__()
        self.lstm = torch.nn.LSTM(in_dim, hidden, 1, bias=False)

    def forward(self, x, h, c):
        x, (h, c) = self.lstm(x, (h, c))
        return x, (h, c)

#todo:
class GeniePathLayer(torch.nn.Module):
    def __init__(self, in_dim, dim, headnum, lstm_hidden):
        super(GeniePathLayer, self).__init__()
        self.breadth_func = Breadth(in_dim, dim, headnum)
        self.depth_func = Depth(dim, lstm_hidden)

    def forward(self, x, edge_index, h, c):
        x = self.breadth_func(x, edge_index)
        x = x[None, :]
        x, (h, c) = self.depth_func(x, h, c)
        x = x[0]
        return x, (h, c)


class GeniePath(torch.nn.Module):
    def __init__(self, in_dim, out_dim, head_num, lstm_hidden, num_layer, dim, residual_weight, device):
        super(GeniePath, self).__init__()
        self.lstm_hidden = lstm_hidden
        self.device = device
        self.lin1 = torch.nn.Linear(in_dim, dim)
        self.gplayers = torch.nn.ModuleList([GeniePathLayer(dim, dim, head_num, lstm_hidden) for i in range(num_layer)])
        self.lin2 = torch.nn.Linear(dim, out_dim)
        self.residual_weight = residual_weight

    def reset_parameters(self):
        pass

    def forward(self, x, edge_index):
        x = self.lin1(x)
        input = x
        h = torch.zeros(1, x.shape[0], self.lstm_hidden).to(self.device)
        c = torch.zeros(1, x.shape[0], self.lstm_hidden).to(self.device)
        for i, l in enumerate(self.gplayers):
            x, (h, c) = self.gplayers[i](x, edge_index, h, c)
            #add skip layer
            x = x+input*self.residual_weight

        x = self.lin2(x)
        return x


class GeniePathLazy(torch.nn.Module):
    def __init__(self, in_dim, out_dim, head_num, lstm_hidden, num_layer, dim, residual_weight, device):
        super(GeniePathLazy, self).__init__()
        self.device = device
        self.lstm_hidden = lstm_hidden
        self.lin1 = torch.nn.Linear(in_dim, dim)
        #if breadth_model == 'GAT':
        self.breaths = torch.nn.ModuleList([Breadth(dim, dim, head_num) for i in range(num_layer)])
        # elif breadth_model == 'SGC':
        #     self.breaths = torch.nn.ModuleList([Breadth(dim, dim, normalization, degree, device) for i in range(num_layer)])
        self.depths = torch.nn.ModuleList([Depth(dim * 2, lstm_hidden) for i in range(num_layer)])
        self.lin2 = torch.nn.Linear(dim, out_dim)
        self.residual_weight = residual_weight

    def reset_parameters(self):
        pass

    def forward(self, x, edge_index):
        x = self.lin1(x)

        #inp = x
        h = torch.zeros(1, x.shape[0], self.lstm_hidden).to(self.device)
        c = torch.zeros(1, x.shape[0], self.lstm_hidden).to(self.device)
        h_tmps = []
        for i, l in enumerate(self.breaths):
            h_tmps.append(self.breaths[i](x, edge_index))
        x = x[None, :]
        inp = x
        for i, l in enumerate(self.depths):
            in_cat = torch.cat((h_tmps[i][None, :], x), -1)
            x, (h, c) = self.depths[i](in_cat, h, c)
            x = x+self.residual_weight*inp
        x = self.lin2(x[0])
        return F.log_softmax(x, dim=-1)

def train(model, x, y_true, train_idx, optimizer, genie=False, adj_t=None):
    model.train()

    optimizer.zero_grad()
    if genie:
        out = model(x, adj_t)[train_idx]
    else: out = model(x[train_idx])
    loss = F.nll_loss(out, y_true.squeeze(1)[train_idx])
    # loss = F.cross_entropy(out, y_true.squeeze(1)[train_idx])

    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, x, y, split_idx, evaluator, genie=False, adj_t=None):
    model.eval()

    if genie:
        out = model(x, adj_t)
    else: out = model(x)

    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return (train_acc, valid_acc, test_acc), out

    
        
            
def main():
    parser = argparse.ArgumentParser(description='gen_models')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='arxiv')
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--use_embeddings', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--head_num', type=int, default=1)
    parser.add_argument('--lstm_hidden', type=int, default=256)
    parser.add_argument('--num_layer', type=int, default=4)
    parser.add_argument('--genie_dim', type=int, default=256)
    parser.add_argument('--residual_weight', type=float, default=0.2)

    args = parser.parse_args()
    print(args)

    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name=f'ogbn-{args.dataset}',transform=T.ToSparseTensor())
    
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    
    x = data.x

    
    split_idx = dataset.get_idx_split()
    # preprocess_data = PygNodePropPredDataset(name=f'ogbn-{args.dataset}')[0]
    # if args.dataset == 'arxiv':
    #     embeddings = torch.cat([preprocess(preprocess_data, 'diffusion', post_fix=args.dataset),
    #                             preprocess(preprocess_data, 'spectral', post_fix=args.dataset)], dim=-1)
    # elif args.dataset == 'products':
    #     embeddings = preprocess(preprocess_data, 'spectral', post_fix=args.dataset)
    #
    # if args.use_embeddings:
    #     x = torch.cat([x, embeddings], dim=-1)
        
    if args.dataset == 'arxiv':
        x = (x-x.mean(0))/x.std(0)

    if args.model == 'mlp':        
        model = MLP(x.size(-1),args.hidden_channels, dataset.num_classes, args.num_layers, 0.5, args.dataset == 'products').to(device)
    elif args.model=='linear':
        model = MLPLinear(x.size(-1), dataset.num_classes).to(device)
    elif args.model=='plain':
        model = MLPLinear(x.size(-1), dataset.num_classes).to(device)
    elif args.model=='sgc':
        model = SGC(x.size(-1), dataset.num_classes).to(device)
    elif args.model=='genie':
        model = GeniePathLazy(x.size(-1), dataset.num_classes, args.head_num, args.lstm_hidden, args.num_layer, args.genie_dim, args.residual_weight, args.device).to(device)

    x = x.to(device)
    y_true = data.y.to(device)
    train_idx = split_idx['train'].to(device)

    
    model_dir = prepare_folder(f'{args.dataset}_{args.model}', model)

    
    evaluator = Evaluator(name=f'ogbn-{args.dataset}')
    logger = Logger(args.runs, args)
    
    for run in range(args.runs):
        import gc
        gc.collect()
        print(sum(p.numel() for p in model.parameters()))
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_valid = 0
        best_out = None
        for epoch in range(1, args.epochs):
            loss = train(model, x, y_true, train_idx, optimizer, genie=args.model=='genie', adj_t=data.adj_t.to(device))
            result, out = test(model, x, y_true, split_idx, evaluator)
            train_acc, valid_acc, test_acc = result
            if valid_acc > best_valid:
                best_valid = valid_acc
                best_out = out.cpu().exp()

            if (epoch % 10 == 0):
                print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {100 * train_acc:.2f}%, '
                          f'Valid: {100 * valid_acc:.2f}% '
                          f'Test: {100 * test_acc:.2f}%')
            logger.add_result(run, result)

        logger.print_statistics(run)
        torch.save(best_out, f'{model_dir}/{run}.pt')

    logger.print_statistics()




if __name__ == "__main__":
    main()
