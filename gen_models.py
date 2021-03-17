import argparse

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import os
import shutil

from diffusion_feature import preprocess
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
        # return F.log_softmax(self.lin(x), dim=-1)
        return self.lin(x)


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
        return F.log_softmax(self.W(x), dim=-1)


def prepare_folder(name, model):
    model_dir = f'models/{name}'

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    with open(f'{model_dir}/metadata', 'w') as f:
        f.write(f'# of params: {sum(p.numel() for p in model.parameters())}\n')
    return model_dir


def train(model, x, y_true, train_idx, optimizer, genie=False, adj_t=None):
    model.train()
    optimizer.zero_grad()
    out = model(x, adj_t)[train_idx] if genie else model(x[train_idx])

    loss = F.nll_loss(out, y_true.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, x, y, split_idx, evaluator, genie=False, adj_t=None):
    model.eval()
    out = model(x, adj_t) if genie else model(x)
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
    parser.add_argument('--use_diffusion',  type=bool, default=True)
    parser.add_argument('--use_sgc',  type=bool, default=True)
    parser.add_argument('--use_sgc_cache', type=bool, default=True)
    parser.add_argument('--pair_norm', type=str, default="PN-SI")
    parser.add_argument('--use_spectral', action='store_true')
    parser.add_argument('--num_propagations', type=int, default=50)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--epochs_print', type=int, default=10)
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
    preprocess_data = PygNodePropPredDataset(name=f'ogbn-{args.dataset}')[0]

    embeddings_list = []
    if args.use_embeddings:
        if args.use_diffusion:
            embeddings_list.append(preprocess(preprocess_data, 'diffusion', post_fix=args.dataset))
        if args.use_sgc:
            embeddings_list.append(preprocess(preprocess_data, 'sgc', post_fix=args.dataset,
                num_propagations=args.num_propagations, use_cache=args.use_sgc_cache, pairnorm=args.pair_norm))
        if args.use_spectral:
            embeddings_list.append(preprocess(preprocess_data, 'spectral', post_fix=args.dataset))

    if args.dataset == 'arxiv' and args.use_embeddings:
        embeddings = torch.cat(embeddings_list, dim=-1)

    elif args.dataset == 'products' and args.use_embeddings:
        embeddings = torch.cat(embeddings_list, dim=-1)

    if args.use_embeddings:
        x = torch.cat([x, embeddings], dim=-1)
        
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
        raise NotImplementedError
        # model = GeniePathLazy(x.size(-1), dataset.num_classes, args.head_num, args.lstm_hidden, args.num_layer, args.genie_dim, args.residual_weight, args.device).to(device)

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
            result, out = test(model, x, y_true, split_idx, evaluator, genie=args.model=='genie', adj_t=data.adj_t.to(device))
            train_acc, valid_acc, test_acc = result
            if valid_acc > best_valid:
                best_valid = valid_acc
                best_out = out.cpu().exp()

            if epoch % args.epochs_print == 0:
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
