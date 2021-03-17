from tqdm import tqdm

import torch
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
import networkx as nx
import numpy as np


np.random.seed(0)


class PairNorm(torch.nn.Module):
    def __init__(self, mode='PN', scale=1):
        """
            mode:
              'None' : No normalization
              'PN'   : Original version
              'PN-SI'  : Scale-Individually version
              'PN-SCS' : Scale-and-Center-Simultaneously version

            ('SCS'-mode is not in the paper but we found it works well in practice,
              especially for GCN and GAT.)
            PairNorm is typically used after each graph convolution operation.
        """
        assert mode in ['None', 'PN', 'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

    def forward(self, x):
        if self.mode == 'None':
            return x

        col_mean = x.mean(0)
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = np.sqrt((1e-6 + np.power(x, 2).sum(1).mean()))
            x = self.scale * x / rownorm_mean

        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = np.sqrt((1e-6 + np.power(x, 2).sum(1, keepdims=True)))
            x = self.scale * x / rownorm_individual

        if self.mode == 'PN-SCS':
            rownorm_individual = np.sqrt((1e-6 + np.power(x, 2).sum(1, keepdims=True)))
            x = self.scale * x / rownorm_individual - col_mean

        return x


def sgc(x, adj, num_propagations, pairnorm=False):
    if pairnorm:
        norm = PairNorm("PN-SI", 1)
        x = norm(x)
        print("SGC layers =", num_propagations)
        for _ in tqdm(range(num_propagations)):
            x = adj @ x
            x = norm(x)
    else:
        for _ in tqdm(range(num_propagations)):
            x = adj @ x

    return torch.from_numpy(x).to(torch.float)


def diffusion(x, adj, num_propagations, p, alpha):
    if p is None:
        p = 1.
    if alpha is None:
        alpha = 0.5

    x = x ** p
    for _ in tqdm(range(num_propagations)):
        x = x - alpha * (sparse.eye(adj.shape[0]) - adj) @ x
        x = x ** p

    return torch.from_numpy(x).to(torch.float)


def spectral(data, post_fix):
    print('Setting up spectral embedding')
    data.edge_index = to_undirected(data.edge_index)

    N = data.num_nodes
    row, col = data.edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    adj = adj.to_scipy(layout='csr')

    G = nx.from_scipy_sparse_matrix(adj)
    lap_mtx = nx.normalized_laplacian_matrix(G)
    tsvd = TruncatedSVD(n_components=128)
    adj_tsvd = tsvd.fit(lap_mtx).transform(lap_mtx)

    result = torch.tensor(adj_tsvd).float()
    torch.save(result, f'embeddings/spectral{post_fix}.pt')
        
    return result


def preprocess(data, preprocess="diffusion", num_propagations=10, p=None, alpha=None, use_cache=True, post_fix="", pairnorm=False):
    if use_cache:
        try:
            x = torch.load(f'embeddings/{preprocess}{post_fix}.pt')
            print('Using cache')
            return x
        except:
            print(f'embeddings/{preprocess}{post_fix}.pt not found or not enough iterations! Regenerating it now')
            # Creates a new file
            with open(f'embeddings/{preprocess}{post_fix}.pt', 'w') as fp:
                pass

    if preprocess == "spectral":
        print(f'Start {preprocess} processing')
        return spectral(data, post_fix)
    
    print('Computing adj...')
    N = data.num_nodes
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    row, col = data.edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    adj = adj.set_diag()
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    adj = adj.to_scipy(layout='csr')

    print(f'Start {preprocess} processing')

    if preprocess == "sgc":
        result = sgc(data.x.numpy(), adj, num_propagations, pairnorm)
    if preprocess == "diffusion":
        result = diffusion(data.x.numpy(), adj, num_propagations, p=p, alpha = alpha)

    torch.save(result, f'embeddings/{preprocess}{post_fix}.pt')
    
    return result
    
