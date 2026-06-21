import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import random
import os.path as osp
import scipy.sparse as sp

from torch_geometric.datasets import (
    Planetoid, CitationFull, Amazon, Coauthor,
    WikipediaNetwork, WebKB, Actor
)
import torch_geometric.transforms as T
from deeprobust.graph.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse


# =============================================================================
# Random Seed Configuration
# =============================================================================

def set_everything(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Dataset Loading Utilities
# =============================================================================

def get_dataset(path, name):
    # Validate dataset name
    assert name in [
        'Cora', 'CiteSeer', "AmazonC", "AmazonP", 
        'CoauthorC', 'CoauthorP', "PubMed", 
        'cora_lcc', 'citeseer_lcc',
        'Cornell', 'Texas', 'Wisconsin', 
        'chameleon', 'squirrel', 'Actor'
    ]
    
    # -------------------------------------------------------------------------
    # Heterophilous Graph Datasets
    # -------------------------------------------------------------------------
    if name == "Actor":
        path = f'{path}/{name}'
        return Actor(path, transform=T.NormalizeFeatures())
    
    if name in ['Cornell', 'Texas', 'Wisconsin']:
        return WebKB(path, name, transform=T.NormalizeFeatures())
    
    if name in ['chameleon', 'squirrel']:
        return WikipediaNetwork(path, name, transform=T.NormalizeFeatures())
    
    # -------------------------------------------------------------------------
    # Amazon Datasets
    # -------------------------------------------------------------------------
    if name == "AmazonC":
        return Amazon(path, "Computers", T.NormalizeFeatures())
    
    if name == "AmazonP":
        return Amazon(path, "Photo", T.NormalizeFeatures())
    
    # -------------------------------------------------------------------------
    # Coauthor Datasets
    # -------------------------------------------------------------------------
    if name == 'CoauthorC':
        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())
    
    if name == 'CoauthorP':
        return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())
    
    # -------------------------------------------------------------------------
    # DeepRobust Datasets (LCC variants)
    # -------------------------------------------------------------------------
    if name == "cora_lcc":
        name = "cora"
        data = Dataset(root=path, name=name, setting='prognn')
        adj, features, labels = data.adj, data.features, data.labels
        dataset = Data()
        dataset.x = torch.from_numpy(features.toarray()).float()
        dataset.y = torch.from_numpy(labels).long()
        dataset.edge_index = dense_to_sparse(torch.from_numpy(adj.toarray()))[0].long()
        return [dataset]
    
    if name == "citeseer_lcc":
        name = "citeseer"
        data = Dataset(root=path, name=name, setting='prognn')
        adj, features, labels = data.adj, data.features, data.labels
        dataset = Data()
        dataset.x = torch.from_numpy(features.toarray()).float()
        dataset.y = torch.from_numpy(labels).long()
        dataset.edge_index = dense_to_sparse(torch.from_numpy(adj.toarray()))[0].long()
        return [dataset]
    
    # -------------------------------------------------------------------------
    # Default: Planetoid Datasets (Cora, CiteSeer, PubMed)
    # -------------------------------------------------------------------------
    return Planetoid(
        path,
        name,
        "public",
        T.NormalizeFeatures()
    )


# =============================================================================
# Community-Search Heterogeneous Graph Datasets (ACM / DBLP / IMDB)
# =============================================================================

CS_DATASETS = {
    'ACM': {
        'dir': 'acm',
        'feat': 'p_feat.npz',
        'meta_paths': ['pap.npz', 'psp.npz'],
        # PSP is extremely dense (~4.3M nnz); default to the sparse PAP only.
        'default': ['pap.npz'],
    },
    'DBLP': {
        'dir': 'dblp',
        'feat': 'a_feat.npz',
        'meta_paths': ['apa.npz', 'apcpa.npz', 'aptpa.npz'],
        # APCPA/APTPA are extremely dense (5M/7M nnz); default to sparse APA only.
        'default': ['apa.npz'],
    },
    'IMDB': {
        'dir': 'self_imdb',
        'feat': 'm_feat.npz',
        'meta_paths': ['mam.npz', 'mdm.npz'],
        # both are sparse enough to merge.
        'default': ['mam.npz', 'mdm.npz'],
    },
}


def _sparse_to_edge_index(adj_sp):
    """Convert scipy sparse matrix to PyG edge_index [2, nnz]."""
    coo = adj_sp.tocoo()
    row = torch.from_numpy(coo.row.astype(np.int64))
    col = torch.from_numpy(coo.col.astype(np.int64))
    return torch.stack([row, col], dim=0)


def get_cs_dataset(root, name, meta_path=None):
    """
    Load community-search heterogeneous graph dataset.

    Args:
        root: dataset root (e.g. './datasets/')
        name: one of 'ACM', 'DBLP', 'IMDB'
        meta_path: 'pap.npz' for a single meta-path, 'all' to merge everything,
                   or None to use a curated default (avoids memory blow-up).
    Returns:
        list containing one PyG Data object
    """
    cfg = CS_DATASETS[name]
    base = osp.join(root, cfg['dir'])

    feat_sp = sp.load_npz(osp.join(base, cfg['feat']))
    x = torch.from_numpy(feat_sp.toarray()).float()
    # row-normalize features (matches T.NormalizeFeatures used for other datasets)
    row_sum = x.sum(dim=1, keepdim=True).clamp(min=1.0)
    x = x / row_sum

    labels = np.load(osp.join(base, 'labels.npy'))
    y = torch.from_numpy(labels.astype(np.int64))

    if meta_path is not None and meta_path != 'all':
        paths_to_merge = [meta_path]
    elif meta_path == 'all':
        paths_to_merge = cfg['meta_paths']
    else:
        paths_to_merge = cfg['default']

    adj = None
    for mp in paths_to_merge:
        m = sp.load_npz(osp.join(base, mp))
        m = (m > 0).astype(np.float64)
        adj = m if adj is None else adj + m
    adj = (adj > 0).astype(np.float64)

    adj = adj + adj.T
    adj = (adj > 0).astype(np.float64)
    adj.setdiag(0)
    adj.eliminate_zeros()

    edge_index = _sparse_to_edge_index(adj)

    data = Data(x=x, edge_index=edge_index, y=y)
    return [data]