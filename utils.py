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


def _binarize_symmetric(adj_sp):
    """二值化 + 对称化 + 去自环, 返回 scipy 稀疏矩阵。"""
    a = (adj_sp > 0).astype(np.float64)
    a = a + a.T
    a = (a > 0).astype(np.float64)
    a.setdiag(0)
    a.eliminate_zeros()
    return a


def _sparsify_topk(adj_sp, k):
    """对每个节点只保留 top-k 邻居 (按 meta-path 原始权重/度数)，结果对称化。

    用于稠密 meta-path (如 ACM-PSP, DBLP-APCPA) 的内存控制。
    """
    csr = adj_sp.tocsr()
    n = csr.shape[0]
    rows, cols = [], []
    for i in range(n):
        start, end = csr.indptr[i], csr.indptr[i + 1]
        if end - start <= k:
            cols.append(csr.indices[start:end])
        else:
            vals = csr.data[start:end]
            topk_idx = np.argpartition(vals, -k)[-k:]
            cols.append(csr.indices[start:end][topk_idx])
        rows.append(np.full(len(cols[-1]), i, dtype=np.int64))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    sparse = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    sparse = sparse + sparse.T
    sparse = (sparse > 0).astype(np.float64)
    sparse.setdiag(0)
    sparse.eliminate_zeros()
    return sparse


def get_cs_dataset(root, name, meta_path=None, multi_relation=False,
                   cs_relations=None, sparsify_topk=None):
    """
    Load community-search heterogeneous graph dataset.

    Args:
        root: dataset root (e.g. './datasets/')
        name: one of 'ACM', 'DBLP', 'IMDB'
        meta_path: 'pap.npz' for a single meta-path, 'all' to merge everything,
                   or None to use a curated default (avoids memory blow-up).
        multi_relation: True 时额外返回 per-relation edge_index_list。
        cs_relations: 多关系模式下可选的 meta-path 名列表 (不含 .npz, 如 ['pap']);
                      None 表示用全部 cfg['meta_paths']。
        sparsify_topk: int or None. 非 None 时, 对平均度数超过此值的稠密 meta-path
                       自动做 top-k 稀疏化 (每节点保留 k 个最强邻居)。
    Returns:
        list containing one PyG Data object. 多关系模式下 data 额外带有
        edge_index_list / num_relations / relation_names 属性。
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
    adj = _binarize_symmetric(adj)
    edge_index = _sparse_to_edge_index(adj)

    data = Data(x=x, edge_index=edge_index, y=y)

    if multi_relation:
        # 选择参与的 meta-path: cs_relations 优先, 否则全部
        all_mp = cfg['meta_paths']
        if cs_relations:
            wanted = set(cs_relations)
            rel_paths = [mp for mp in all_mp
                         if mp.replace('.npz', '') in wanted]
            if not rel_paths:
                raise ValueError(
                    f"cs_relations={cs_relations} 未匹配 {name} 的任何 meta-path "
                    f"{[mp.replace('.npz','') for mp in all_mp]}")
        else:
            rel_paths = list(all_mp)

        edge_index_list = []
        relation_names = []
        for mp in rel_paths:
            m_raw = sp.load_npz(osp.join(base, mp))
            n = m_raw.shape[0]
            nnz = m_raw.nnz
            avg_deg = nnz / max(n, 1)
            if sparsify_topk is not None and avg_deg > sparsify_topk:
                # 用原始 path-count 权重做 top-k, 再二值对称化
                m = _sparsify_topk(m_raw, sparsify_topk)
                print(f"  [sparsify] {mp}: {nnz} -> {m.nnz} edges "
                      f"(avg_deg {avg_deg:.0f} -> {m.nnz/max(n,1):.0f}, "
                      f"top-k={sparsify_topk})")
            else:
                m = _binarize_symmetric(m_raw)
            edge_index_list.append(_sparse_to_edge_index(m))
            relation_names.append(mp.replace('.npz', ''))

        data.edge_index_list = edge_index_list
        data.num_relations = len(edge_index_list)
        data.relation_names = relation_names

    return [data]