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
# SNAP Community Dataset Loaders (com-Amazon / com-DBLP / com-Youtube)
# =============================================================================

def _load_snap_community_dataset(root_path, data_dir, edge_filename, cmty_filename, label):
    """
    通用 SNAP 社区数据集加载器。

    适用于 SNAP "Networks with Ground-Truth Communities" 系列数据集:
    com-Amazon, com-DBLP, com-Youtube, com-LiveJournal 等。

    数据格式:
    - {edge_filename}: 无向边列表 (可能有 # 开头的注释行)
    - {cmty_filename}: 社区列表 (每行一个社区的成员节点 ID)

    注意: 这些数据集没有节点属性特征, 使用图结构特征 (degree, log-degree) 代替。

    Args:
        root_path: datasets/ 根目录
        data_dir: 数据集子目录名 (如 'com-Amazon')
        edge_filename: 边文件名 (如 'com-amazon.ungraph.txt')
        cmty_filename: 社区文件名 (如 'com-amazon.top5000.cmty.txt')
        label: 日志标签 (如 'com-amazon')

    返回: PyG Data 对象的列表 (单元素)
    """
    data_path = osp.join(root_path, data_dir)

    # ===== Step 1: 读取边文件 =====
    # SNAP 数据可能直接是文件，也可能被放在同名子目录里
    edge_file = osp.join(data_path, edge_filename, edge_filename)
    if not osp.exists(edge_file):
        edge_file = osp.join(data_path, edge_filename)

    print(f"[{label}] Loading edges from: {edge_file}")

    raw_edges = []
    max_node_id = 0
    with open(edge_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                src, dst = int(parts[0]), int(parts[1])
                raw_edges.append((src, dst))
                max_node_id = max(max_node_id, src, dst)

    num_nodes = max_node_id + 1
    print(f"[{label}] Raw edges: {len(raw_edges)}, nodes: {num_nodes}")

    # 构建 edge_index (对称化)
    edge_src = [e[0] for e in raw_edges]
    edge_dst = [e[1] for e in raw_edges]
    edge_src_sym = edge_src + edge_dst
    edge_dst_sym = edge_dst + edge_src
    edge_index = torch.tensor([edge_src_sym, edge_dst_sym], dtype=torch.long)

    print(f"[{label}] After symmetrize: {edge_index.size(1)} edges")

    # ===== Step 2: 构建图结构特征 =====
    print(f"[{label}] Computing structural features...")

    degree = torch.zeros(num_nodes, dtype=torch.float32)
    for e in raw_edges:
        degree[e[0]] += 1
        degree[e[1]] += 1

    degree_norm = degree / (degree.max() + 1e-8)
    log_degree = torch.log1p(degree)
    log_degree = log_degree / (log_degree.max() + 1e-8)

    # 特征: [归一化度, 对数度, 常数1]
    x = torch.stack([degree_norm, log_degree, torch.ones(num_nodes)], dim=1)
    feat_dim = x.size(1)

    print(f"[{label}] Feature dimension: {feat_dim} (structural)")

    # ===== Step 3: 读取社区标签 =====
    cmty_file = osp.join(data_path, cmty_filename, cmty_filename)
    if not osp.exists(cmty_file):
        cmty_file = osp.join(data_path, cmty_filename)

    print(f"[{label}] Loading communities from: {cmty_file}")

    communities = []
    with open(cmty_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                members = [int(x) for x in parts]
                communities.append(members)

    num_communities = len(communities)
    print(f"[{label}] Total communities: {num_communities}")

    # 为每个节点分配社区标签 (取它所属的第一个社区)
    node_to_community = {}
    for cid, members in enumerate(communities):
        for nid in members:
            if nid not in node_to_community:
                node_to_community[nid] = cid

    y = torch.full((num_nodes,), -1, dtype=torch.long)
    for nid, cid in node_to_community.items():
        if 0 <= nid < num_nodes:
            y[nid] = cid

    num_labeled = (y >= 0).sum().item()
    print(f"[{label}] Nodes with community labels: {num_labeled} / {num_nodes}")

    # ===== Step 4: 构建 PyG Data 对象 =====
    data = Data()
    data.x = x
    data.y = y
    data.edge_index = edge_index
    data.num_nodes = num_nodes
    data.num_classes = num_communities
    data.communities = communities

    # 创建训练/验证/测试掩码 (只对有标签的节点)
    labeled_indices = torch.where(y >= 0)[0]
    num_labeled_total = labeled_indices.size(0)
    perm = torch.randperm(num_labeled_total)
    n_train = int(0.6 * num_labeled_total)
    n_val = int(0.2 * num_labeled_total)

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[labeled_indices[perm[:n_train]]] = True
    data.val_mask[labeled_indices[perm[n_train:n_train + n_val]]] = True
    data.test_mask[labeled_indices[perm[n_train + n_val:]]] = True

    print(f"[{label}] Dataset summary:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {edge_index.size(1)}")
    print(f"  Features: {feat_dim} (structural)")
    print(f"  Communities: {num_communities}")
    print(f"  Labeled nodes: {num_labeled}")
    print(f"  Train/Val/Test: {n_train}/{n_val}/{num_labeled_total - n_train - n_val}")

    return [data]


def _load_com_amazon_dataset(root_path):
    return _load_snap_community_dataset(
        root_path, 'com-Amazon',
        'com-amazon.ungraph.txt', 'com-amazon.top5000.cmty.txt',
        'com-amazon')


def _load_com_dblp_dataset(root_path):
    return _load_snap_community_dataset(
        root_path, 'com-same-DBLP',
        'com-dblp.ungraph.txt', 'com-dblp.top5000.cmty.txt',
        'com-dblp')


def _load_com_youtube_dataset(root_path):
    return _load_snap_community_dataset(
        root_path, 'com-Youtube',
        'com-youtube.ungraph.txt', 'com-youtube.top5000.cmty.txt',
        'com-youtube')


# =============================================================================
# Twitter Ego-Network Dataset Loader (SNAP)
# =============================================================================

def _load_twitter_dataset(root_path):
    """
    加载 Stanford SNAP Twitter ego-network 数据集。

    数据来源: https://snap.stanford.edu/data/egonets-Twitter.html

    数据格式:
    - twitter_combined.txt: 合并的有向边 (a follows b)
    - twitter/twitter/{ego_id}.edges: 每个 ego 的边
    - twitter/twitter/{ego_id}.feat: 节点特征 (稀疏二值)
    - twitter/twitter/{ego_id}.circles: ego 标注的社区

    返回: PyG Data 对象的列表 (单元素)
    """
    import collections

    twitter_dir = osp.join(root_path, 'twitter')

    # ===== Step 1: 读取合并边文件, 构建节点 ID 映射 =====
    combined_file = osp.join(twitter_dir, 'twitter_combined.txt', 'twitter_combined.txt')
    if not osp.exists(combined_file):
        combined_file = osp.join(twitter_dir, 'twitter_combined.txt')

    print(f"[twitter] Loading edges from: {combined_file}")

    # 读取所有边
    raw_edges = []
    all_node_ids = set()
    with open(combined_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                src, dst = int(parts[0]), int(parts[1])
                raw_edges.append((src, dst))
                all_node_ids.add(src)
                all_node_ids.add(dst)

    print(f"[twitter] Raw edges: {len(raw_edges)}, unique nodes: {len(all_node_ids)}")

    # 构建节点 ID 到 0..N-1 的映射
    sorted_nodes = sorted(all_node_ids)
    node_to_idx = {nid: idx for idx, nid in enumerate(sorted_nodes)}
    num_nodes = len(sorted_nodes)

    # 转换边为索引格式
    edge_src = [node_to_idx[e[0]] for e in raw_edges]
    edge_dst = [node_to_idx[e[1]] for e in raw_edges]

    # 对称化 (Twitter 原始是有向的, 转为无向)
    edge_src_sym = edge_src + edge_dst
    edge_dst_sym = edge_dst + edge_src
    edge_index = torch.tensor([edge_src_sym, edge_dst_sym], dtype=torch.long)

    print(f"[twitter] After symmetrize: {edge_index.size(1)} edges")

    # ===== Step 2: 读取特征 =====
    # 收集所有 ego 网络的特征
    ego_dir = osp.join(twitter_dir, 'twitter', 'twitter')
    if not osp.exists(ego_dir):
        ego_dir = osp.join(twitter_dir, 'twitter')

    print(f"[twitter] Loading features from: {ego_dir}")

    # 先统计特征维度 (从第一个 .feat 文件)
    feat_files = [f for f in os.listdir(ego_dir) if f.endswith('.feat')]
    if not feat_files:
        raise FileNotFoundError(f"No .feat files found in {ego_dir}")

    # 读取所有特征, 建立 原始ID -> 特征 映射
    node_feat_raw = {}
    feat_dim = None

    for fname in feat_files:
        fpath = osp.join(ego_dir, fname)
        with open(fpath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    nid = int(parts[0])
                    feat = [int(x) for x in parts[1:]]
                    node_feat_raw[nid] = feat
                    if feat_dim is None:
                        feat_dim = len(feat)

    print(f"[twitter] Feature dimension: {feat_dim}, nodes with features: {len(node_feat_raw)}")

    # 构建完整特征矩阵 (缺失特征的节点用零向量)
    x = torch.zeros(num_nodes, feat_dim, dtype=torch.float32)
    for nid, idx in node_to_idx.items():
        if nid in node_feat_raw:
            x[idx] = torch.tensor(node_feat_raw[nid], dtype=torch.float32)

    # ===== Step 3: 读取 circles, 生成伪标签 =====
    # 每个节点取它所属的第一个 circle 的 ID 作为标签
    circle_files = [f for f in os.listdir(ego_dir) if f.endswith('.circles')]
    print(f"[twitter] Loading circles from {len(circle_files)} ego networks")

    node_label = {}  # 原始ID -> circle_id
    circle_count = 0

    for fname in circle_files:
        ego_id_str = fname.replace('.circles', '')
        fpath = osp.join(ego_dir, fname)
        with open(fpath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    circle_name = parts[0]
                    members = [int(x) for x in parts[1:]]
                    for nid in members:
                        if nid not in node_label:
                            node_label[nid] = circle_count
                    circle_count += 1

    print(f"[twitter] Total circles: {circle_count}, nodes with labels: {len(node_label)}")

    # 构建标签张量 (没有标签的节点用 -1, 后续可以过滤)
    y = torch.full((num_nodes,), -1, dtype=torch.long)
    for nid, idx in node_to_idx.items():
        if nid in node_label:
            y[idx] = node_label[nid]

    # ===== Step 4: 构建 PyG Data 对象 =====
    data = Data()
    data.x = x
    data.y = y
    data.edge_index = edge_index
    data.num_nodes = num_nodes
    data.num_classes = circle_count

    # 额外属性: ego 节点列表 (用于评估)
    ego_ids = [int(f.replace('.circles', '')) for f in circle_files]
    data.ego_nodes = torch.tensor([node_to_idx[eid] for eid in ego_ids if eid in node_to_idx],
                                   dtype=torch.long)

    # 创建训练/验证/测试掩码 (只对有标签的节点)
    labeled_mask = (y >= 0)
    num_labeled = labeled_mask.sum().item()
    print(f"[twitter] Labeled nodes: {num_labeled} / {num_nodes}")

    # 随机划分
    labeled_indices = torch.where(labeled_mask)[0]
    perm = torch.randperm(num_labeled)
    n_train = int(0.6 * num_labeled)
    n_val = int(0.2 * num_labeled)

    train_idx = labeled_indices[perm[:n_train]]
    val_idx = labeled_indices[perm[n_train:n_train + n_val]]
    test_idx = labeled_indices[perm[n_train + n_val:]]

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    print(f"[twitter] Dataset summary:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {edge_index.size(1)}")
    print(f"  Features: {feat_dim}")
    print(f"  Classes (circles): {circle_count}")
    print(f"  Train/Val/Test: {n_train}/{n_val}/{num_labeled - n_train - n_val}")
    print(f"  Ego nodes: {data.ego_nodes.size(0)}")

    return [data]


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
        'chameleon', 'squirrel', 'Actor',
        'twitter', 'com-amazon', 'com-dblp', 'com-youtube',
    ]

    # -------------------------------------------------------------------------
    # Twitter Ego-Network Dataset (SNAP)
    # -------------------------------------------------------------------------
    if name == 'twitter':
        return _load_twitter_dataset(path)

    # -------------------------------------------------------------------------
    # SNAP Community Datasets (同构, 有真实社区 ground truth)
    # -------------------------------------------------------------------------
    if name == 'com-amazon':
        return _load_com_amazon_dataset(path)
    if name == 'com-dblp':
        return _load_com_dblp_dataset(path)
    if name == 'com-youtube':
        return _load_com_youtube_dataset(path)
    
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
    'IMDB_NEW': {
        'dir': 'imdb_new',
        'feat': 'm_feat.npz',
        'meta_paths': ['mam.npz', 'mdm.npz'],
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
                   cs_relations=None, sparsify_topk=None, cs_full_graph=True):
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
        cs_full_graph: True 时额外构建 data.cs_edge_index —— 全量合并所有
                       cfg['meta_paths'] (不稀疏化) 的无向图, 专供社区搜索的贪婪/
                       AC 扩展使用。ACM 的默认 edge_index 只有 PAP, 缺 PSP 主题社区;
                       cs_edge_index 补齐 PSP 等稠密路径, 让扩展能到达同主题节点。
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

    if cs_full_graph:
        # 全量合并所有 meta-path (不稀疏化), 专供社区搜索扩展。
        cs_adj = None
        for mp in cfg['meta_paths']:
            m = sp.load_npz(osp.join(base, mp))
            m = (m > 0).astype(np.float64)
            cs_adj = m if cs_adj is None else cs_adj + m
        cs_adj = _binarize_symmetric(cs_adj)
        data.cs_edge_index = _sparse_to_edge_index(cs_adj)
        print(f"  [cs_full_graph] {name}: merged {cfg['meta_paths']} -> "
              f"{data.cs_edge_index.size(1)} directed edges (no sparsify)")

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