
import os
import torch
import torch.nn.functional as F
import functools
import numpy as np
from abc import ABC, abstractmethod
from collections import deque
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import f1_score


def get_split(num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.8):
    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    indices = torch.randperm(num_samples)
    return {
        'train': indices[:train_size],
        'valid': indices[train_size: num_samples - test_size],
        'test': indices[num_samples - test_size:]
    }


class LogisticRegression(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        z = self.fc(x)
        return z


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict) -> dict:
        pass

    def __call__(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict) -> dict:
        for key in ['train', 'test', 'valid']:
            assert key in split

        result = self.evaluate(x, y, split)
        return result


class LREvaluator(BaseEvaluator):
    def __init__(self, num_epochs: int = 5000, learning_rate: float = 0.01,
                 weight_decay: float = 0.0, test_interval: int = 20):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.test_interval = test_interval

    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict):
        device = x.device
        x = x.detach().to(device)
        input_dim = x.size()[1]
        y = y.to(device)
        num_classes = y.max().item() + 1
        classifier = LogisticRegression(input_dim, num_classes).to(device)
        optimizer = Adam(classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        output_fn = nn.LogSoftmax(dim=-1)
        criterion = nn.NLLLoss()

        best_val_acc = 0
        best_test_acc = 0
        best_test_micro = 0
        best_test_macro = 0
        best_epoch = 0

        with tqdm(total=self.num_epochs, desc='(LR)',
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:
            for epoch in range(self.num_epochs):
                classifier.train()
                optimizer.zero_grad()

                output = classifier(x[split['train']])
                loss = criterion(output_fn(output), y[split['train']])

                loss.backward()
                optimizer.step()

                if (epoch + 1) % self.test_interval == 0:
                    classifier.eval()
                    y_test = y[split['test']].detach().cpu().numpy()
                    y_pred = classifier(x[split['test']]).argmax(-1).detach().cpu().numpy()
                    test_micro = f1_score(y_test, y_pred, average='micro')
                    test_macro = f1_score(y_test, y_pred, average='macro')
                    test_acc = (y_test == y_pred).mean()

                    y_val = y[split['valid']].detach().cpu().numpy()
                    y_pred = classifier(x[split['valid']]).argmax(-1).detach().cpu().numpy()
                    val_acc = (y_val == y_pred).mean()

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_test_acc = test_acc
                        best_test_micro = test_micro
                        best_test_macro = test_macro
                        best_epoch = epoch

                    pbar.set_postfix({'best test ACC': best_test_acc, 'F1Mi': best_test_micro, 'F1Ma': best_test_macro})
                    pbar.update(self.test_interval)

        return {
            'micro_f1': best_test_micro,
            'macro_f1': best_test_macro,
            'acc': best_test_acc,
        }

def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = [f(*args, **kwargs) for _ in range(n_times)]
            statistics = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)}
            print_statistics(statistics, f.__name__)
            return statistics
        return wrapper
    return decorator

def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret

def print_statistics(statistics, function_name):
    print(f'(E) | {function_name}:', end=' ')
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]['mean']
        std = statistics[key]['std']
        print(f'{key}={mean:.4f}+-{std:.4f}', end='')
        if i != len(statistics.keys()) - 1:
            print(',', end=' ')
        else:
            print()
            
def label_classification(embeddings, data, dataset_name, ratio = 0.1, test_repeat = 10):
    y = data.y
    micro_f1 = torch.zeros(test_repeat)
    macro_f1 = torch.zeros(test_repeat)
    acc= torch.zeros(test_repeat)
    for num in range(test_repeat):  
        split = get_split(embeddings.shape[0], train_ratio = 0.1, test_ratio = 0.8)
        logreg = LREvaluator(num_epochs=20000)
        result = logreg.evaluate(embeddings, y, split)
        micro_f1[num]= result['micro_f1']
        macro_f1[num]= result['macro_f1']
        acc[num]= result['acc']
    print('micro_f1:', micro_f1.mean().item(),'std:', micro_f1.std().item())
    print('macro_f1:', macro_f1.mean().item(),'std:', macro_f1.std().item())
    print('accuracy:', acc.mean().item(),'std:', acc.std().item())
    return micro_f1.mean().item()*100, micro_f1.std().item()*100, macro_f1.mean().item()*100, macro_f1.std().item()*100, acc.mean().item()*100, acc.std().item()*100


def _boost_multiplier(node_boost, boost_factor, N):
    """把可疑分 [0,1] 转成相似度乘子: 满分节点 ×boost_factor, 零分 ×1。"""
    if node_boost is None:
        return None
    nb = node_boost.detach().cpu().numpy() if hasattr(node_boost, 'detach') \
        else np.asarray(node_boost)
    nb = nb.astype(np.float64)
    rng = nb.max() - nb.min()
    nb = (nb - nb.min()) / (rng + 1e-8)
    return 1.0 + (boost_factor - 1.0) * nb


def build_fixed_queries(data, num_queries=40, seed=0, query_file=None):
    """构建/加载一组固定查询节点, 用于跨配置(GCN vs HII)可复现对比 (对齐 CLUHCS 40 查询协议)。

    优先从 query_file 读取(每行一个节点 id); 文件不存在时按类别分层采样
    num_queries 个节点并写入 query_file, 之后所有配置都复用同一组查询。
    """
    y = data.y.detach().cpu().numpy()
    N = len(y)

    if query_file is not None and os.path.exists(query_file):
        with open(query_file) as f:
            q = [int(line.strip()) for line in f if line.strip()]
        print(f"[query] 载入固定查询 {len(q)} 个 <- {query_file}")
        return np.array(q)

    rng = np.random.default_rng(seed)
    labels = np.unique(y)
    per = max(1, num_queries // len(labels))
    chosen = []
    for lb in labels:
        idx = np.where(y == lb)[0]
        take = min(per, len(idx))
        chosen.extend(rng.choice(idx, size=take, replace=False).tolist())
    if len(chosen) < num_queries:
        remaining = np.setdiff1d(np.arange(N), np.array(chosen))
        extra = rng.choice(remaining,
                           size=min(num_queries - len(chosen), len(remaining)),
                           replace=False)
        chosen.extend(extra.tolist())
    chosen = np.array(chosen[:num_queries])

    if query_file is not None:
        os.makedirs(os.path.dirname(query_file) or '.', exist_ok=True)
        with open(query_file, 'w') as f:
            for q in chosen:
                f.write(f"{int(q)}\n")
        print(f"[query] 生成固定查询 {len(chosen)} 个 (分层采样) -> {query_file}")
    return chosen


def community_search(embeddings, data, topk=(10, 20, 50), num_queries=None, seed=0,
                     node_boost=None, boost_factor=1.5, queries=None):
    """
    基于嵌入的社区搜索评估 (Precision / Recall / F1 / Jaccard)。

    协议: 对每个查询节点 q, 按嵌入余弦相似度取 top-k 节点作为预测社区 Cq,
    与 q 同标签的节点集合作为真实社区 Ct, 比较两个集合。

    topk 中可包含字符串 'oracle', 表示 k = |Ct| (每个查询自适应)。

    Args:
        embeddings: [N, D] 节点嵌入
        data: 含 data.y 标签
        topk: 评估的 k 值列表, 支持整数或 'oracle'
        num_queries: 采样的查询节点数; None 表示用全部节点作为查询
        seed: 查询采样随机种子
    Returns:
        dict: {k: {'precision','recall','f1','jaccard'}} (百分比)
    """
    y = data.y.detach().cpu().numpy()
    emb = F.normalize(embeddings.detach().cpu(), dim=-1)
    N = emb.shape[0]

    if queries is not None:
        queries = np.asarray(queries)
    elif num_queries is None or num_queries >= N:
        queries = np.arange(N)
    else:
        rng = np.random.default_rng(seed)
        queries = rng.choice(N, size=num_queries, replace=False)

    sims = (emb @ emb.t()).numpy()

    mult = _boost_multiplier(node_boost, boost_factor, N)
    if mult is not None:
        sims = sims * mult[None, :]      # 对可疑候选节点的相似度加权

    # 预计算每个标签的节点集合 (避免每次查询重复 np.where)
    label_sets = {}
    for label in np.unique(y):
        label_sets[label] = set(np.where(y == label)[0].tolist())

    results = {}
    for k in topk:
        P, R, Fm, J = [], [], [], []
        for q in queries:
            truth = label_sets[y[q]].copy()
            truth.discard(int(q))
            if len(truth) == 0:
                continue

            actual_k = len(truth) if k == 'oracle' else k

            order = np.argsort(-sims[q])
            order = order[order != q][:actual_k]
            pred = set(order.tolist())

            inter = len(pred & truth)
            union = len(pred | truth)
            p = inter / len(pred) if len(pred) > 0 else 0.0
            r = inter / len(truth)
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            j = inter / union if union > 0 else 0.0
            P.append(p); R.append(r); Fm.append(f); J.append(j)

        k_label = 'oracle' if k == 'oracle' else str(k)
        results[k] = {
            'precision': float(np.mean(P)) * 100,
            'recall': float(np.mean(R)) * 100,
            'f1': float(np.mean(Fm)) * 100,
            'jaccard': float(np.mean(J)) * 100,
        }
        print(f'[CS] k={k_label:<7s} '
              f"P={results[k]['precision']:.2f} "
              f"R={results[k]['recall']:.2f} "
              f"F1={results[k]['f1']:.2f} "
              f"Jaccard={results[k]['jaccard']:.2f}")
    return results


def _cs_edge_index(data):
    """社区搜索用的邻接来源: 优先全量合并图 cs_edge_index (含 PSP 等稠密路径),
    退化到 data.edge_index。让贪婪/AC 扩展能走到同主题社区, 而非只走默认稀疏路径。"""
    return getattr(data, 'cs_edge_index', None) if getattr(
        data, 'cs_edge_index', None) is not None else data.edge_index


def _build_adj_list(edge_index, num_nodes):
    """从 edge_index(评估时为双向边)构建邻接表 list[set[int]],去自环。"""
    adj = [set() for _ in range(num_nodes)]
    ei = edge_index.detach().cpu().numpy()
    for s, d in zip(ei[0].tolist(), ei[1].tolist()):
        if s != d:
            adj[s].add(d)
            adj[d].add(s)
    return adj


def _limit_hse_pool(cand_arr, scores, pool_size):
    pool_size = int(pool_size)
    if pool_size <= 0 or cand_arr.size <= pool_size:
        return cand_arr, scores
    top_idx = np.argpartition(-scores, pool_size - 1)[:pool_size]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return cand_arr[top_idx], scores[top_idx]


def _minmax_norm(values):
    values = values.astype(np.float64, copy=False)
    span = float(values.max() - values.min()) if values.size > 0 else 0.0
    if span <= 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - values.min()) / span


def _hse_candidate_scores(q, cand_arr, visited, adj, community_halo,
                          high_order_beta=0.0, comm_cohesion_beta=0.0,
                          boundary_gamma=0.0, comm_direct_beta=0.0,
                          normalize=False):
    if (high_order_beta <= 0 and comm_cohesion_beta <= 0
            and boundary_gamma <= 0 and comm_direct_beta <= 0):
        return None
    q_nbrs = adj[int(q)]
    q_deg = max(1, len(q_nbrs))
    visited_set = visited if isinstance(visited, set) else set(visited)
    reach_vals = np.zeros(cand_arr.size, dtype=np.float64)
    direct_vals = np.zeros(cand_arr.size, dtype=np.float64)
    cohesion_vals = np.zeros(cand_arr.size, dtype=np.float64)
    boundary_vals = np.zeros(cand_arr.size, dtype=np.float64)
    for i, c in enumerate(cand_arr):
        c = int(c)
        c_nbrs = adj[c]
        c_deg = max(1, len(c_nbrs))
        if high_order_beta > 0:
            reach_vals[i] = len(q_nbrs & c_nbrs) / np.sqrt(q_deg * c_deg)
        if comm_direct_beta > 0:
            direct_vals[i] = len(c_nbrs & visited_set) / c_deg
        if comm_cohesion_beta > 0:
            cohesion_vals[i] = len(c_nbrs & community_halo) / c_deg
        if boundary_gamma > 0:
            boundary_vals[i] = len(c_nbrs - community_halo - visited_set) / c_deg
    if normalize:
        if high_order_beta > 0:
            reach_vals = _minmax_norm(reach_vals)
        if comm_direct_beta > 0:
            direct_vals = _minmax_norm(direct_vals)
        if comm_cohesion_beta > 0:
            cohesion_vals = _minmax_norm(cohesion_vals)
        if boundary_gamma > 0:
            boundary_vals = _minmax_norm(boundary_vals)
    scores = np.zeros(cand_arr.size, dtype=np.float64)
    scores += high_order_beta * reach_vals
    scores += comm_direct_beta * direct_vals
    scores += comm_cohesion_beta * cohesion_vals
    scores -= boundary_gamma * boundary_vals
    return scores


def _idbr_diffusion_bridge(q, adj, steps=(2, 4), decay=1.0, max_states=50000,
                           sims_q=None, fanout=0):
    steps = tuple(sorted({int(s) for s in steps if int(s) > 0}))
    if not steps:
        return None
    max_step = max(steps)
    max_states = max(1, int(max_states))
    fanout = max(0, int(fanout))
    decay = max(0.0, float(decay))
    cur = {int(q): 1.0}
    scores = {}

    def _neighbors(u):
        nbrs = adj[int(u)]
        if fanout <= 0 or sims_q is None or len(nbrs) <= fanout:
            return nbrs
        arr = np.fromiter(nbrs, dtype=np.int64)
        vals = sims_q[arr]
        top_idx = np.argpartition(-vals, fanout - 1)[:fanout]
        return arr[top_idx].tolist()

    for step in range(1, max_step + 1):
        nxt = {}
        for u, prob in cur.items():
            nbrs = _neighbors(u)
            if not nbrs:
                continue
            share = prob / float(len(nbrs))
            for v in nbrs:
                nxt[int(v)] = nxt.get(int(v), 0.0) + share
        if len(nxt) > max_states:
            top = sorted(nxt.items(), key=lambda x: x[1], reverse=True)[:max_states]
            total = sum(v for _, v in top)
            nxt = {k: v / total for k, v in top} if total > 0 else dict(top)
        cur = nxt
        if step in steps:
            weight = decay ** (step - 1)
            for v, prob in cur.items():
                scores[v] = scores.get(v, 0.0) + weight * prob
    scores.pop(int(q), None)
    if not scores:
        return None
    vals = np.fromiter(scores.values(), dtype=np.float64)
    vmin = float(vals.min())
    span = float(vals.max() - vmin)
    if span <= 1e-12:
        return {k: 0.0 for k in scores}
    return {k: (float(v) - vmin) / span for k, v in scores.items()}


def _idbr_bridge_candidate_scores(cand_arr, bridge_scores):
    if not bridge_scores:
        return None
    return np.array([bridge_scores.get(int(c), 0.0) for c in cand_arr], dtype=np.float64)


def _idbr_local_bridge_candidate_scores(cand_arr, anchors, adj):
    anchors = [int(a) for a in anchors]
    if not anchors or cand_arr.size == 0:
        return None
    vals = np.zeros(cand_arr.size, dtype=np.float64)
    for ai, anchor in enumerate(anchors):
        a_nbrs = adj[int(anchor)]
        a_deg = max(1, len(a_nbrs))
        if not a_nbrs:
            continue
        for i, cand in enumerate(cand_arr):
            c_nbrs = adj[int(cand)]
            mids = a_nbrs & c_nbrs
            if not mids:
                continue
            vals[i] += sum(1.0 / max(1, len(adj[int(mid)])) for mid in mids) / a_deg
    vals = vals / float(len(anchors))
    return _minmax_norm(vals)


def _idbr_boundary_penalty_scores(cand_arr, visited, adj, community_halo, sims_q=None,
                                  bridge_scores=None, sim_gamma=0.5,
                                  cohesion_gamma=0.5, bridge_gamma=1.0,
                                  normalize=True):
    if cand_arr.size == 0 or bridge_gamma <= 0:
        return None
    local_bridge = _idbr_local_bridge_candidate_scores(cand_arr, visited, adj)
    global_bridge = _idbr_bridge_candidate_scores(cand_arr, bridge_scores)
    if local_bridge is None and global_bridge is None:
        return None
    if local_bridge is None:
        bridge = _minmax_norm(global_bridge)
    elif global_bridge is None:
        bridge = local_bridge
    else:
        bridge = np.maximum(local_bridge, _minmax_norm(global_bridge))
    cohesion = np.zeros(cand_arr.size, dtype=np.float64)
    for i, c in enumerate(cand_arr):
        c_nbrs = adj[int(c)]
        cohesion[i] = len(c_nbrs & community_halo) / max(1, len(c_nbrs))
    cohesion_risk = 1.0 - _minmax_norm(cohesion)
    risk = cohesion_gamma * cohesion_risk
    if sims_q is not None and sim_gamma > 0:
        sim_risk = 1.0 - _minmax_norm(sims_q[cand_arr].astype(np.float64, copy=False))
        risk = risk + sim_gamma * sim_risk
    vals = bridge_gamma * bridge * risk
    if normalize:
        vals = _minmax_norm(vals)
    return vals


def _recall_expand_community(q, comm, sims_q, adj, max_add=0, pool_size=0,
                             min_sim=None, high_order_beta=0.0,
                             comm_direct_beta=0.0, comm_cohesion_beta=0.0,
                             boundary_gamma=0.0, hse_normalize=False):
    max_add = max(0, int(max_add))
    if max_add <= 0 or not comm:
        return comm
    comm = set(int(v) for v in comm)
    frontier = set()
    for v in comm:
        frontier.update(adj[v])
    frontier.difference_update(comm)
    if not frontier:
        return comm
    cand_arr = np.array(list(frontier), dtype=np.int64)
    scores = sims_q[cand_arr].astype(np.float64, copy=True)
    if min_sim is not None:
        keep = scores >= float(min_sim)
        cand_arr = cand_arr[keep]
        scores = scores[keep]
        if cand_arr.size == 0:
            return comm
    cand_arr, scores = _limit_hse_pool(cand_arr, scores, pool_size)
    community_halo = set(comm)
    for v in comm:
        community_halo.update(adj[v])
    hse_scores = _hse_candidate_scores(
        q, cand_arr, comm, adj, community_halo,
        high_order_beta, comm_cohesion_beta, boundary_gamma,
        comm_direct_beta, normalize=hse_normalize)
    if hse_scores is not None:
        scores = scores + hse_scores
    take = min(max_add, cand_arr.size)
    if take <= 0:
        return comm
    chosen_idx = np.argpartition(-scores, take - 1)[:take]
    chosen_idx = chosen_idx[np.argsort(-scores[chosen_idx])]
    comm.update(int(cand_arr[i]) for i in chosen_idx)
    return comm


def _bfs_farthest(start, node_set, adj):
    """从 start 出发在 node_set 诱导子图上 BFS,返回 (最远节点, 最远距离)。"""
    dist = {start: 0}
    q = deque([start])
    farthest, max_d = start, 0
    while q:
        u = q.popleft()
        for nb in adj[u]:
            if nb in node_set and nb not in dist:
                dist[nb] = dist[u] + 1
                q.append(nb)
                if dist[nb] > max_d:
                    max_d = dist[nb]
                    farthest = nb
    return farthest, max_d


def _structure_metrics(nodes, adj, total_vol):
    """社区结构质量: density / conductance / diameter(纯 Python,无外部依赖)。"""
    node_set = set(int(v) for v in nodes)
    n = len(node_set)
    if n <= 1:
        return 0.0, 0.0, 0.0

    internal_ends = 0
    boundary = 0
    vol_in = 0
    for v in node_set:
        vol_in += len(adj[v])
        for nb in adj[v]:
            if nb in node_set:
                internal_ends += 1
            else:
                boundary += 1
    m = internal_ends // 2
    density = 2.0 * m / (n * (n - 1))

    vol_out = total_vol - vol_in
    denom = min(vol_in, vol_out)
    conductance = boundary / denom if denom > 0 else 0.0

    # 双向 BFS 近似直径: 任取一点 → BFS 找最远点 u → 从 u 再 BFS 找最远距离
    seed = next(iter(node_set))
    u, _ = _bfs_farthest(seed, node_set, adj)
    _, diameter = _bfs_farthest(u, node_set, adj)

    return density, conductance, float(diameter)


def _greedy_one(q, sims_q, avg, adj, w, max_iter):
    """对单个查询节点做贪心 frontier 扩展,密度峰值停止,返回社区节点 set。"""
    q = int(q)
    visited = {q}
    frontier = set(adj[q]) - visited
    cur_sum = float(sims_q[q])

    best_density = -np.inf
    best_comm = set(visited)

    for _ in range(max_iter):
        if not frontier:
            break
        cand_arr = np.array(list(frontier))
        scores = sims_q[cand_arr]
        best_node = int(cand_arr[np.argmax(scores)])

        visited.add(best_node)
        frontier.discard(best_node)
        cur_sum += float(sims_q[best_node])
        frontier.update(adj[best_node] - visited)

        c = len(visited)
        density = (cur_sum - c * avg) / (c ** w)
        if density > best_density:
            best_density = density
            best_comm = set(visited)
        else:
            break
    return best_comm


def _greedy_expand_trace(q, sims_q, adj, max_iter,
                         frontier_batch_size=1, connectivity_boost=0.0,
                         init_seed_size=1, init_seed_hops=1,
                         init_seed_conn_beta=0.3,
                         init_seed_min_sim=None,
                         hse_high_order_beta=0.0,
                         hse_comm_cohesion_beta=0.0,
                         hse_boundary_gamma=0.0,
                         hse_pool_size=0,
                         hse_comm_direct_beta=0.0,
                         hse_normalize=False,
                         hse_density=False,
                         hse_density_alpha=1.0,
                         idbr_bridge_scores=None,
                         idbr_bridge_beta=0.0,
                         idbr_local_seed_beta=0.0,
                         idbr_boundary_penalty_gamma=0.0,
                         idbr_boundary_penalty_sim_gamma=0.5,
                         idbr_boundary_penalty_cohesion_gamma=0.5,
                         idbr_boundary_penalty_bridge_gamma=1.0,
                         early_stop_w=None,
                         early_stop_avg=None,
                         early_stop_patience=0,
                         early_stop_min_gain_tol=0.0,
                         early_stop_min_size=1):
    """
    单次贪心扩展: 从 frontier 中按相似度/结构连接度选择节点, 记录累计 sim 和。
    HSE: 高阶可达 + 社区凝聚 + 边界惩罚 可选加入候选排序。
    """
    q = int(q)
    frontier_batch_size = max(1, int(frontier_batch_size))
    connectivity_boost = max(0.0, float(connectivity_boost))
    init_seed_size = max(1, int(init_seed_size))
    init_seed_hops = max(1, int(init_seed_hops))
    init_seed_conn_beta = max(0.0, float(init_seed_conn_beta))
    hse_high_order_beta = max(0.0, float(hse_high_order_beta))
    hse_comm_cohesion_beta = max(0.0, float(hse_comm_cohesion_beta))
    hse_boundary_gamma = max(0.0, float(hse_boundary_gamma))
    hse_pool_size = max(0, int(hse_pool_size))
    hse_comm_direct_beta = max(0.0, float(hse_comm_direct_beta))
    hse_normalize = bool(hse_normalize)
    hse_density = bool(hse_density)
    hse_density_alpha = max(0.0, float(hse_density_alpha))
    idbr_bridge_beta = max(0.0, float(idbr_bridge_beta))
    use_idbr_bridge = idbr_bridge_beta > 0
    idbr_boundary_penalty_gamma = max(0.0, float(idbr_boundary_penalty_gamma))
    idbr_boundary_penalty_sim_gamma = max(0.0, float(idbr_boundary_penalty_sim_gamma))
    idbr_boundary_penalty_cohesion_gamma = max(0.0, float(idbr_boundary_penalty_cohesion_gamma))
    idbr_boundary_penalty_bridge_gamma = max(0.0, float(idbr_boundary_penalty_bridge_gamma))
    use_idbr_boundary_penalty = (idbr_boundary_penalty_gamma > 0
                                 or idbr_boundary_penalty_sim_gamma > 0
                                 or idbr_boundary_penalty_cohesion_gamma > 0
                                 or idbr_boundary_penalty_bridge_gamma > 0)
    use_hse = (hse_high_order_beta > 0 or hse_comm_cohesion_beta > 0
               or hse_boundary_gamma > 0 or hse_comm_direct_beta > 0)
    visited = {q}
    frontier = set(adj[q]) - visited
    cur_sum = float(sims_q[q])

    node_order = [q]
    cum_sims = [cur_sum]
    added = 0
    community_halo = set(adj[q]) | {q} if use_hse else None
    use_early_stop = early_stop_w is not None and early_stop_avg is not None
    early_stop_patience = max(0, int(early_stop_patience))
    early_stop_min_gain_tol = max(0.0, float(early_stop_min_gain_tol))
    early_stop_min_size = max(1, int(early_stop_min_size))
    early_best_density = None
    early_bad_steps = 0

    def _utility_map(cand_arr, scores):
        if not hse_density:
            return {}
        raw_scores = sims_q[cand_arr].astype(np.float64, copy=False)
        adjust = scores.astype(np.float64, copy=False) - raw_scores
        adjust = adjust - float(adjust.mean()) if adjust.size > 0 else adjust
        utilities = raw_scores + hse_density_alpha * adjust
        return {int(c): float(u) for c, u in zip(cand_arr.tolist(), utilities.tolist())}

    def _node_utility(node, utilities):
        if not hse_density:
            return float(sims_q[node])
        return float(utilities.get(int(node), sims_q[node]))

    def _should_stop_trace():
        nonlocal early_best_density, early_bad_steps
        if not use_early_stop or len(cum_sims) < early_stop_min_size:
            return False
        size = float(len(cum_sims))
        density = (cum_sims[-1] - size * float(early_stop_avg)) / (size ** float(early_stop_w))
        if early_best_density is None or density > early_best_density:
            early_best_density = density
            early_bad_steps = 0
            return False
        if early_stop_min_gain_tol > 0 and early_best_density - density <= early_stop_min_gain_tol:
            return False
        early_bad_steps += 1
        return early_bad_steps > early_stop_patience

    if init_seed_size > 1:
        seed_target = min(init_seed_size - 1, max_iter)
        for _ in range(init_seed_hops):
            if added >= seed_target or not frontier:
                break
            cand_arr = np.array(list(frontier))
            scores = sims_q[cand_arr].astype(np.float64, copy=True)
            if init_seed_min_sim is not None:
                keep = scores >= float(init_seed_min_sim)
                cand_arr = cand_arr[keep]
                scores = scores[keep]
                if cand_arr.size == 0:
                    break
            if init_seed_conn_beta > 0:
                conn = np.array([
                    len(adj[int(c)] & visited) / max(1, len(adj[int(c)]))
                    for c in cand_arr
                ], dtype=np.float64)
                scores = scores + init_seed_conn_beta * conn
            if use_idbr_local_seed:
                local_bridge = _idbr_local_bridge_candidate_scores(cand_arr, visited, adj)
                if local_bridge is not None:
                    scores = scores + idbr_local_seed_beta * local_bridge
            if use_idbr_bridge:
                bridge = _idbr_bridge_candidate_scores(cand_arr, idbr_bridge_scores)
                if bridge is not None:
                    scores = scores + idbr_bridge_beta * bridge
            if use_idbr_boundary_penalty:
                boundary_pen = _idbr_boundary_penalty_scores(
                    cand_arr, visited, adj, community_halo, sims_q=sims_q,
                    bridge_scores=idbr_bridge_scores,
                    sim_gamma=idbr_boundary_penalty_sim_gamma,
                    cohesion_gamma=idbr_boundary_penalty_cohesion_gamma,
                    bridge_gamma=idbr_boundary_penalty_bridge_gamma)
                if boundary_pen is not None:
                    scores = scores - idbr_boundary_penalty_gamma * boundary_pen
            if use_hse:
                cand_arr, scores = _limit_hse_pool(cand_arr, scores, hse_pool_size)
                hse_scores = _hse_candidate_scores(
                    q, cand_arr, visited, adj, community_halo,
                    hse_high_order_beta, hse_comm_cohesion_beta,
                    hse_boundary_gamma, hse_comm_direct_beta,
                    normalize=hse_normalize)
                scores = scores + hse_scores
            selected_utilities = _utility_map(cand_arr, scores)
            take = min(seed_target - added, cand_arr.size)
            if take <= 0:
                break
            if take == 1:
                chosen = [int(cand_arr[np.argmax(scores)])]
            else:
                chosen_idx = np.argpartition(-scores, take - 1)[:take]
                chosen_idx = chosen_idx[np.argsort(-scores[chosen_idx])]
                chosen = [int(cand_arr[i]) for i in chosen_idx]
            new_nodes = []
            for node in chosen:
                if node in visited:
                    continue
                visited.add(node)
                frontier.discard(node)
                cur_sum += _node_utility(node, selected_utilities)
                node_order.append(node)
                cum_sims.append(cur_sum)
                added += 1
                new_nodes.append(node)
                if added >= seed_target:
                    break
            for node in new_nodes:
                frontier.update(adj[node] - visited)
                if use_hse:
                    community_halo.update(adj[node])
                    community_halo.add(node)

    while added < max_iter and frontier:
        cand_arr = np.array(list(frontier))
        scores = sims_q[cand_arr].astype(np.float64, copy=True)

        if connectivity_boost > 0:
            conn = np.array([
                len(adj[int(c)] & visited) / max(1, len(adj[int(c)]))
                for c in cand_arr
            ], dtype=np.float64)
            span = float(conn.max() - conn.min()) if conn.size > 0 else 0.0
            if span > 1e-12:
                conn = (conn - conn.min()) / span
            scores = scores + connectivity_boost * conn

        if use_idbr_bridge:
            bridge = _idbr_bridge_candidate_scores(cand_arr, idbr_bridge_scores)
            if bridge is not None:
                scores = scores + idbr_bridge_beta * bridge

        if use_idbr_boundary_penalty:
            boundary_pen = _idbr_boundary_penalty_scores(
                cand_arr, visited, adj, community_halo, sims_q=sims_q,
                bridge_scores=idbr_bridge_scores,
                sim_gamma=idbr_boundary_penalty_sim_gamma,
                cohesion_gamma=idbr_boundary_penalty_cohesion_gamma,
                bridge_gamma=idbr_boundary_penalty_bridge_gamma)
            if boundary_pen is not None:
                scores = scores - idbr_boundary_penalty_gamma * boundary_pen

        if use_hse:
            cand_arr, scores = _limit_hse_pool(cand_arr, scores, hse_pool_size)
            hse_scores = _hse_candidate_scores(
                q, cand_arr, visited, adj, community_halo,
                hse_high_order_beta, hse_comm_cohesion_beta,
                hse_boundary_gamma, hse_comm_direct_beta,
                normalize=hse_normalize)
            scores = scores + hse_scores
        selected_utilities = _utility_map(cand_arr, scores)

        take = min(frontier_batch_size, len(cand_arr), max_iter - added)
        if take <= 0:
            break
        if take == 1:
            chosen = [int(cand_arr[np.argmax(scores)])]
        else:
            chosen_idx = np.argpartition(-scores, take - 1)[:take]
            chosen_idx = chosen_idx[np.argsort(-scores[chosen_idx])]
            chosen = [int(cand_arr[i]) for i in chosen_idx]

        stop_trace = False
        for node in chosen:
            if node in visited:
                continue
            visited.add(node)
            frontier.discard(node)
            cur_sum += _node_utility(node, selected_utilities)
            frontier.update(adj[node] - visited)
            if use_hse:
                community_halo.update(adj[node])
                community_halo.add(node)
            node_order.append(node)
            cum_sims.append(cur_sum)
            added += 1
            if _should_stop_trace():
                stop_trace = True
                break
            if added >= max_iter:
                break
        if stop_trace:
            break

    return node_order, np.array(cum_sims, dtype=np.float64)


def _best_community_for_w(node_order, cum_sims, avg, w,
                          patience=0, min_gain_tol=0.0,
                          select_mode='first_drop', min_size=1):
    """在一条扩展轨迹上, 对给定 w 找密度峰值, 返回对应社区 set。"""
    sizes = np.arange(1, len(cum_sims) + 1, dtype=np.float64)
    densities = (cum_sims - sizes * avg) / (sizes ** w)
    select_mode = str(select_mode).lower()
    min_idx = min(max(0, int(min_size) - 1), len(cum_sims) - 1)
    if select_mode == 'global':
        rel_best = int(np.argmax(densities[min_idx:]))
        best_idx = min_idx + rel_best
        return set(node_order[:best_idx + 1])

    patience = max(0, int(patience))
    min_gain_tol = max(0.0, float(min_gain_tol))
    best_idx = min_idx
    best_d = densities[min_idx]
    bad_steps = 0
    for i in range(min_idx + 1, len(densities)):
        d = densities[i]
        if d > best_d:
            best_d = d
            best_idx = i
            bad_steps = 0
        elif min_gain_tol > 0 and best_d - d <= min_gain_tol:
            continue
        else:
            bad_steps += 1
            if bad_steps > patience:
                break
    return set(node_order[:best_idx + 1])


def community_search_greedy(embeddings, data, w_list=(0.0, 0.1, 0.2, 0.3, 0.5),
                            num_queries=None, seed=0, max_iter=10000,
                            compute_structure=False,
                            node_boost=None, boost_factor=1.5, queries=None,
                            greedy_patience=0, greedy_min_gain_tol=0.0,
                            frontier_batch_size=1, include_query_in_pred=False,
                            greedy_connectivity_boost=0.0,
                            greedy_select_mode='first_drop',
                            greedy_init_seed_size=1,
                            greedy_init_seed_hops=1,
                            greedy_init_seed_conn_beta=0.3,
                            greedy_init_seed_min_sim=None,
                            hse_high_order_beta=0.0,
                            hse_comm_cohesion_beta=0.0,
                            hse_boundary_gamma=0.0,
                            hse_pool_size=0,
                            hse_comm_direct_beta=0.0,
                            hse_normalize=False,
                            hse_density=False,
                            hse_density_alpha=1.0,
                            idbr_bridge_beta=0.0,
                            idbr_bridge_steps=(2, 4),
                            idbr_bridge_decay=1.0,
                            idbr_bridge_max_states=50000,
                            idbr_bridge_fanout=0,
                            idbr_local_seed_beta=0.0,
                            idbr_boundary_penalty_gamma=0.0,
                            idbr_boundary_penalty_sim_gamma=0.5,
                            idbr_boundary_penalty_cohesion_gamma=0.5,
                            idbr_boundary_penalty_bridge_gamma=1.0,
                            recall_expand_size=0,
                            recall_expand_min_sim_delta=0.0):
    """
    贪心 + 密度自适应的社区搜索评估。

    优化: 每个查询只做一次扩展, 多个 w 在同一条轨迹上各自找密度峰值(省 ×|w_list| 倍)。
    compute_structure=False(默认)跳过 density/conductance/diameter, 大幅提速。

    Returns:
        dict: {w: {'precision','recall','f1','jaccard','avg_size', ...}}
    """
    y = data.y.detach().cpu().numpy()
    emb = F.normalize(embeddings.detach().cpu(), dim=-1)
    N = emb.shape[0]

    if queries is not None:
        queries = np.asarray(queries)
    elif num_queries is None or num_queries >= N:
        queries = np.arange(N)
    else:
        rng = np.random.default_rng(seed)
        queries = rng.choice(N, size=num_queries, replace=False)

    sims = (emb @ emb.t()).numpy()

    mult = _boost_multiplier(node_boost, boost_factor, N)
    if mult is not None:
        sims = sims * mult[None, :]

    adj = _build_adj_list(_cs_edge_index(data), N)
    total_vol = sum(len(a) for a in adj) if compute_structure else 0

    label_sets = {}
    for label in np.unique(y):
        label_sets[label] = set(np.where(y == label)[0].tolist())

    # 初始化每个 w 的累加器
    accum = {w: {'P': [], 'R': [], 'F': [], 'J': [], 'sizes': [],
                 'Den': [], 'Con': [], 'Dia': []} for w in w_list}

    for q in queries:
        truth = label_sets[y[q]].copy()
        if not include_query_in_pred:
            truth.discard(int(q))
        if len(truth) == 0:
            continue

        sims_q = sims[q]
        avg = float(sims_q.mean())
        density_avg = avg
        bridge_scores = _idbr_diffusion_bridge(
            q, adj, steps=idbr_bridge_steps,
            decay=idbr_bridge_decay,
            max_states=idbr_bridge_max_states,
            sims_q=sims_q,
            fanout=idbr_bridge_fanout) if idbr_bridge_beta > 0 else None
        early_w = w_list[0] if len(w_list) == 1 and str(greedy_select_mode).lower() == 'first_drop' else None

        # 只扩展一次
        node_order, cum_sims = _greedy_expand_trace(
            q, sims_q, adj, max_iter,
            frontier_batch_size=frontier_batch_size,
            connectivity_boost=greedy_connectivity_boost,
            init_seed_size=greedy_init_seed_size,
            init_seed_hops=greedy_init_seed_hops,
            init_seed_conn_beta=greedy_init_seed_conn_beta,
            init_seed_min_sim=greedy_init_seed_min_sim,
            hse_high_order_beta=hse_high_order_beta,
            hse_comm_cohesion_beta=hse_comm_cohesion_beta,
            hse_boundary_gamma=hse_boundary_gamma,
            hse_pool_size=hse_pool_size,
            hse_comm_direct_beta=hse_comm_direct_beta,
            hse_normalize=hse_normalize,
            hse_density=hse_density,
            hse_density_alpha=hse_density_alpha,
            idbr_bridge_scores=bridge_scores,
            idbr_bridge_beta=idbr_bridge_beta,
            idbr_local_seed_beta=idbr_local_seed_beta,
            idbr_boundary_penalty_gamma=idbr_boundary_penalty_gamma,
            idbr_boundary_penalty_sim_gamma=idbr_boundary_penalty_sim_gamma,
            idbr_boundary_penalty_cohesion_gamma=idbr_boundary_penalty_cohesion_gamma,
            idbr_boundary_penalty_bridge_gamma=idbr_boundary_penalty_bridge_gamma,
            early_stop_w=early_w,
            early_stop_avg=density_avg,
            early_stop_patience=greedy_patience,
            early_stop_min_gain_tol=greedy_min_gain_tol,
            early_stop_min_size=greedy_init_seed_size)

        for w in w_list:
            comm = _best_community_for_w(
                node_order, cum_sims, density_avg, w,
                patience=greedy_patience,
                min_gain_tol=greedy_min_gain_tol,
                select_mode=greedy_select_mode,
                min_size=greedy_init_seed_size)
            if recall_expand_size > 0:
                comm = _recall_expand_community(
                    q, comm, sims_q, adj,
                    max_add=recall_expand_size,
                    pool_size=hse_pool_size,
                    min_sim=avg + float(recall_expand_min_sim_delta),
                    high_order_beta=hse_high_order_beta,
                    comm_direct_beta=hse_comm_direct_beta,
                    comm_cohesion_beta=hse_comm_cohesion_beta,
                    boundary_gamma=hse_boundary_gamma,
                    hse_normalize=hse_normalize)
            pred = set(comm)
            if not include_query_in_pred:
                pred.discard(int(q))

            inter = len(pred & truth)
            union = len(pred | truth)
            p = inter / len(pred) if len(pred) > 0 else 0.0
            r = inter / len(truth)
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            j = inter / union if union > 0 else 0.0

            a = accum[w]
            a['P'].append(p); a['R'].append(r); a['F'].append(f); a['J'].append(j)
            a['sizes'].append(len(comm))

            if compute_structure:
                den, con, dia = _structure_metrics(comm, adj, total_vol)
                a['Den'].append(den); a['Con'].append(con); a['Dia'].append(dia)

    results = {}
    for w in w_list:
        a = accum[w]
        results[w] = {
            'precision': float(np.mean(a['P'])) * 100 if a['P'] else 0.0,
            'recall': float(np.mean(a['R'])) * 100 if a['R'] else 0.0,
            'f1': float(np.mean(a['F'])) * 100 if a['F'] else 0.0,
            'jaccard': float(np.mean(a['J'])) * 100 if a['J'] else 0.0,
            'avg_size': float(np.mean(a['sizes'])) if a['sizes'] else 0.0,
            'density': float(np.mean(a['Den'])) if a['Den'] else 0.0,
            'conductance': float(np.mean(a['Con'])) if a['Con'] else 0.0,
            'diameter': float(np.mean(a['Dia'])) if a['Dia'] else 0.0,
        }
        extra = ''
        if compute_structure:
            extra = (f" den={results[w]['density']:.3f}"
                     f" cond={results[w]['conductance']:.3f}"
                     f" diam={results[w]['diameter']:.2f}")
        print(f'[CS-greedy] w={w:<4} '
              f"P={results[w]['precision']:.2f} "
              f"R={results[w]['recall']:.2f} "
              f"F1={results[w]['f1']:.2f} "
              f"Jaccard={results[w]['jaccard']:.2f} "
              f"size={results[w]['avg_size']:.1f}{extra}")
    return results


def community_search_rl(builder, embeddings, data, queries,
                        node_boost=None, intent=None, max_sizes=None,
                        oracle_size=False):
    """
    Actor-Critic 社区搜索评测 (§7.2 Step4)。
    对每个查询, 用训练好的 builder 从查询节点扩展出社区集合, 再与真值标签比对。
    标签仅用于计算 P/R/F1 (评测), 社区生成过程本身自监督、不看标签。

    max_sizes=None: 单 size 评测(用 builder.max_size), 返回扁平 dict。
    max_sizes=[...]: 一次扫多个 max_size(P-R 曲线), 每个查询只展开一次到最大 cap、
        前缀切片还原各 size, 返回 {size: dict}。

    Returns:
        dict: {'precision','recall','f1','jaccard','avg_size'} 或 {size: 同结构}
    """
    y = data.y.detach().cpu().numpy()
    N = embeddings.size(0)
    adj = _build_adj_list(_cs_edge_index(data), N)

    label_sets = {}
    for label in np.unique(y):
        label_sets[label] = set(np.where(y == label)[0].tolist())

    queries = np.asarray(queries)

    def _oracle_eval():
        """oracle-size 对照线: 每个查询扩展到其真值社区大小 |truth| 并截断。
        隔离 '排序质量'(embedding 能否把同社区节点排前) 与 '停止准则'
        (贪婪/AC 何时停)。oracle F1 高而实际 F1 低 => 瓶颈在扩展/停止, 非表示。"""
        caps = {int(q): len(label_sets[y[q]]) for q in queries}
        global_cap = max(caps.values()) if caps else 0
        orders_o = {int(q): builder.build_sequence(
                        embeddings, adj, int(q), intent,
                        node_boost=node_boost, max_size=global_cap)
                    for q in queries}
        P, R, Fm, J, sizes = [], [], [], [], []
        for q in queries:
            truth = label_sets[y[q]].copy()
            truth.discard(int(q))
            if len(truth) == 0:
                continue
            # 序列首元素为 q; 取 |truth|+1 个 => 去掉 q 后恰好 |truth| 个候选
            seq = orders_o[int(q)][:len(truth) + 1]
            pred = set(seq)
            pred.discard(int(q))
            inter = len(pred & truth)
            union = len(pred | truth)
            p = inter / len(pred) if len(pred) > 0 else 0.0
            r = inter / len(truth)
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            j = inter / union if union > 0 else 0.0
            P.append(p); R.append(r); Fm.append(f); J.append(j)
            sizes.append(len(pred))
        res = {
            'precision': float(np.mean(P)) * 100 if P else 0.0,
            'recall': float(np.mean(R)) * 100 if R else 0.0,
            'f1': float(np.mean(Fm)) * 100 if Fm else 0.0,
            'jaccard': float(np.mean(J)) * 100 if J else 0.0,
            'avg_size': float(np.mean(sizes)) if sizes else 0.0,
        }
        print(f"[CS-rl] oracle-size  P={res['precision']:.2f} "
              f"R={res['recall']:.2f} F1={res['f1']:.2f} "
              f"Jaccard={res['jaccard']:.2f} size={res['avg_size']:.1f}")
        return res

    # ---- 扫多个 max_size: 每个查询只展开一次到最大 cap, 前缀切片还原各 size ----
    if max_sizes is not None:
        max_cap = max(max_sizes)
        orders = {int(q): builder.build_sequence(
                      embeddings, adj, int(q), intent,
                      node_boost=node_boost, max_size=max_cap)
                  for q in queries}
        sweep = {}
        for ms in sorted(max_sizes):
            P, R, Fm, J, sizes = [], [], [], [], []
            for q in queries:
                truth = label_sets[y[q]].copy()
                truth.discard(int(q))
                if len(truth) == 0:
                    continue
                seq = orders[int(q)][:ms]
                pred = set(seq)
                pred.discard(int(q))
                inter = len(pred & truth)
                union = len(pred | truth)
                p = inter / len(pred) if len(pred) > 0 else 0.0
                r = inter / len(truth)
                f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                j = inter / union if union > 0 else 0.0
                P.append(p); R.append(r); Fm.append(f); J.append(j)
                sizes.append(len(seq))
            res = {
                'precision': float(np.mean(P)) * 100 if P else 0.0,
                'recall': float(np.mean(R)) * 100 if R else 0.0,
                'f1': float(np.mean(Fm)) * 100 if Fm else 0.0,
                'jaccard': float(np.mean(J)) * 100 if J else 0.0,
                'avg_size': float(np.mean(sizes)) if sizes else 0.0,
            }
            sweep[ms] = res
            print(f"[CS-rl] max_size={ms:5d}  P={res['precision']:.2f} "
                  f"R={res['recall']:.2f} F1={res['f1']:.2f} "
                  f"Jaccard={res['jaccard']:.2f} size={res['avg_size']:.1f}")
        if oracle_size:
            sweep['oracle'] = _oracle_eval()
        return sweep

    P, R, Fm, J, sizes = [], [], [], [], []

    for q in queries:
        truth = label_sets[y[q]].copy()
        truth.discard(int(q))
        if len(truth) == 0:
            continue
        comm = builder.build(embeddings, adj, int(q), intent,
                             node_boost=node_boost, greedy=True)
        pred = set(comm)
        pred.discard(int(q))

        inter = len(pred & truth)
        union = len(pred | truth)
        p = inter / len(pred) if len(pred) > 0 else 0.0
        r = inter / len(truth)
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        j = inter / union if union > 0 else 0.0
        P.append(p); R.append(r); Fm.append(f); J.append(j)
        sizes.append(len(comm))

    results = {
        'precision': float(np.mean(P)) * 100 if P else 0.0,
        'recall': float(np.mean(R)) * 100 if R else 0.0,
        'f1': float(np.mean(Fm)) * 100 if Fm else 0.0,
        'jaccard': float(np.mean(J)) * 100 if J else 0.0,
        'avg_size': float(np.mean(sizes)) if sizes else 0.0,
    }
    print(f"[CS-rl] P={results['precision']:.2f} "
          f"R={results['recall']:.2f} "
          f"F1={results['f1']:.2f} "
          f"Jaccard={results['jaccard']:.2f} "
          f"size={results['avg_size']:.1f}")
    if oracle_size:
        results['oracle'] = _oracle_eval()
    return results


def community_search_dynamic(encoder_fn, intent_generator, data, edge_weight,
                              topk=(10, 20, 50, 'oracle'), num_queries=200, seed=0,
                              node_boost=None, boost_factor=1.5, queries=None,
                              edge_index=None,
                              intent_proj_fn=None, intent_rerank_alpha=0.0):
    """
    动态意图社区搜索: 每个查询节点根据自己的特征生成意图,
    用该意图重新编码全图, 再按余弦相似度排序。

    比静态版慢 (每查询一次 forward), 但意图真正按查询适配。
    """
    y = data.y.detach().cpu().numpy()
    N = data.x.size(0)

    if queries is not None:
        queries = np.asarray(queries)
    elif num_queries is None or num_queries >= N:
        queries = np.arange(N)
    else:
        rng_np = np.random.default_rng(seed)
        queries = rng_np.choice(N, size=num_queries, replace=False)

    label_sets = {}
    for label in np.unique(y):
        label_sets[label] = set(np.where(y == label)[0].tolist())

    mult = _boost_multiplier(node_boost, boost_factor, N)

    per_k = {k: {'P': [], 'R': [], 'F': [], 'J': []} for k in topk}

    for q in queries:
        truth = label_sets[y[q]].copy()
        truth.discard(int(q))
        if len(truth) == 0:
            continue

        with torch.no_grad():
            intent = intent_generator(data.x[q])
            ei = data.edge_index if edge_index is None else edge_index
            emb = encoder_fn(data.x, ei, edge_weight, intent)
            emb_norm = F.normalize(emb, dim=-1)
            sims_q = (emb_norm[q] @ emb_norm.t()).cpu().numpy()
            if intent_rerank_alpha > 0 and intent_proj_fn is not None:
                z_proj = F.normalize(intent_proj_fn(emb), dim=-1)
                iq_norm = F.normalize(intent.unsqueeze(0), dim=-1)
                intent_align = (z_proj @ iq_norm.t()).squeeze(-1).cpu().numpy()
                sims_q = sims_q + intent_rerank_alpha * intent_align

        if mult is not None:
            sims_q = sims_q * mult

        order = np.argsort(-sims_q)
        order = order[order != q]

        for k in topk:
            actual_k = len(truth) if k == 'oracle' else k
            pred = set(order[:actual_k].tolist())

            inter = len(pred & truth)
            union = len(pred | truth)
            p = inter / len(pred) if len(pred) > 0 else 0.0
            r = inter / len(truth)
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            j = inter / union if union > 0 else 0.0
            per_k[k]['P'].append(p); per_k[k]['R'].append(r)
            per_k[k]['F'].append(f); per_k[k]['J'].append(j)

    results = {}
    for k in topk:
        a = per_k[k]
        k_label = 'oracle' if k == 'oracle' else str(k)
        results[k] = {
            'precision': float(np.mean(a['P'])) * 100,
            'recall': float(np.mean(a['R'])) * 100,
            'f1': float(np.mean(a['F'])) * 100,
            'jaccard': float(np.mean(a['J'])) * 100,
        }
        print(f'[CS-dyn] k={k_label:<7s} '
              f"P={results[k]['precision']:.2f} "
              f"R={results[k]['recall']:.2f} "
              f"F1={results[k]['f1']:.2f} "
              f"Jaccard={results[k]['jaccard']:.2f}")
    return results


def community_search_greedy_dynamic(encoder_fn, intent_generator, data, edge_weight,
                                     w_list=(0.0, 0.1, 0.2, 0.3, 0.5),
                                     num_queries=200, seed=0, max_iter=10000,
                                     compute_structure=False,
                                     node_boost=None, boost_factor=1.5, queries=None,
                                     edge_index=None,
                                     intent_proj_fn=None, intent_rerank_alpha=0.0,
                                     greedy_patience=0, greedy_min_gain_tol=0.0,
                                     frontier_batch_size=1, include_query_in_pred=False,
                                     greedy_connectivity_boost=0.0,
                                     greedy_select_mode='first_drop',
                                     greedy_init_seed_size=1,
                                     greedy_init_seed_hops=1,
                                     greedy_init_seed_conn_beta=0.3,
                                     greedy_init_seed_min_sim=None,
                                     hse_high_order_beta=0.0,
                                     hse_comm_cohesion_beta=0.0,
                                     hse_boundary_gamma=0.0,
                                     hse_pool_size=0,
                                     hse_comm_direct_beta=0.0,
                                     hse_normalize=False,
                                     hse_density=False,
                                     hse_density_alpha=1.0,
                                     idbr_bridge_beta=0.0,
                                     idbr_bridge_steps=(2, 4),
                                     idbr_bridge_decay=1.0,
                                     idbr_bridge_max_states=50000,
                                     idbr_bridge_fanout=0,
                                     idbr_local_seed_beta=0.0,
                                     idbr_boundary_penalty_gamma=0.0,
                                     idbr_boundary_penalty_sim_gamma=0.5,
                                     idbr_boundary_penalty_cohesion_gamma=0.5,
                                     idbr_boundary_penalty_bridge_gamma=1.0,
                                     recall_expand_size=0,
                                     recall_expand_min_sim_delta=0.0):
    """
    动态意图 + 贪心扩展社区搜索。
    每个查询节点生成意图→重新编码→贪心扩展。
    """
    y = data.y.detach().cpu().numpy()
    N = data.x.size(0)

    if queries is not None:
        queries = np.asarray(queries)
    elif num_queries is None or num_queries >= N:
        queries = np.arange(N)
    else:
        rng_np = np.random.default_rng(seed)
        queries = rng_np.choice(N, size=num_queries, replace=False)

    adj = _build_adj_list(_cs_edge_index(data), N)
    total_vol = sum(len(a) for a in adj) if compute_structure else 0

    label_sets = {}
    for label in np.unique(y):
        label_sets[label] = set(np.where(y == label)[0].tolist())

    mult = _boost_multiplier(node_boost, boost_factor, N)

    accum = {w: {'P': [], 'R': [], 'F': [], 'J': [], 'sizes': [],
                 'Den': [], 'Con': [], 'Dia': []} for w in w_list}

    for q in queries:
        truth = label_sets[y[q]].copy()
        if not include_query_in_pred:
            truth.discard(int(q))
        if len(truth) == 0:
            continue

        with torch.no_grad():
            intent = intent_generator(data.x[q])
            ei = data.edge_index if edge_index is None else edge_index
            emb = encoder_fn(data.x, ei, edge_weight, intent)
            emb_norm = F.normalize(emb, dim=-1)
            sims_q = (emb_norm[q] @ emb_norm.t()).cpu().numpy()
            if intent_rerank_alpha > 0 and intent_proj_fn is not None:
                z_proj = F.normalize(intent_proj_fn(emb), dim=-1)
                iq_norm = F.normalize(intent.unsqueeze(0), dim=-1)
                intent_align = (z_proj @ iq_norm.t()).squeeze(-1).cpu().numpy()
                sims_q = sims_q + intent_rerank_alpha * intent_align

        if mult is not None:
            sims_q = sims_q * mult

        avg = float(sims_q.mean())
        density_avg = avg
        bridge_scores = _idbr_diffusion_bridge(
            q, adj, steps=idbr_bridge_steps,
            decay=idbr_bridge_decay,
            max_states=idbr_bridge_max_states,
            sims_q=sims_q,
            fanout=idbr_bridge_fanout) if idbr_bridge_beta > 0 else None
        early_w = w_list[0] if len(w_list) == 1 and str(greedy_select_mode).lower() == 'first_drop' else None
        node_order, cum_sims = _greedy_expand_trace(
            q, sims_q, adj, max_iter,
            frontier_batch_size=frontier_batch_size,
            connectivity_boost=greedy_connectivity_boost,
            init_seed_size=greedy_init_seed_size,
            init_seed_hops=greedy_init_seed_hops,
            init_seed_conn_beta=greedy_init_seed_conn_beta,
            init_seed_min_sim=greedy_init_seed_min_sim,
            hse_high_order_beta=hse_high_order_beta,
            hse_comm_cohesion_beta=hse_comm_cohesion_beta,
            hse_boundary_gamma=hse_boundary_gamma,
            hse_pool_size=hse_pool_size,
            hse_comm_direct_beta=hse_comm_direct_beta,
            hse_normalize=hse_normalize,
            hse_density=hse_density,
            hse_density_alpha=hse_density_alpha,
            idbr_bridge_scores=bridge_scores,
            idbr_bridge_beta=idbr_bridge_beta,
            idbr_local_seed_beta=idbr_local_seed_beta,
            idbr_boundary_penalty_gamma=idbr_boundary_penalty_gamma,
            idbr_boundary_penalty_sim_gamma=idbr_boundary_penalty_sim_gamma,
            idbr_boundary_penalty_cohesion_gamma=idbr_boundary_penalty_cohesion_gamma,
            idbr_boundary_penalty_bridge_gamma=idbr_boundary_penalty_bridge_gamma,
            early_stop_w=early_w,
            early_stop_avg=density_avg,
            early_stop_patience=greedy_patience,
            early_stop_min_gain_tol=greedy_min_gain_tol,
            early_stop_min_size=greedy_init_seed_size)

        for w in w_list:
            comm = _best_community_for_w(
                node_order, cum_sims, density_avg, w,
                patience=greedy_patience,
                min_gain_tol=greedy_min_gain_tol,
                select_mode=greedy_select_mode,
                min_size=greedy_init_seed_size)
            if recall_expand_size > 0:
                comm = _recall_expand_community(
                    q, comm, sims_q, adj,
                    max_add=recall_expand_size,
                    pool_size=hse_pool_size,
                    min_sim=avg + float(recall_expand_min_sim_delta),
                    high_order_beta=hse_high_order_beta,
                    comm_direct_beta=hse_comm_direct_beta,
                    comm_cohesion_beta=hse_comm_cohesion_beta,
                    boundary_gamma=hse_boundary_gamma,
                    hse_normalize=hse_normalize)
            pred = set(comm)
            if not include_query_in_pred:
                pred.discard(int(q))

            inter = len(pred & truth)
            union = len(pred | truth)
            p = inter / len(pred) if len(pred) > 0 else 0.0
            r = inter / len(truth)
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            j = inter / union if union > 0 else 0.0

            a = accum[w]
            a['P'].append(p); a['R'].append(r); a['F'].append(f); a['J'].append(j)
            a['sizes'].append(len(comm))
            if compute_structure:
                den, con, dia = _structure_metrics(comm, adj, total_vol)
                a['Den'].append(den); a['Con'].append(con); a['Dia'].append(dia)

    results = {}
    for w in w_list:
        a = accum[w]
        results[w] = {
            'precision': float(np.mean(a['P'])) * 100 if a['P'] else 0.0,
            'recall': float(np.mean(a['R'])) * 100 if a['R'] else 0.0,
            'f1': float(np.mean(a['F'])) * 100 if a['F'] else 0.0,
            'jaccard': float(np.mean(a['J'])) * 100 if a['J'] else 0.0,
            'avg_size': float(np.mean(a['sizes'])) if a['sizes'] else 0.0,
            'density': float(np.mean(a['Den'])) if a['Den'] else 0.0,
            'conductance': float(np.mean(a['Con'])) if a['Con'] else 0.0,
            'diameter': float(np.mean(a['Dia'])) if a['Dia'] else 0.0,
        }
        extra = ''
        if compute_structure:
            extra = (f" den={results[w]['density']:.3f}"
                     f" cond={results[w]['conductance']:.3f}"
                     f" diam={results[w]['diameter']:.2f}")
        print(f'[CS-greedy-dyn] w={w:<4} '
              f"P={results[w]['precision']:.2f} "
              f"R={results[w]['recall']:.2f} "
              f"F1={results[w]['f1']:.2f} "
              f"Jaccard={results[w]['jaccard']:.2f} "
              f"size={results[w]['avg_size']:.1f}{extra}")
    return results

