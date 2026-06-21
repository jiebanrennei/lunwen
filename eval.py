
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


def community_search(embeddings, data, topk=(10, 20, 50), num_queries=None, seed=0):
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

    if num_queries is None or num_queries >= N:
        queries = np.arange(N)
    else:
        rng = np.random.default_rng(seed)
        queries = rng.choice(N, size=num_queries, replace=False)

    sims = (emb @ emb.t()).numpy()

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


def _build_adj_list(edge_index, num_nodes):
    """从 edge_index(评估时为双向边)构建邻接表 list[set[int]],去自环。"""
    adj = [set() for _ in range(num_nodes)]
    ei = edge_index.detach().cpu().numpy()
    for s, d in zip(ei[0].tolist(), ei[1].tolist()):
        if s != d:
            adj[s].add(d)
            adj[d].add(s)
    return adj


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


def _greedy_expand_trace(q, sims_q, adj, max_iter):
    """
    单次贪心扩展: 不断加入 frontier 里 sim 最高的邻居, 记录每步的累计 sim 和。
    返回 (node_order, cum_sims): node_order[i] 为第 i 步加入的节点,
    cum_sims[i] 为前 i+1 个节点的 sim 总和。不做任何 break-on-density,
    一路扩到 frontier 空或 max_iter。
    """
    q = int(q)
    visited = {q}
    frontier = set(adj[q]) - visited
    cur_sum = float(sims_q[q])

    node_order = [q]
    cum_sims = [cur_sum]

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

        node_order.append(best_node)
        cum_sims.append(cur_sum)

    return node_order, np.array(cum_sims, dtype=np.float64)


def _best_community_for_w(node_order, cum_sims, avg, w):
    """在一条扩展轨迹上, 对给定 w 找密度峰值, 返回对应社区 set。"""
    sizes = np.arange(1, len(cum_sims) + 1, dtype=np.float64)
    densities = (cum_sims - sizes * avg) / (sizes ** w)
    # 找密度首次下降的位置(贪心 break), 或取全局最大
    best_idx = 0
    best_d = densities[0]
    for i in range(1, len(densities)):
        if densities[i] > best_d:
            best_d = densities[i]
            best_idx = i
        else:
            break
    return set(node_order[:best_idx + 1])


def community_search_greedy(embeddings, data, w_list=(0.0, 0.1, 0.2, 0.3, 0.5),
                            num_queries=None, seed=0, max_iter=10000,
                            compute_structure=False):
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

    if num_queries is None or num_queries >= N:
        queries = np.arange(N)
    else:
        rng = np.random.default_rng(seed)
        queries = rng.choice(N, size=num_queries, replace=False)

    sims = (emb @ emb.t()).numpy()
    adj = _build_adj_list(data.edge_index, N)
    total_vol = sum(len(a) for a in adj) if compute_structure else 0

    label_sets = {}
    for label in np.unique(y):
        label_sets[label] = set(np.where(y == label)[0].tolist())

    # 初始化每个 w 的累加器
    accum = {w: {'P': [], 'R': [], 'F': [], 'J': [], 'sizes': [],
                 'Den': [], 'Con': [], 'Dia': []} for w in w_list}

    for q in queries:
        truth = label_sets[y[q]].copy()
        truth.discard(int(q))
        if len(truth) == 0:
            continue

        sims_q = sims[q]
        avg = float(sims_q.mean())

        # 只扩展一次
        node_order, cum_sims = _greedy_expand_trace(q, sims_q, adj, max_iter)

        for w in w_list:
            comm = _best_community_for_w(node_order, cum_sims, avg, w)
            pred = set(comm)
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

