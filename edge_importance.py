"""
创新点四: 对抗社区感知的边重要性学习 + 可疑节点识别

边重要性综合四个维度:
1. 拓扑矛盾性: 无直接边但共享异常邻居 (共同邻居分析)
2. 语义背离度: 语义相似但拓扑疏远 ("说得多, 连得少")
3. 意图相关性: 边与查询意图的相关程度
4. 时序异常: 节点时间戳偏离群体中心

可疑节点识别器: 边重要性聚合到节点 + 节点异常分 + 时序异常 -> Top-K。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree


def compute_temporal_anomaly(node_time, num_nodes, device=None):
    if node_time is None:
        return torch.zeros(num_nodes, device=device)
    if not torch.is_tensor(node_time):
        node_time = torch.as_tensor(node_time, dtype=torch.float32)
    else:
        node_time = node_time.to(dtype=torch.float32)
    if device is not None:
        node_time = node_time.to(device)
    node_time = node_time.reshape(-1)
    if node_time.numel() != num_nodes:
        return torch.zeros(num_nodes, device=node_time.device)
    valid = torch.isfinite(node_time)
    if not valid.any().item():
        return torch.zeros(num_nodes, device=node_time.device)
    valid_time = node_time[valid]
    eps = 1e-8
    median = valid_time.median()
    mad = (valid_time - median).abs().median()
    if float(mad) > eps:
        score = (node_time - median).abs() / (1.4826 * mad + eps)
    else:
        mean = valid_time.mean()
        std = valid_time.std(unbiased=False)
        if float(std) > eps:
            score = (node_time - mean).abs() / (std + eps)
        else:
            score = torch.zeros_like(node_time)
    score = torch.where(valid, score, torch.zeros_like(score))
    max_score = score.max()
    if float(max_score) > eps:
        score = score / (max_score + eps)
    return score


class AdversarialCommunityAwareEdgeImportance(nn.Module):
    def __init__(self, num_hidden, intent_dim, drop_p=0.1):
        super().__init__()

        # 维度一: 拓扑矛盾性 (端点特征编码 × 归一化共同邻居数)
        self.topo_mlp = nn.Sequential(
            nn.Linear(num_hidden * 2, num_hidden), nn.ReLU(),
            nn.Linear(num_hidden, 1)
        )
        # 维度二: 语义背离度
        self.sem_mlp = nn.Sequential(
            nn.Linear(1, num_hidden // 2), nn.ReLU(),
            nn.Linear(num_hidden // 2, 1)
        )
        # 维度三: 意图相关性
        self.intent_mlp = nn.Sequential(
            nn.Linear(num_hidden + intent_dim, num_hidden), nn.ReLU(),
            nn.Dropout(drop_p), nn.Linear(num_hidden, 1)
        )
        # 三维融合
        self.fuse = nn.Sequential(
            nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid()
        )
        # 共同邻居数只依赖固定图拓扑, 首次算好后缓存, 避免每 epoch 摊出 N×N 稠密邻接
        self._cn_norm = None
        self._cn_sig = None

    def _common_neighbor_norm(self, edge_index, num_nodes):
        """用稀疏矩阵乘法 A²[u,v] = u,v 的共同邻居数, 替代 Python 循环。

        原实现用 set 交集逐边遍历, 对 E 条边 × 平均度 d 是 O(E·d) 次 Python 操作;
        改为 torch.sparse.mm(A, A) + 二分查找, 全部在 C++/CUDA 层完成,
        在 ACM/IMDB_NEW 规模上可提速 2 个数量级。
        """
        src, dst = edge_index[0], edge_index[1]
        sig = (num_nodes, edge_index.shape[1], edge_index.data_ptr())
        if (self._cn_norm is not None and self._cn_sig == sig
                and self._cn_norm.device == edge_index.device):
            return self._cn_norm
        with torch.no_grad():
            dev = edge_index.device
            N = num_nodes
            E = src.size(0)
            if E == 0 or N == 0:
                cn_norm = torch.zeros(E, dtype=torch.float32, device=dev)
            else:
                # 去掉自环后再构造稀疏邻接, 与原 Python 实现的语义保持一致
                no_self = src != dst
                row, col = src[no_self], dst[no_self]
                E2 = row.size(0)
                if E2 == 0:
                    cn_norm = torch.zeros(E, dtype=torch.float32, device=dev)
                else:
                    # 去重: 原 Python 用 set 自动去重, 这里用 torch.unique 对齐语义
                    edge_keys_build = row * N + col
                    unique_keys = torch.unique(edge_keys_build)
                    unique_row = unique_keys // N
                    unique_col = unique_keys % N
                    idx = torch.stack([unique_row, unique_col], dim=0)
                    val = torch.ones(unique_keys.size(0), device=dev, dtype=torch.float32)
                    A = torch.sparse_coo_tensor(idx, val, (N, N)).coalesce()
                    # A²[u, v] = Σ_k A[u,k]·A[k,v] = |N(u) ∩ N(v)|
                    A2 = torch.sparse.mm(A, A).coalesce()
                    a2_row = A2.indices()[0]
                    a2_col = A2.indices()[1]
                    a2_val = A2.values()
                    # 将 (i, j) 编码为 i*N+j, 用排序 + searchsorted 做批量查找
                    a2_keys = a2_row * N + a2_col
                    edge_keys = src * N + dst
                    sorted_order = torch.argsort(a2_keys)
                    sorted_keys = a2_keys[sorted_order]
                    sorted_vals = a2_val[sorted_order]
                    positions = torch.searchsorted(
                        sorted_keys, edge_keys
                    ).clamp(max=sorted_keys.size(0) - 1)
                    found = sorted_keys[positions] == edge_keys
                    cn = torch.where(
                        found, sorted_vals[positions],
                        torch.zeros(E, dtype=torch.float32, device=dev),
                    )
                    # 自环在原 Python 中显式跳过 (if u == v: continue), 这里对齐
                    if not no_self.all():
                        cn = torch.where(no_self, cn, torch.zeros_like(cn))
                    cn_max = cn.max()
                    if cn.numel() > 0 and float(cn_max) > 0:
                        cn_norm = cn / (cn_max + 1e-8)
                    else:
                        cn_norm = cn
        self._cn_norm = cn_norm
        self._cn_sig = sig
        return cn_norm

    def forward(self, z, edge_index, intent_vector, num_nodes):
        src, dst = edge_index[0], edge_index[1]
        E = edge_index.shape[1]

        # --- 维度一: 拓扑矛盾性 ---
        cn_norm = self._common_neighbor_norm(edge_index, num_nodes)

        # 分块计算避免一次性创建大张量 (E × 2*hidden 可能几百 MB)
        # 减小 chunk_size 降低峰值显存 (4096 → 1024)
        chunk_size = min(1024, max(1, E))
        s_topo_list, s_sem_list, s_intent_list = [], [], []

        for start in range(0, E, chunk_size):
            end = min(start + chunk_size, E)
            src_chunk = src[start:end]
            dst_chunk = dst[start:end]
            cn_chunk = cn_norm[start:end]

            # 拓扑矛盾性
            topo_feat = torch.cat([z[src_chunk], z[dst_chunk]], dim=-1)
            s_topo_chunk = self.topo_mlp(topo_feat).squeeze(-1) * cn_chunk
            del topo_feat  # 及时释放

            # 语义背离度
            sem_sim = F.cosine_similarity(z[src_chunk], z[dst_chunk], dim=-1)
            deg = degree(src_chunk, num_nodes=num_nodes).clamp(min=1.0)
            strength = torch.sqrt(deg[src_chunk] * deg[dst_chunk])
            strength = strength / (strength.max() + 1e-8)
            deviation = F.relu(sem_sim - strength)
            s_sem_chunk = self.sem_mlp(deviation.unsqueeze(-1)).squeeze(-1) * deviation
            del sem_sim, deg, strength, deviation  # 及时释放

            # 意图相关性
            edge_center = 0.5 * (z[src_chunk] + z[dst_chunk])
            intent_exp = intent_vector.unsqueeze(0).expand(edge_center.size(0), -1)
            s_intent_chunk = self.intent_mlp(
                torch.cat([edge_center, intent_exp], dim=-1)
            ).squeeze(-1)
            del edge_center, intent_exp  # 及时释放

            s_topo_list.append(s_topo_chunk)
            s_sem_list.append(s_sem_chunk)
            s_intent_list.append(s_intent_chunk)

        # 拼接所有块
        s_topo = torch.cat(s_topo_list, dim=0)
        s_sem = torch.cat(s_sem_list, dim=0)
        s_intent = torch.cat(s_intent_list, dim=0)

        # 显式释放列表引用, 避免梯度图持有所有 chunk
        del s_topo_list, s_sem_list, s_intent_list

        # --- 融合 ---
        stacked = torch.stack([s_topo, s_sem, s_intent], dim=-1)
        del s_topo, s_sem, s_intent  # 及时释放
        return self.fuse(stacked).squeeze(-1)                  # [E] in (0,1)


class SuspiciousNodeIdentifier(nn.Module):
    def __init__(self, num_hidden, intent_dim, top_k=50, drop_p=0.1):
        super().__init__()
        self.top_k = top_k
        self.edge_importance = AdversarialCommunityAwareEdgeImportance(
            num_hidden, intent_dim, drop_p
        )
        # 语义异常: 仅基于节点特征在空间中的位置 (高异常 = 远离群体)
        self.sem_anomaly_mlp = nn.Sequential(
            nn.Linear(num_hidden, num_hidden), nn.ReLU(),
            nn.Dropout(drop_p), nn.Linear(num_hidden, 1)
        )
        # 意图异常: 基于节点特征与意图向量的联合偏离
        self.intent_anomaly_mlp = nn.Sequential(
            nn.Linear(num_hidden + intent_dim, num_hidden), nn.ReLU(),
            nn.Dropout(drop_p), nn.Linear(num_hidden, 1)
        )

    def forward(self, z, edge_index, intent_vector, node_time=None):
        num_nodes = z.size(0)
        edge_imp = self.edge_importance(z, edge_index, intent_vector, num_nodes)

        # 边重要性聚合到节点 (均值)
        src = edge_index[0]
        node_imp_sum = torch.zeros(num_nodes, device=z.device)
        node_deg = torch.zeros(num_nodes, device=z.device)
        node_imp_sum.index_add_(0, src, edge_imp)
        node_deg.index_add_(0, src, torch.ones_like(edge_imp))
        node_edge_score = node_imp_sum / node_deg.clamp(min=1.0)
        del node_imp_sum, node_deg  # 及时释放

        # 分块计算节点级语义/意图异常, 避免一次性对全部 N 节点跑 MLP
        # (N × hidden 可能几百 MB, 加上梯度图翻倍)
        node_chunk = 1024
        sem_list, intent_list = [], []
        intent_exp_full = intent_vector.unsqueeze(0).expand(num_nodes, -1)
        for start in range(0, num_nodes, node_chunk):
            end = min(start + node_chunk, num_nodes)
            z_chunk = z[start:end]
            # 语义异常
            sem_list.append(
                torch.sigmoid(self.sem_anomaly_mlp(z_chunk).squeeze(-1))
            )
            # 意图异常
            intent_anom_chunk = torch.sigmoid(
                self.intent_anomaly_mlp(
                    torch.cat([z_chunk, intent_exp_full[start:end]], dim=-1)
                ).squeeze(-1)
            )
            intent_list.append(intent_anom_chunk)
        sem_anomaly = torch.cat(sem_list, dim=0)
        intent_anomaly = torch.cat(intent_list, dim=0)
        del sem_list, intent_list, intent_exp_full

        # 时序异常 (无梯度, 纯统计量)
        time_anomaly = compute_temporal_anomaly(node_time, num_nodes, device=z.device)
        has_time = float(time_anomaly.abs().sum()) > 0.0

        # 统一四维异常打分 (四个独立维度)
        w_topo = 0.3
        w_sem = 0.3
        w_intent = 0.3
        w_time = 0.1 if has_time else 0.0
        if has_time:
            total = w_topo + w_sem + w_intent + w_time
            w_topo = w_topo / total
            w_sem = w_sem / total
            w_intent = w_intent / total
            w_time = w_time / total
            node_score = (w_topo * node_edge_score +
                          w_sem * sem_anomaly +
                          w_intent * intent_anomaly +
                          w_time * time_anomaly)
        else:
            total = w_topo + w_sem + w_intent
            w_topo = w_topo / total
            w_sem = w_sem / total
            w_intent = w_intent / total
            node_score = (w_topo * node_edge_score +
                          w_sem * sem_anomaly +
                          w_intent * intent_anomaly)

        k = min(self.top_k, num_nodes)
        topk_idx = torch.topk(node_score, k).indices
        return topk_idx, node_score
