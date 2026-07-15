"""
创新点四: 对抗社区感知的边重要性学习 + 可疑节点识别

边重要性综合三个维度:
1. 拓扑矛盾性: 无直接边但共享异常邻居 (共同邻居分析)
2. 语义背离度: 语义相似但拓扑疏远 ("说得多, 连得少")
3. 意图相关性: 边与查询意图的相关程度

可疑节点识别器: 边重要性聚合到节点 + 节点异常分 -> Top-K。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree, to_dense_adj


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
        src, dst = edge_index[0], edge_index[1]
        sig = (num_nodes, edge_index.shape[1], edge_index.data_ptr())
        if (self._cn_norm is not None and self._cn_sig == sig
                and self._cn_norm.device == edge_index.device):
            return self._cn_norm
        with torch.no_grad():
            adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
            cn = (adj @ adj)[src, dst]                     # 每条边的共同邻居数
            cn_norm = cn / (cn.max() + 1e-8)
        self._cn_norm = cn_norm
        self._cn_sig = sig
        return cn_norm

    def forward(self, z, edge_index, intent_vector, num_nodes):
        src, dst = edge_index[0], edge_index[1]

        # --- 维度一: 拓扑矛盾性 ---
        cn_norm = self._common_neighbor_norm(edge_index, num_nodes)
        topo_feat = torch.cat([z[src], z[dst]], dim=-1)
        s_topo = self.topo_mlp(topo_feat).squeeze(-1) * cn_norm

        # --- 维度二: 语义背离度 ---
        sem_sim = F.cosine_similarity(z[src], z[dst], dim=-1)
        deg = degree(src, num_nodes=num_nodes).clamp(min=1.0)
        strength = torch.sqrt(deg[src] * deg[dst])
        strength = strength / (strength.max() + 1e-8)
        deviation = F.relu(sem_sim - strength)
        s_sem = self.sem_mlp(deviation.unsqueeze(-1)).squeeze(-1) * deviation

        # --- 维度三: 意图相关性 ---
        edge_center = 0.5 * (z[src] + z[dst])
        intent_exp = intent_vector.unsqueeze(0).expand(edge_center.size(0), -1)
        s_intent = self.intent_mlp(
            torch.cat([edge_center, intent_exp], dim=-1)
        ).squeeze(-1)

        # --- 融合 ---
        stacked = torch.stack([s_topo, s_sem, s_intent], dim=-1)
        return self.fuse(stacked).squeeze(-1)                  # [E] in (0,1)


class SuspiciousNodeIdentifier(nn.Module):
    def __init__(self, num_hidden, intent_dim, top_k=50, drop_p=0.1):
        super().__init__()
        self.top_k = top_k
        self.edge_importance = AdversarialCommunityAwareEdgeImportance(
            num_hidden, intent_dim, drop_p
        )
        self.anomaly_mlp = nn.Sequential(
            nn.Linear(num_hidden + intent_dim, num_hidden), nn.ReLU(),
            nn.Dropout(drop_p), nn.Linear(num_hidden, 1)
        )

    def forward(self, z, edge_index, intent_vector):
        num_nodes = z.size(0)
        edge_imp = self.edge_importance(z, edge_index, intent_vector, num_nodes)

        # 边重要性聚合到节点 (均值)
        src = edge_index[0]
        node_imp_sum = torch.zeros(num_nodes, device=z.device)
        node_deg = torch.zeros(num_nodes, device=z.device)
        node_imp_sum.index_add_(0, src, edge_imp)
        node_deg.index_add_(0, src, torch.ones_like(edge_imp))
        node_edge_score = node_imp_sum / node_deg.clamp(min=1.0)

        # 节点异常分
        intent_exp = intent_vector.unsqueeze(0).expand(num_nodes, -1)
        anomaly = torch.sigmoid(
            self.anomaly_mlp(torch.cat([z, intent_exp], dim=-1)).squeeze(-1)
        )

        node_score = 0.5 * node_edge_score + 0.5 * anomaly
        k = min(self.top_k, num_nodes)
        topk_idx = torch.topk(node_score, k).indices
        return topk_idx, node_score
