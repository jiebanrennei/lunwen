"""
多关系图编码 + 意图条件化关系注意力融合 (ICRA)

每条 meta-path (关系) 用一个独立的 HII-GNN 编码, 再用 ICRA 把 R 个关系视图
融合成单一节点表示。

与 CLUHCS 语义注意力的本质区别:
- CLUHCS: 关系权重 = δ_φᵀ·tanh(W·z_r), δ_φ 是训练完即固定的全局向量,
  所有查询、所有节点共享同一套 meta-path 偏好, 与查询意图无关。
- ICRA: 关系权重 = softmax_r( q·(W_k·z_r + rel_emb_r) / √d ), 其中 q = W_q·intent。
  权重是 (意图, 节点) 的函数 —— 不同查询意图激活不同 meta-path, 且逐节点自适应。
  这天然与本项目的动态意图生成器衔接, 是 CLUHCS 结构上拿不到的能力。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hii_gnn import HierarchicalIntentInjectedGNN


class IntentConditionedRelationAttention(nn.Module):
    """意图条件化关系注意力 (ICRA)。

    forward(zs, intent) -> (fused[N,H], alpha[R,N,heads])
      zs: 长度 R 的列表, 每个 [N, num_hidden]
      intent: [intent_dim]
    """

    def __init__(self, num_hidden, intent_dim, num_relations, heads=4,
                 attn_dim=128):
        super().__init__()
        assert attn_dim % heads == 0, "attn_dim 必须能整除 heads"
        assert num_hidden % heads == 0, "num_hidden 必须能整除 heads"
        self.num_hidden = num_hidden
        self.num_relations = num_relations
        self.heads = heads
        self.attn_dim = attn_dim
        self.d_h = attn_dim // heads          # 每头打分维度
        self.v_h = num_hidden // heads        # 每头取值维度

        self.W_q = nn.Linear(intent_dim, attn_dim)
        self.W_k = nn.Linear(num_hidden, attn_dim)
        self.rel_emb = nn.Parameter(torch.empty(num_relations, attn_dim))
        self._reset()

    def _reset(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.zeros_(self.W_q.bias)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.zeros_(self.W_k.bias)
        nn.init.normal_(self.rel_emb, std=0.02)

    def forward(self, zs, intent):
        R = self.num_relations
        Z = torch.stack(zs, dim=0)                      # [R, N, H]
        N = Z.size(1)

        # query: 意图 -> [heads, d_h]
        q = self.W_q(intent).view(self.heads, self.d_h)

        # key: 节点表示 + 关系嵌入 -> [R, N, heads, d_h]
        K = self.W_k(Z) + self.rel_emb.unsqueeze(1)     # [R, N, attn_dim]
        K = K.view(R, N, self.heads, self.d_h)

        # 打分 + 对关系维 softmax
        scores = torch.einsum('hd,rnhd->rnh', q, K)     # [R, N, heads]
        scores = scores / (self.d_h ** 0.5)
        alpha = F.softmax(scores, dim=0)                # [R, N, heads]

        # 加权融合: 每头在 num_hidden/heads 分块上独立选关系组合
        Z_h = Z.view(R, N, self.heads, self.v_h)        # [R, N, heads, v_h]
        fused = (alpha.unsqueeze(-1) * Z_h).sum(dim=0)  # [N, heads, v_h]
        fused = fused.reshape(N, self.num_hidden)       # [N, H]
        return fused, alpha


class MultiRelationEncoder(nn.Module):
    """R 个独立 HII-GNN 编码器 + ICRA 融合。

    forward(x, edge_index_list, edge_weight_list, intent) -> [N, num_hidden]
    与 Encoder / HII-GNN 的 4 参签名对齐, 但第 2/3 参为 list。
    """

    def __init__(self, in_channels, num_hidden, activation, intent_dim,
                 num_relations, num_layers=2, heads=4, icra_heads=4,
                 icra_dim=128, drop_p=0.0):
        super().__init__()
        self.num_relations = num_relations
        self.encoders = nn.ModuleList([
            HierarchicalIntentInjectedGNN(
                in_channels, num_hidden, activation, intent_dim,
                num_layers=num_layers, heads=heads, drop_p=drop_p)
            for _ in range(num_relations)
        ])
        self.icra = IntentConditionedRelationAttention(
            num_hidden, intent_dim, num_relations,
            heads=icra_heads, attn_dim=icra_dim)

    def encode_per_relation(self, x, edge_index_list, edge_weight_list, intent):
        """返回 (zs, fused, alpha)。zs 暴露给对抗模型做 per-relation 扰动。"""
        zs = []
        for r in range(self.num_relations):
            ew = edge_weight_list[r] if edge_weight_list is not None else None
            zs.append(self.encoders[r](x, edge_index_list[r], ew, intent))
        fused, alpha = self.icra(zs, intent)
        return zs, fused, alpha

    def forward(self, x, edge_index_list, edge_weight_list, intent):
        _, fused, _ = self.encode_per_relation(
            x, edge_index_list, edge_weight_list, intent)
        return fused
