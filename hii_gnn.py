"""
创新点三: 层次化意图注入的图神经网络 (HII-GNN)

在 GNN 的不同层次注入意图信息, 实现从局部到全局的意图感知表示学习:
- Layer1 局部意图注入: 意图参与边注意力, 影响消息传递权重
- Layer2 邻域意图聚合: 意图门控选择性聚合邻居
- Layer3 全局意图融合: 交叉注意力实现节点表示与意图深度融合

forward 签名与 model.Encoder 对齐 (x, edge_index, edge_weight, intent),
GCN 基线忽略 intent, 两者可经 --encoder 开关互换。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class LocalIntentInjectionLayer(MessagePassing):
    """Layer1: 意图门控调制源节点特征后做多头注意力消息传递。

    旧版把意图向量直接拼接进 attention score, 但意图对所有边是常量,
    在 per-target softmax 中被完全抵消——等于没注入。

    修正: 意图先通过 sigmoid 门控调制源节点特征维度 (乘性交互),
    使不同源节点在意图视角下呈现不同的 "重要性剖面", 从而真正
    影响注意力分布。同时增加残差投影 + LayerNorm 稳定训练。
    """

    def __init__(self, in_channels, out_channels, heads=4, drop_p=0.0):
        super().__init__(aggr='add', node_dim=0)
        assert out_channels % heads == 0
        self.heads = heads
        self.out_channels = out_channels
        self.head_dim = out_channels // heads

        self.lin = nn.Linear(in_channels, out_channels)
        # 意图门控: sigmoid(W·intent) ⊙ h_src → 不同源节点被不同程度放大
        self.intent_gate = nn.Sequential(
            nn.Linear(out_channels, out_channels), nn.Sigmoid()
        )
        # 2-part attention: [gated_src || dst]
        self.att = nn.Parameter(torch.empty(1, heads, 2 * self.head_dim))
        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(drop_p)
        # 残差投影 + LayerNorm
        self.residual_proj = (nn.Linear(in_channels, out_channels)
                              if in_channels != out_channels else nn.Identity())
        self.norm = nn.LayerNorm(out_channels)
        self._reset()

    def _reset(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
        nn.init.xavier_uniform_(self.att)
        if isinstance(self.residual_proj, nn.Linear):
            nn.init.xavier_uniform_(self.residual_proj.weight)
            nn.init.zeros_(self.residual_proj.bias)

    def forward(self, x, edge_index, edge_weight, intent_h):
        h = self.lin(x).view(-1, self.heads, self.head_dim)
        gate = self.intent_gate(intent_h).view(self.heads, self.head_dim)
        num_nodes = x.size(0)
        out = self.propagate(edge_index, x=h, gate=gate,
                             edge_weight=edge_weight, size=(num_nodes, num_nodes))
        out = out.view(-1, self.out_channels)
        return self.norm(out + self.residual_proj(x))

    def message(self, x_i, x_j, gate, index, edge_weight, size_i):
        E = x_j.size(0)
        gate_exp = gate.unsqueeze(0).expand(E, -1, -1)
        x_i_gated = x_i * gate_exp                                # 意图调制源节点
        cat = torch.cat([x_i_gated, x_j], dim=-1)                 # [E,heads,2*hd]
        alpha = self.leaky((cat * self.att).sum(dim=-1))           # [E,heads]
        alpha = softmax(alpha, index, num_nodes=size_i)
        if edge_weight is not None:
            alpha = alpha * edge_weight.view(-1, 1)
        alpha = self.dropout(alpha)
        return x_j * alpha.unsqueeze(-1)


class NeighborIntentAggregationLayer(MessagePassing):
    """Layer2: 意图门控选择性聚合邻居 + 残差 + LayerNorm。"""

    def __init__(self, channels, intent_dim_h, drop_p=0.0):
        super().__init__(aggr='add', node_dim=0)
        self.msg_mlp = nn.Sequential(
            nn.Linear(channels, channels), nn.ReLU(), nn.Dropout(drop_p)
        )
        self.gate = nn.Sequential(
            nn.Linear(channels + intent_dim_h, channels), nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x, edge_index, edge_weight, intent_h):
        agg = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        agg = self.msg_mlp(agg)
        intent_exp = intent_h.unsqueeze(0).expand(x.size(0), -1)
        gate = self.gate(torch.cat([x, intent_exp], dim=-1))
        return self.norm(x + agg * gate)

    def message(self, x_j, edge_weight):
        if edge_weight is not None:
            return edge_weight.view(-1, 1) * x_j
        return x_j


class GlobalIntentFusionLayer(nn.Module):
    """Layer3: 节点为 Query、意图为 Key/Value 的交叉注意力 + FFN + 双残差。"""

    def __init__(self, channels, heads=4, drop_p=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, heads, dropout=drop_p,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, 4 * channels), nn.GELU(),
            nn.Dropout(drop_p), nn.Linear(4 * channels, channels)
        )
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x, intent_h):
        # x:[N,C] -> query [N,1,C]; intent kv [N,1,C]
        q = x.unsqueeze(1)
        kv = intent_h.unsqueeze(0).expand(x.size(0), -1).unsqueeze(1)
        att, _ = self.attn(q, kv, kv)
        x = self.norm1(x + att.squeeze(1))
        x = self.norm2(x + self.ffn(x))
        return x


class HierarchicalIntentInjectedGNN(nn.Module):
    """三层意图注入编码器。输出维度 = out_channels, 与下游对齐。

    forward(x, edge_index, edge_weight, intent) 与 model.Encoder 兼容。
    """

    def __init__(self, in_channels, out_channels, activation, intent_dim,
                 num_layers=3, heads=4, drop_p=0.0):
        super().__init__()
        self.activation = activation
        self.intent_adapter = nn.Linear(intent_dim, out_channels)

        self.local = LocalIntentInjectionLayer(in_channels, out_channels,
                                               heads=heads, drop_p=drop_p)
        # num_layers=1: 只有 local
        # num_layers=2: local + global
        # num_layers>=3: local + (num_layers-2) 个 neighbor + global
        self.neighbor_layers = nn.ModuleList([
            NeighborIntentAggregationLayer(out_channels, out_channels, drop_p)
            for _ in range(max(0, num_layers - 2))
        ])
        self.use_global = num_layers >= 2
        if self.use_global:
            self.global_fusion = GlobalIntentFusionLayer(out_channels, heads,
                                                         drop_p)

    def forward(self, x, edge_index, edge_weight, intent):
        intent_h = self.intent_adapter(intent)            # [out_channels]

        h = self.local(x, edge_index, edge_weight, intent_h)
        h = self.activation(h)

        for layer in self.neighbor_layers:
            h = layer(h, edge_index, edge_weight, intent_h)
            h = self.activation(h)

        if self.use_global:
            h = self.global_fusion(h, intent_h)
        return h
