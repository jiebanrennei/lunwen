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
from torch_geometric.nn import GCNConv
from torch.utils.checkpoint import checkpoint


class LocalIntentInjectionLayer(nn.Module):
    """Layer1: 意图门控调制源节点特征后做轻量稀疏传播。"""

    def __init__(self, in_channels, out_channels, heads=4, drop_p=0.0):
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.lin = nn.Linear(in_channels, out_channels)
        self.intent_gate = nn.Sequential(
            nn.Linear(out_channels, out_channels), nn.Sigmoid()
        )
        self.conv = GCNConv(out_channels, out_channels, add_self_loops=False, normalize=False)
        self.dropout = nn.Dropout(drop_p)
        self.residual_proj = (nn.Linear(in_channels, out_channels)
                              if in_channels != out_channels else nn.Identity())
        self.norm = nn.LayerNorm(out_channels)
        self._reset()

    def _reset(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
        if isinstance(self.intent_gate[0], nn.Linear):
            nn.init.xavier_uniform_(self.intent_gate[0].weight)
            nn.init.zeros_(self.intent_gate[0].bias)
        self.conv.reset_parameters()
        if isinstance(self.residual_proj, nn.Linear):
            nn.init.xavier_uniform_(self.residual_proj.weight)
            nn.init.zeros_(self.residual_proj.bias)

    def forward(self, x, edge_index, edge_weight, intent_h):
        h = self.lin(x)
        gate = self.intent_gate(intent_h).unsqueeze(0)
        h = self.dropout(h * gate)
        out = self.conv(h, edge_index, edge_weight)
        return self.norm(out + self.residual_proj(x))


class NeighborIntentAggregationLayer(nn.Module):
    """Layer2: 意图门控选择性聚合邻居 + 残差 + LayerNorm。"""

    def __init__(self, channels, intent_dim_h, drop_p=0.0):
        super().__init__()
        self.conv = GCNConv(channels, channels, add_self_loops=False, normalize=False)
        self.msg_mlp = nn.Sequential(
            nn.Linear(channels, channels), nn.ReLU(), nn.Dropout(drop_p)
        )
        self.gate = nn.Sequential(
            nn.Linear(channels + intent_dim_h, channels), nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x, edge_index, edge_weight, intent_h):
        agg = self.conv(x, edge_index, edge_weight)
        agg = self.msg_mlp(agg)
        intent_exp = intent_h.unsqueeze(0).expand(x.size(0), -1)
        gate = self.gate(torch.cat([x, intent_exp], dim=-1))
        return self.norm(x + agg * gate)



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

        # 训练时让 x 要求梯度, 这样后续所有 checkpoint 都能真正启用
        if self.training and not x.requires_grad:
            x = x.detach().requires_grad_(True)

        def _local_forward(inp, edge_weight=edge_weight, edge_index=edge_index, intent_h=intent_h):
            return self.activation(self.local(inp, edge_index, edge_weight, intent_h))

        if self.training:
            if edge_weight is None:
                h = checkpoint(lambda inp: _local_forward(inp, None), x)
            else:
                h = checkpoint(_local_forward, x, edge_weight)
        else:
            h = _local_forward(x)

        for layer in self.neighbor_layers:
            def _neighbor_forward(inp, ew, layer=layer, edge_index=edge_index,
                                  intent_h=intent_h):
                return self.activation(layer(inp, edge_index, ew, intent_h))

            if self.training and h.requires_grad:
                if edge_weight is None:
                    h = checkpoint(lambda inp: _neighbor_forward(inp, None), h)
                else:
                    h = checkpoint(_neighbor_forward, h, edge_weight)
            else:
                h = _neighbor_forward(h, edge_weight)

        if self.use_global:
            def _global_forward(inp, intent_h=intent_h):
                return self.global_fusion(inp, intent_h)

            if self.training and h.requires_grad:
                h = checkpoint(_global_forward, h)
            else:
                h = _global_forward(h)
        return h
