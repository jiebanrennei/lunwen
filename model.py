import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch
from torch.nn import Linear, BatchNorm1d, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch.utils.checkpoint import checkpoint


class GINConvWithEdgeWeight(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GINConvWithEdgeWeight, self).__init__(aggr='add')  # "add" aggregation
        self.mlp = torch.nn.Sequential(
            Linear(in_channels, out_channels),
            BatchNorm1d(out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
        )

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), device=x.device)

        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        # x_j: [E, out_channels]
        return edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        # MLP after aggregation
        return self.mlp(aggr_out)


class GraphEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 num_layers: int = 1, drop_p: float = 0.0):
        super(GraphEncoder, self).__init__()

        assert num_layers >= 1
        self.num_layers = num_layers
        self.dropout = nn.Dropout(drop_p)
        self.conv = nn.ModuleList()
        self.batch_norm = nn.ModuleList()

        for i in range(num_layers):
            if num_layers == 1:
                self.conv.append(GINConvWithEdgeWeight(in_channels, out_channels))
                bn = nn.BatchNorm1d(out_channels)
            else:
                if i == 0:
                    self.conv.append(GINConvWithEdgeWeight(in_channels, 2 * out_channels))
                    bn = nn.BatchNorm1d(2 * out_channels)
                elif i == num_layers - 1:
                    self.conv.append(GINConvWithEdgeWeight(2 * out_channels, out_channels))
                    bn = nn.BatchNorm1d(out_channels)
                else:
                    self.conv.append(GINConvWithEdgeWeight(2 * out_channels, 2 * out_channels))
                    bn = nn.BatchNorm1d(2 * out_channels)
            self.batch_norm.append(bn)

        self.activation = activation

    def forward(self, x, edge_index, edge_weight=None):
        for i in range(self.num_layers):
            x = self.conv[i](x, edge_index, edge_weight)
            x = self.batch_norm[i](x)
            x = self.dropout(x)
            x = self.activation(x)
        return x


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, num_layers: int = 1, drop_p: float = 0.0):
        super(Encoder, self).__init__()

        assert num_layers >= 1
        self.num_layers = num_layers
        self.dropout = nn.Dropout(drop_p)
        self.conv = nn.ModuleList()

        for i in range(num_layers):
            if num_layers == 1:
                self.conv.append(base_model(in_channels, out_channels))
            else:
                if i == 0:
                    self.conv.append(base_model(in_channels, 2 * out_channels))
                elif i == num_layers - 1:
                    self.conv.append(base_model(2 * out_channels, out_channels))
                else:
                    self.conv.append(base_model(2 * out_channels, 2 * out_channels))

        self.activation = activation

    def forward(self, x, edge_index, edge_weight, intent=None):
        # intent 形参仅为与 HII-GNN 接口对齐, vanilla GCN 忽略它
        for i in range(self.num_layers):
            conv = self.conv[i]

            def _layer_forward(inp, ew, conv=conv):
                out = conv(inp, edge_index, ew)
                out = self.dropout(out)
                return self.activation(out)

            # 训练时总是 checkpoint: data.x 默认不要求梯度, 不能用 x.requires_grad 判断
            if self.training:
                # 保证至少一个输入可导, 让 checkpoint 真正记录并反向重算
                if not x.requires_grad:
                    x = x.detach().requires_grad_(True)
                if edge_weight is None:
                    x = checkpoint(lambda inp: _layer_forward(inp, None), x)
                else:
                    x = checkpoint(_layer_forward, x, edge_weight)
            else:
                x = _layer_forward(x, edge_weight)
        return x


class FrozenEmbeddingEncoder(torch.nn.Module):
    """冻结嵌入编码器: 返回预计算的节点嵌入, 用于 Node2Vec 等基线对照。
    接口与 Encoder 对齐 (forward(x, edge_index, edge_weight, intent=None)),
    但忽略所有输入, 直接返回预计算的嵌入向量。
    """
    def __init__(self, embeddings: torch.Tensor):
        super().__init__()
        # 注册为 buffer, 不参与梯度更新, 但会随 model.to(device) 迁移
        self.register_buffer('emb', embeddings.detach())

    def forward(self, x, edge_index, edge_weight=None, intent=None):
        return self.emb


class TrainModel(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5):
        super(TrainModel, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def _chunked_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        num_nodes = z1.size(0)
        chunk_size = min(512, max(1, num_nodes))
        losses = []
        for start in range(0, num_nodes, chunk_size):
            end = min(start + chunk_size, num_nodes)
            z1_chunk = z1[start:end]
            refl_sim = torch.exp(torch.mm(z1_chunk, z1.t()) / self.tau)
            between_sim = torch.exp(torch.mm(z1_chunk, z2.t()) / self.tau)
            pos_sim = torch.exp((z1_chunk * z2[start:end]).sum(dim=1) / self.tau)
            refl_diag = torch.exp((z1_chunk * z1[start:end]).sum(dim=1) / self.tau)
            losses.append(-torch.log(
                pos_sim / (refl_sim.sum(1) + between_sim.sum(1) - refl_diag)
            ))
        return torch.cat(losses, dim=0)

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        return self._chunked_semi_loss(z1, z2)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret
    
    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index, edge_weight)


class AdversarialModel(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 num_edge_hidden: int, tau: float = 0.5, drop_p: float = 0.0):
        super(AdversarialModel, self).__init__()
        
        self.encoder: Encoder = encoder
        self.tau: float = tau
        
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)
        
        self.mlp_edge_model = nn.Sequential(
            nn.Linear(num_hidden * 2, num_hidden),
            nn.Dropout(drop_p),
            nn.ReLU(),
            nn.Linear(num_hidden, num_edge_hidden),
            nn.Dropout(drop_p),
            nn.ReLU(),
            nn.Linear(num_edge_hidden, 1)
        )
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
        
    def filter_upper_edges(self, edges):
        u, v = edges[0], edges[1]
        mask = u < v
        filtered_u, filtered_v = u[mask], v[mask]
        result = torch.stack([filtered_u, filtered_v], dim=0)
        return result
    
    def forward(self, x, edge_index, edge_weight):
        z = self.encoder(x, edge_index, edge_weight)

        upper_edges = self.filter_upper_edges(edge_index)
        lower_edges = torch.stack([upper_edges[1], upper_edges[0]], dim=0)

        upper_edge_fea = torch.cat([z[upper_edges[0]], z[upper_edges[1]]], dim=1)
        lower_edge_fea = torch.cat([z[lower_edges[0]], z[lower_edges[1]]], dim=1)

        upper_edge_logits = self.mlp_edge_model(upper_edge_fea)
        lower_edge_logits = self.mlp_edge_model(lower_edge_fea)

        return {
        'upper_edge_logits': upper_edge_logits,
        'lower_edge_logits': lower_edge_logits,
        'upper_edge_fea': upper_edge_fea,
        'lower_edge_fea': lower_edge_fea
        }  