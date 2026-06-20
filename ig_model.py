"""
意图引导的边自对抗增强模型 (IG-ESAA) + 对抗-重构双视图对比学习 (AR-DVCL)

核心创新:
1. 将意图向量注入边权重学习,使边扰动具有目标导向性
2. 生成互补的对抗视图(减边)和重构视图(加边)
3. 多目标损失: 对比学习 + 意图一致性 + 对抗正则化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Encoder


class IntentGuidedEdgeModel(nn.Module):
    """
    意图引导的边权重学习模型

    与原版 AdversarialModel 的 MLP 区别:
    - 原版输入: [z_src || z_dst]  (2 * num_hidden)
    - 本版输入: [z_src || z_dst || intent]  (2 * num_hidden + intent_dim)
    意图向量让边权重学习具有目标导向性
    """

    def __init__(self, num_hidden, intent_dim, num_edge_hidden, drop_p=0.1):
        super().__init__()

        input_dim = num_hidden * 2 + intent_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, num_hidden),
            nn.Dropout(drop_p),
            nn.ReLU(),
            nn.Linear(num_hidden, num_edge_hidden),
            nn.Dropout(drop_p),
            nn.ReLU(),
            nn.Linear(num_edge_hidden, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z, edge_index, intent_vector):
        """
        Args:
            z: 节点表示 [N, num_hidden]
            edge_index: 边索引 [2, E]
            intent_vector: 意图向量 [intent_dim]
        Returns:
            edge_logits: [E, 1]
        """
        src, dst = edge_index[0], edge_index[1]
        num_edges = edge_index.shape[1]

        z_src = z[src]
        z_dst = z[dst]
        intent_expanded = intent_vector.unsqueeze(0).expand(num_edges, -1)

        edge_features = torch.cat([z_src, z_dst, intent_expanded], dim=-1)
        return self.mlp(edge_features)


class IntentGuidedAdversarialModel(nn.Module):
    """
    意图引导的对抗模型 (IG-ESAA)

    与原版 AdversarialModel 的区别:
    1. 边权重 MLP 额外接收意图向量
    2. 双向边分别用同一个 MLP 但方向不同,产生非对称权重
    3. 输出的两组权重语义不同: 对抗视图(减边) vs 重构视图(加边)
    """

    def __init__(self, encoder, num_hidden, intent_dim, num_edge_hidden,
                 drop_p=0.1):
        super().__init__()

        self.encoder = encoder

        self.edge_model = IntentGuidedEdgeModel(
            num_hidden, intent_dim, num_edge_hidden, drop_p
        )

    def filter_upper_edges(self, edges):
        u, v = edges[0], edges[1]
        mask = u < v
        return torch.stack([u[mask], v[mask]], dim=0)

    def forward(self, x, edge_index, edge_weight, intent_vector):
        """
        Args:
            x: 节点特征 [N, F]
            edge_index: 边索引 [2, E]
            edge_weight: 边权重 [E]
            intent_vector: 意图向量 [intent_dim]
        Returns:
            dict: 上三角/下三角的 logits 和特征
        """
        z = self.encoder(x, edge_index, edge_weight)

        upper_edges = self.filter_upper_edges(edge_index)
        lower_edges = torch.stack([upper_edges[1], upper_edges[0]], dim=0)

        upper_edge_logits = self.edge_model(z, upper_edges, intent_vector)
        lower_edge_logits = self.edge_model(z, lower_edges, intent_vector)

        upper_edge_fea = torch.cat(
            [z[upper_edges[0]], z[upper_edges[1]]], dim=1
        )
        lower_edge_fea = torch.cat(
            [z[lower_edges[0]], z[lower_edges[1]]], dim=1
        )

        return {
            'upper_edge_logits': upper_edge_logits,
            'lower_edge_logits': lower_edge_logits,
            'upper_edge_fea': upper_edge_fea,
            'lower_edge_fea': lower_edge_fea,
        }


class IntentContrastiveModel(nn.Module):
    """
    对抗-重构双视图对比学习模型 (AR-DVCL)

    与原版 TrainModel 的区别:
    1. 除对比损失外,增加意图一致性损失
    2. 对抗视图(减边)和重构视图(加边)的语义互补
    """

    def __init__(self, encoder, num_hidden, num_proj_hidden, intent_dim,
                 tau=0.5, lambda_intent=0.3):
        super().__init__()

        self.encoder = encoder
        self.tau = tau
        self.lambda_intent = lambda_intent

        # 投影头 (对比学习)
        self.fc1 = nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = nn.Linear(num_proj_hidden, num_hidden)

        # 意图对齐投影 (将节点表示映射到意图空间)
        self.intent_proj = nn.Sequential(
            nn.Linear(num_hidden, intent_dim),
            nn.ReLU(),
            nn.Linear(intent_dim, intent_dim)
        )

    def forward(self, x, edge_index, edge_weight):
        return self.encoder(x, edge_index, edge_weight)

    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())
        )

    def contrastive_loss(self, z1, z2):
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)
        return (l1 + l2).mean() * 0.5

    def intent_consistency_loss(self, z_adv, z_rec, intent_vector):
        """
        意图一致性损失: 两个视图的全局表示都应与意图向量对齐

        原理: 意图向量是"共享锚点",确保两视图的差异围绕
        意图展开,而不是漫无目的地扰动
        """
        z_adv_proj = self.intent_proj(z_adv.mean(dim=0, keepdim=True))
        z_rec_proj = self.intent_proj(z_rec.mean(dim=0, keepdim=True))
        intent_norm = F.normalize(intent_vector.unsqueeze(0), dim=-1)

        sim_adv = F.cosine_similarity(z_adv_proj, intent_norm)
        sim_rec = F.cosine_similarity(z_rec_proj, intent_norm)

        return -0.5 * (sim_adv + sim_rec).mean()

    def total_loss(self, z_adv, z_rec, intent_vector, reg_loss,
                   reg_lambda=0.5, adv_lambda=1.0, edge_fea_adv=None,
                   edge_fea_rec=None):
        """
        总损失 = L_contrastive + λ_intent * L_intent + λ_adv * L_edge - λ_reg * L_reg

        - L_contrastive: 对比损失,拉近同节点在两视图中的表示
        - L_intent: 意图一致性,确保两视图都与意图对齐
        - L_edge: 边特征一致性 (继承自 EDA-GCL)
        - L_reg: 对抗正则化,鼓励两视图差异
        """
        l_contrastive = self.contrastive_loss(z_adv, z_rec)
        l_intent = self.intent_consistency_loss(z_adv, z_rec, intent_vector)

        loss = l_contrastive + self.lambda_intent * l_intent

        if edge_fea_adv is not None and edge_fea_rec is not None:
            l_edge = F.mse_loss(edge_fea_adv, edge_fea_rec)
            loss = loss + adv_lambda * l_edge

        loss = loss - reg_lambda * reg_loss

        return loss, {
            'contrastive': l_contrastive.item(),
            'intent': l_intent.item(),
            'reg': reg_loss.item(),
            'total': loss.item(),
        }
