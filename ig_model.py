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
                 drop_p=0.1, num_cand_per_node=5):
        super().__init__()

        self.encoder = encoder
        self.num_cand_per_node = num_cand_per_node
        self._cand_edges = None  # 候选新边缓存: 首次 forward 算一次后复用, 避免每步 N×N

        # 两套独立边头: 意图分别引导"减边(对抗)"与"加边/重构(重构)",
        # 使两个视图的差异真正来自意图,而非仅 Gumbel 噪声与边方向。
        self.edge_model_adv = IntentGuidedEdgeModel(
            num_hidden, intent_dim, num_edge_hidden, drop_p
        )
        self.edge_model_rec = IntentGuidedEdgeModel(
            num_hidden, intent_dim, num_edge_hidden, drop_p
        )

    def filter_upper_edges(self, edges):
        u, v = edges[0], edges[1]
        mask = u < v
        return torch.stack([u[mask], v[mask]], dim=0)

    @torch.no_grad()
    def generate_candidate_edges(self, z, upper_edges, num_nodes):
        """
        为重构视图挑选候选新边: 每个节点在嵌入空间取 top-k 最相似的"非邻居",
        屏蔽已有边与自环,返回去重后的上三角候选边 [2, M]。

        仅做"选择"(no_grad);候选边的打分在 forward 中用 edge_model_rec 完成,
        梯度照常回流。小图直接用 N×N 相似度;节点数很大时应改用分块/近邻索引。
        """
        k = self.num_cand_per_node
        if k <= 0 or num_nodes <= 1:
            return torch.zeros((2, 0), dtype=torch.long, device=z.device)

        zc = F.normalize(z, dim=-1)
        sim = zc @ zc.t()                       # [N, N]
        sim.fill_diagonal_(float('-inf'))
        sim[upper_edges[0], upper_edges[1]] = float('-inf')
        sim[upper_edges[1], upper_edges[0]] = float('-inf')

        k = min(k, num_nodes - 1)
        vals, idx = sim.topk(k, dim=1)          # [N, k]
        valid = torch.isfinite(vals)
        src = torch.arange(num_nodes, device=z.device).unsqueeze(1).expand(-1, k)[valid]
        dst = idx[valid]

        mask = src < dst                        # 取上三角,避免方向重复
        u, v = src[mask], dst[mask]
        if u.numel() == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=z.device)
        pair = torch.unique(u * num_nodes + v)  # 去重
        return torch.stack([pair // num_nodes, pair % num_nodes], dim=0)

    def refresh_candidate_edges(self, x, edge_index, edge_weight, intent_vector):
        """手动刷新候选新边缓存(用当前嵌入重新选一批)。"""
        z = self.encoder(x, edge_index, edge_weight)
        upper_edges = self.filter_upper_edges(edge_index)
        self._cand_edges = self.generate_candidate_edges(z, upper_edges, x.size(0))

    def forward(self, x, edge_index, edge_weight, intent_vector):
        z = self.encoder(x, edge_index, edge_weight)

        upper_edges = self.filter_upper_edges(edge_index)
        lower_edges = torch.stack([upper_edges[1], upper_edges[0]], dim=0)

        upper_edge_logits = self.edge_model_adv(z, upper_edges, intent_vector)
        lower_edge_logits = self.edge_model_rec(z, lower_edges, intent_vector)

        upper_edge_fea = torch.cat(
            [z[upper_edges[0]], z[upper_edges[1]]], dim=1
        )
        lower_edge_fea = torch.cat(
            [z[lower_edges[0]], z[lower_edges[1]]], dim=1
        )

        # 首次调用时生成候选新边并缓存, 后续复用(周期性刷新由外部调 refresh_candidate_edges)
        if self._cand_edges is None:
            self._cand_edges = self.generate_candidate_edges(z, upper_edges, x.size(0))

        cand_edges = self._cand_edges
        if cand_edges.size(1) > 0:
            cand_edge_logits = self.edge_model_rec(z, cand_edges, intent_vector)
        else:
            cand_edge_logits = z.new_zeros((0, 1))

        return {
            'upper_edge_logits': upper_edge_logits,
            'lower_edge_logits': lower_edge_logits,
            'upper_edge_fea': upper_edge_fea,
            'lower_edge_fea': lower_edge_fea,
            'cand_edges': cand_edges,
            'cand_edge_logits': cand_edge_logits,
        }


class IntentContrastiveModel(nn.Module):
    """
    对抗-重构双视图对比学习模型 (AR-DVCL)

    与原版 TrainModel 的区别:
    1. 除对比损失外,增加意图一致性损失
    2. 对抗视图(减边)和重构视图(加边)的语义互补
    """

    def __init__(self, encoder, num_hidden, num_proj_hidden, intent_dim,
                 tau=0.5, lambda_intent=0.3, num_neg_intents=128):
        super().__init__()

        self.encoder = encoder
        self.tau = tau
        self.lambda_intent = lambda_intent

        # 投影头 (对比学习)
        self.fc1 = nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = nn.Linear(num_proj_hidden, num_hidden)

        # 意图对齐投影 (将节点表示映射到意图空间, 最后一层无偏置防止捷径)
        self.intent_proj = nn.Sequential(
            nn.Linear(num_hidden, intent_dim),
            nn.ReLU(),
            nn.Linear(intent_dim, intent_dim, bias=False)
        )

        # 负意图库: K 个随机单位向量, 固定不训练
        neg_intents = torch.randn(num_neg_intents, intent_dim)
        neg_intents = F.normalize(neg_intents, dim=-1)
        self.register_buffer('neg_intents', neg_intents)

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

    def _intent_infonce(self, z_proj_norm, intent_pos):
        """单视图 InfoNCE: 每个节点 vs (真实意图 + K 个负意图)。"""
        # 正样本: 每个节点 vs 真实意图 -> [N, 1]
        pos_sim = (z_proj_norm * intent_pos).sum(dim=-1, keepdim=True) / self.tau
        # 负样本: 每个节点 vs K 个负意图 -> [N, K]
        neg_sim = torch.mm(z_proj_norm, self.neg_intents.t()) / self.tau
        # 拼接 -> [N, 1+K], 正样本固定在索引 0
        logits = torch.cat([pos_sim, neg_sim], dim=-1)
        labels = torch.zeros(logits.size(0), dtype=torch.long,
                             device=logits.device)
        return F.cross_entropy(logits, labels)

    def intent_consistency_loss(self, z_adv, z_rec, intent_vector):
        """
        意图一致性损失 (节点级 Intent InfoNCE)

        每个节点的投影嵌入应与"真实意图"的相似度高于与 K 个随机
        负意图的相似度。相比旧的全图均值余弦对齐:
        - 损失有界 [0, log(1+K)], 不会饱和到 -1
        - 节点级计算消除 mean-pooling 捷径, 梯度真正流到编码器
        - 意图向量作为正锚点, 负意图库提供对比信号
        """
        intent_pos = F.normalize(intent_vector.unsqueeze(0), dim=-1)

        z_adv_norm = F.normalize(self.intent_proj(z_adv), dim=-1)
        z_rec_norm = F.normalize(self.intent_proj(z_rec), dim=-1)

        loss_adv = self._intent_infonce(z_adv_norm, intent_pos)
        loss_rec = self._intent_infonce(z_rec_norm, intent_pos)

        return 0.5 * (loss_adv + loss_rec)

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
