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

        # 意图多头注意力 (创新点一 §3.2.2): 源节点表示关注意图信息, 残差保留原特征。
        # 意图先投到隐藏维, 与对端拼成 kv, 使注意力权重非平凡。
        self.intent_proj = nn.Linear(intent_dim, num_hidden)
        self.attn = nn.MultiheadAttention(
            num_hidden, num_heads=4, dropout=drop_p, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(num_hidden)

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

        # 意图多头注意力: query=源节点, kv=[意图, 对端], 残差回源节点表示。
        intent_h = self.intent_proj(intent_expanded)          # [E, H]
        q = z_src.unsqueeze(1)                                # [E, 1, H]
        kv = torch.stack([intent_h, z_dst], dim=1)            # [E, 2, H]
        att, _ = self.attn(q, kv, kv)                         # [E, 1, H]
        z_src_enh = self.attn_norm(z_src + att.squeeze(1))    # 残差 + LayerNorm

        edge_features = torch.cat([z_src_enh, z_dst, intent_expanded], dim=-1)
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
                 drop_p=0.1, num_cand_per_node=5, num_relations=1):
        super().__init__()

        self.encoder = encoder
        self.num_cand_per_node = num_cand_per_node
        self.num_relations = num_relations

        if num_relations > 1:
            self.edge_model_adv = nn.ModuleList([
                IntentGuidedEdgeModel(num_hidden, intent_dim, num_edge_hidden, drop_p)
                for _ in range(num_relations)
            ])
            self.edge_model_rec = nn.ModuleList([
                IntentGuidedEdgeModel(num_hidden, intent_dim, num_edge_hidden, drop_p)
                for _ in range(num_relations)
            ])
            self._cand_edges = [None] * num_relations
        else:
            self.edge_model_adv = IntentGuidedEdgeModel(
                num_hidden, intent_dim, num_edge_hidden, drop_p
            )
            self.edge_model_rec = IntentGuidedEdgeModel(
                num_hidden, intent_dim, num_edge_hidden, drop_p
            )
            self._cand_edges = None

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
        梯度照常回流。相似度按行分块计算,峰值内存 chunk×N 而非 N×N。
        """
        k = self.num_cand_per_node
        if k <= 0 or num_nodes <= 1:
            return torch.zeros((2, 0), dtype=torch.long, device=z.device)

        with torch.no_grad():
            zc = F.normalize(z, dim=-1)
            k = min(k, num_nodes - 1)

            # 需要屏蔽的已有边(两个方向),避免摊出 N×N 再逐元素填 -inf
            msrc = torch.cat([upper_edges[0], upper_edges[1]])
            mdst = torch.cat([upper_edges[1], upper_edges[0]])

            chunk = min(num_nodes, 2048)
            src_list, dst_list = [], []
            for c0 in range(0, num_nodes, chunk):
                c1 = min(c0 + chunk, num_nodes)
                rows = torch.arange(c0, c1, device=z.device)
                sim_c = zc[c0:c1] @ zc.t()                          # [c1-c0, N]
                sim_c[rows - c0, rows] = float('-inf')             # 自环
                sel = (msrc >= c0) & (msrc < c1)
                sim_c[msrc[sel] - c0, mdst[sel]] = float('-inf')   # 已有边
                vals, idx = sim_c.topk(k, dim=1)                   # [c1-c0, k]
                valid = torch.isfinite(vals)
                src_list.append(rows.unsqueeze(1).expand(-1, k)[valid])
                dst_list.append(idx[valid])

            src = torch.cat(src_list)
            dst = torch.cat(dst_list)
            mask = src < dst                    # 取上三角,避免方向重复
            u, v = src[mask], dst[mask]
            if u.numel() == 0:
                return torch.zeros((2, 0), dtype=torch.long, device=z.device)
            pair = torch.unique(u * num_nodes + v)  # 去重
            return torch.stack([pair // num_nodes, pair % num_nodes], dim=0)

    def refresh_candidate_edges(self, x, edge_index, edge_weight, intent_vector):
        """手动刷新候选新边缓存(用当前嵌入重新选一批)。"""
        z = self.encoder(x, edge_index, edge_weight, intent_vector)
        upper_edges = self.filter_upper_edges(edge_index)
        self._cand_edges = self.generate_candidate_edges(z, upper_edges, x.size(0))

    def forward(self, x, edge_index, edge_weight, intent_vector):
        z = self.encoder(x, edge_index, edge_weight, intent_vector)

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

    def _edge_info_one(self, z, edge_index, intent_vector, adv_head, rec_head,
                       cand_slot_idx):
        """对单条关系的节点表示 z 生成 adv/rec/cand 边信息 dict。

        z 用该关系自己的 HII-GNN 嵌入, 保证 PAP 候选边用 PAP 表示打分,
        语义自洽。cand_slot_idx 指定 self._cand_edges 列表中的缓存槽位。
        """
        upper_edges = self.filter_upper_edges(edge_index)
        lower_edges = torch.stack([upper_edges[1], upper_edges[0]], dim=0)

        upper_edge_logits = adv_head(z, upper_edges, intent_vector)
        lower_edge_logits = rec_head(z, lower_edges, intent_vector)

        upper_edge_fea = torch.cat([z[upper_edges[0]], z[upper_edges[1]]], dim=1)
        lower_edge_fea = torch.cat([z[lower_edges[0]], z[lower_edges[1]]], dim=1)

        if self._cand_edges[cand_slot_idx] is None:
            self._cand_edges[cand_slot_idx] = self.generate_candidate_edges(
                z, upper_edges, z.size(0))
        cand_edges = self._cand_edges[cand_slot_idx]
        if cand_edges.size(1) > 0:
            cand_edge_logits = rec_head(z, cand_edges, intent_vector)
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

    def forward_multi(self, x, edge_index_list, edge_weight_list, intent_vector):
        """多关系: 每条 meta-path 用自己的 HII-GNN 嵌入分别扰动。

        返回长度 R 的 list, 每个元素结构与 forward 的 dict 相同。
        """
        zs, _, _ = self.encoder.encode_per_relation(
            x, edge_index_list, edge_weight_list, intent_vector)
        infos = []
        for r in range(self.num_relations):
            infos.append(self._edge_info_one(
                zs[r], edge_index_list[r], intent_vector,
                self.edge_model_adv[r], self.edge_model_rec[r], r))
        return infos

    def refresh_candidate_edges_multi(self, x, edge_index_list,
                                      edge_weight_list, intent_vector):
        """逐关系刷新候选新边缓存(各自嵌入空间内重选)。"""
        zs, _, _ = self.encoder.encode_per_relation(
            x, edge_index_list, edge_weight_list, intent_vector)
        for r in range(self.num_relations):
            upper = self.filter_upper_edges(edge_index_list[r])
            self._cand_edges[r] = self.generate_candidate_edges(
                zs[r], upper, x.size(0))


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

    def forward(self, x, edge_index, edge_weight, intent=None):
        return self.encoder(x, edge_index, edge_weight, intent)

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

    def reconstruction_loss(self, z_adv, z_rec, suspicious_idx):
        """
        对抗特征保持损失: 可疑节点在两视图中应呈现不同的邻近分布。
        对抗视图中分散 (被稀疏化), 重构视图中聚集 (被恢复)。
        用 KL 散度度量差异, 鼓励互补。
        """
        sa = F.normalize(z_adv[suspicious_idx], dim=-1)
        sr = F.normalize(z_rec[suspicious_idx], dim=-1)
        p_adv = F.log_softmax(sa @ sa.t() / self.tau, dim=-1)
        p_rec = F.softmax(sr @ sr.t() / self.tau, dim=-1)
        return F.kl_div(p_adv, p_rec, reduction='batchmean')

    def total_loss(self, z_adv, z_rec, intent_vector, reg_loss,
                   reg_lambda=0.5, adv_lambda=1.0, edge_fea_adv=None,
                   edge_fea_rec=None, suspicious_idx=None, lambda_rec=0.1):
        """
        总损失 = L_contrastive + λ_intent * L_intent + λ_adv * L_edge
                 - λ_reg * L_reg + λ_rec * L_reconstruction

        - L_contrastive: 对比损失,拉近同节点在两视图中的表示
        - L_intent: 意图一致性,确保两视图都与意图对齐
        - L_edge: 边特征一致性 (继承自 EDA-GCL)
        - L_reg: 对抗正则化,鼓励两视图差异
        - L_reconstruction: 可疑节点在两视图中的 KL 散度 (创新点二/四)
        """
        l_contrastive = self.contrastive_loss(z_adv, z_rec)
        l_intent = self.intent_consistency_loss(z_adv, z_rec, intent_vector)

        loss = l_contrastive + self.lambda_intent * l_intent

        if edge_fea_adv is not None and edge_fea_rec is not None:
            l_edge = F.mse_loss(edge_fea_adv, edge_fea_rec)
            loss = loss + adv_lambda * l_edge

        loss = loss - reg_lambda * reg_loss

        l_rec = torch.tensor(0.0, device=z_adv.device)
        if suspicious_idx is not None and suspicious_idx.numel() > 1:
            l_rec = self.reconstruction_loss(z_adv, z_rec, suspicious_idx)
            loss = loss + lambda_rec * l_rec

        return loss, {
            'contrastive': l_contrastive.item(),
            'intent': l_intent.item(),
            'reg': reg_loss.item(),
            'reconstruction': l_rec.item(),
            'total': loss.item(),
        }


class QueryIntentGenerator(nn.Module):
    """根据查询节点特征动态生成意图向量。

    训练时每轮随机采样查询节点 → 生成意图 → 让编码器学会响应不同意图;
    评估社区搜索时每个查询节点用自己的意图重新编码全图。
    """

    def __init__(self, in_features, intent_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, intent_dim * 2),
            nn.ReLU(),
            nn.Linear(intent_dim * 2, intent_dim)
        )

    def forward(self, x_query):
        return F.normalize(self.mlp(x_query), dim=-1)
