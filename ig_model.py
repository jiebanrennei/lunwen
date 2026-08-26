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
                 drop_p=0.1, num_cand_per_node=5, num_relations=1,
                 cand_sources='embed', cand_source_topk=None,
                 cand_label_mode='soft', cand_hard_threshold=0.5,
                 cand_intent_dist_k=16, cand_intent_dist_tau=0.2):
        super().__init__()

        self.encoder = encoder
        self.num_cand_per_node = num_cand_per_node
        self.num_relations = num_relations
        self.cand_sources = tuple(
            s.strip().lower() for s in str(cand_sources).split(',')
            if s.strip() and s.strip().lower() != 'none'
        )
        self.cand_source_topk = cand_source_topk
        self.cand_label_mode = cand_label_mode
        self.cand_hard_threshold = cand_hard_threshold
        self.cand_intent_dist_k = cand_intent_dist_k
        self.cand_intent_dist_tau = cand_intent_dist_tau

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
            self._cand_targets = [None] * num_relations
            self._cand_scores = [None] * num_relations
        else:
            self.edge_model_adv = IntentGuidedEdgeModel(
                num_hidden, intent_dim, num_edge_hidden, drop_p
            )
            self.edge_model_rec = IntentGuidedEdgeModel(
                num_hidden, intent_dim, num_edge_hidden, drop_p
            )
            self._cand_edges = None
            self._cand_targets = None
            self._cand_scores = None

    def filter_upper_edges(self, edges):
        u, v = edges[0], edges[1]
        mask = u < v
        return torch.stack([u[mask], v[mask]], dim=0)

    def _empty_candidates(self, device):
        edges = torch.zeros((2, 0), dtype=torch.long, device=device)
        scores = torch.zeros(0, dtype=torch.float, device=device)
        return edges, scores, scores.clone()

    def _source_k(self, num_nodes):
        k = self.cand_source_topk or max(self.num_cand_per_node * 2, self.num_cand_per_node)
        return max(0, min(k, max(0, num_nodes - 1)))

    def _similarity_candidates(self, feat, upper_edges, num_nodes, k):
        if feat is None or k <= 0 or num_nodes <= 1:
            return self._empty_candidates(upper_edges.device)[:2]

        zc = F.normalize(feat, dim=-1)
        msrc = torch.cat([upper_edges[0], upper_edges[1]])
        mdst = torch.cat([upper_edges[1], upper_edges[0]])

        chunk = min(num_nodes, 512 if num_nodes >= 2048 else 2048)
        src_list, dst_list, score_list = [], [], []
        for c0 in range(0, num_nodes, chunk):
            c1 = min(c0 + chunk, num_nodes)
            rows = torch.arange(c0, c1, device=feat.device)
            sim_c = zc[c0:c1] @ zc.t()
            sim_c[rows - c0, rows] = float('-inf')
            sel = (msrc >= c0) & (msrc < c1)
            sim_c[msrc[sel] - c0, mdst[sel]] = float('-inf')
            vals, idx = sim_c.topk(k, dim=1)
            valid = torch.isfinite(vals)
            src_list.append(rows.unsqueeze(1).expand(-1, k)[valid])
            dst_list.append(idx[valid])
            score_list.append(vals[valid])

        if not src_list or sum(t.numel() for t in src_list) == 0:
            return self._empty_candidates(feat.device)[:2]
        src = torch.cat(src_list)
        dst = torch.cat(dst_list)
        scores = torch.cat(score_list).float()
        u = torch.minimum(src, dst)
        v = torch.maximum(src, dst)
        mask = u != v
        return torch.stack([u[mask], v[mask]], dim=0), scores[mask]

    def _twohop_candidates(self, upper_edges, num_nodes, k, binary_score=False):
        device = upper_edges.device
        if k <= 0 or num_nodes <= 1:
            return self._empty_candidates(device)[:2]

        adj = [set() for _ in range(num_nodes)]
        eu = upper_edges[0].detach().cpu().tolist()
        ev = upper_edges[1].detach().cpu().tolist()
        for u, v in zip(eu, ev):
            if u != v:
                adj[u].add(v)
                adj[v].add(u)

        src, dst, scores = [], [], []
        for u in range(num_nodes):
            counts = {}
            for nb in adj[u]:
                for w in adj[nb]:
                    if w == u or w in adj[u]:
                        continue
                    counts[w] = counts.get(w, 0) + 1
            if not counts:
                continue
            ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:k]
            for v, c in ranked:
                a, b = (u, v) if u < v else (v, u)
                src.append(a)
                dst.append(b)
                scores.append(1.0 if binary_score else float(c))

        if not src:
            return self._empty_candidates(device)[:2]
        return (torch.tensor([src, dst], dtype=torch.long, device=device),
                torch.tensor(scores, dtype=torch.float, device=device))

    def _intent_distribution_candidates(self, z, upper_edges, num_nodes, k):
        device = z.device
        if k <= 0 or num_nodes <= 1:
            return self._empty_candidates(device)[:2]

        proto_k = min(num_nodes, max(2, int(self.cand_intent_dist_k)))
        tau = max(1e-6, float(self.cand_intent_dist_tau))
        z_norm = F.normalize(z, dim=-1)
        if proto_k >= num_nodes:
            proto = z_norm
        else:
            proto_idx = torch.topk(z.norm(dim=-1), proto_k).indices
            proto = z_norm[proto_idx]
        intent_dist = torch.softmax((z_norm @ proto.t()) / tau, dim=-1)
        intent_dist = F.normalize(intent_dist, dim=-1)
        return self._similarity_candidates(intent_dist, upper_edges, num_nodes, k)

    def _merge_candidates(self, candidate_items, num_nodes, device):
        score_by_pair = {}
        count_by_pair = {}
        final_k = max(0, self.num_cand_per_node)
        if final_k <= 0:
            return self._empty_candidates(device)

        for edges, scores in candidate_items:
            if edges is None or edges.numel() == 0:
                continue
            u = torch.minimum(edges[0], edges[1]).detach().cpu().tolist()
            v = torch.maximum(edges[0], edges[1]).detach().cpu().tolist()
            s = scores.detach().float().cpu().tolist()
            for a, b, score in zip(u, v, s):
                if a == b:
                    continue
                pair = int(a) * num_nodes + int(b)
                score_by_pair[pair] = score_by_pair.get(pair, 0.0) + float(score)
                count_by_pair[pair] = count_by_pair.get(pair, 0) + 1

        if not score_by_pair:
            return self._empty_candidates(device)

        pairs_cpu = list(score_by_pair.keys())
        raw_scores = torch.tensor([score_by_pair[p] for p in pairs_cpu],
                                  dtype=torch.float, device=device)
        mn, mx = raw_scores.min(), raw_scores.max()
        if (mx - mn).abs() > 1e-12:
            norm_scores = (raw_scores - mn) / (mx - mn)
        else:
            norm_scores = torch.ones_like(raw_scores)

        counts = torch.tensor([count_by_pair[p] for p in pairs_cpu],
                              dtype=torch.float, device=device)
        if self.cand_label_mode == 'hard':
            targets = (norm_scores >= self.cand_hard_threshold).float()
        elif self.cand_label_mode == 'consensus':
            targets = (counts >= 2).float()
        else:
            targets = norm_scores.clamp(0.0, 1.0)

        order = torch.argsort(norm_scores, descending=True).detach().cpu().tolist()
        node_counts = [0] * num_nodes
        keep = []
        for idx in order:
            pair = pairs_cpu[idx]
            u, v = divmod(pair, num_nodes)
            if node_counts[u] >= final_k or node_counts[v] >= final_k:
                continue
            keep.append(idx)
            node_counts[u] += 1
            node_counts[v] += 1

        if not keep:
            return self._empty_candidates(device)
        keep_t = torch.tensor(keep, dtype=torch.long, device=device)
        pair_t = torch.tensor([pairs_cpu[i] for i in keep], dtype=torch.long, device=device)
        edges = torch.stack([pair_t // num_nodes, pair_t % num_nodes], dim=0)
        return edges, targets[keep_t], norm_scores[keep_t]

    @torch.no_grad()
    def generate_candidate_edges(self, z, upper_edges, num_nodes, x=None,
                                 edge_index=None, intent_vector=None,
                                 rec_head=None):
        if self.num_cand_per_node <= 0 or num_nodes <= 1 or not self.cand_sources:
            return self._empty_candidates(z.device)

        k = self._source_k(num_nodes)
        candidate_items = []
        sources = set(self.cand_sources)

        if 'embed' in sources:
            candidate_items.append(self._similarity_candidates(z, upper_edges, num_nodes, k))
        if 'semantic' in sources:
            feat = x if x is not None else z
            candidate_items.append(self._similarity_candidates(feat, upper_edges, num_nodes, k))
        if 'twohop' in sources:
            candidate_items.append(self._twohop_candidates(upper_edges, num_nodes, k, binary_score=True))
        if 'common' in sources:
            candidate_items.append(self._twohop_candidates(upper_edges, num_nodes, k, binary_score=False))
        if 'dist' in sources or 'intent_dist' in sources or 'distribution' in sources:
            candidate_items.append(self._intent_distribution_candidates(z, upper_edges, num_nodes, k))

        if 'intent' in sources and rec_head is not None and intent_vector is not None:
            pool_edges = self._merge_candidates(candidate_items, num_nodes, z.device)[0]
            if pool_edges.numel() == 0:
                pool_edges = self._similarity_candidates(
                    z, upper_edges, num_nodes, max(k, self.num_cand_per_node * 4))[0]
            if pool_edges.numel() > 0:
                intent_scores = torch.sigmoid(rec_head(z, pool_edges, intent_vector).squeeze(-1))
                candidate_items.append((pool_edges, intent_scores.detach()))

        return self._merge_candidates(candidate_items, num_nodes, z.device)

    def refresh_candidate_edges(self, x, edge_index, edge_weight, intent_vector):
        """Refresh cached candidate missing edges with the current encoder state."""
        z = self.encoder(x, edge_index, edge_weight, intent_vector)
        upper_edges = self.filter_upper_edges(edge_index)
        edges, targets, scores = self.generate_candidate_edges(
            z, upper_edges, x.size(0), x=x, edge_index=edge_index,
            intent_vector=intent_vector, rec_head=self.edge_model_rec)
        self._cand_edges = edges
        self._cand_targets = targets
        self._cand_scores = scores

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
            edges, targets, scores = self.generate_candidate_edges(
                z, upper_edges, x.size(0), x=x, edge_index=edge_index,
                intent_vector=intent_vector, rec_head=self.edge_model_rec)
            self._cand_edges = edges
            self._cand_targets = targets
            self._cand_scores = scores

        cand_edges = self._cand_edges
        cand_targets = self._cand_targets
        cand_scores = self._cand_scores
        if cand_edges.size(1) > 0:
            cand_edge_logits = self.edge_model_rec(z, cand_edges, intent_vector)
        else:
            cand_edge_logits = z.new_zeros((0, 1))
            cand_targets = z.new_zeros(0)
            cand_scores = z.new_zeros(0)

        return {
            'upper_edge_logits': upper_edge_logits,
            'lower_edge_logits': lower_edge_logits,
            'upper_edge_fea': upper_edge_fea,
            'lower_edge_fea': lower_edge_fea,
            'cand_edges': cand_edges,
            'cand_edge_logits': cand_edge_logits,
            'cand_edge_targets': cand_targets,
            'cand_edge_scores': cand_scores,
        }

    def _edge_info_one(self, z, edge_index, intent_vector, adv_head, rec_head,
                       cand_slot_idx, x=None):
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
            edges, targets, scores = self.generate_candidate_edges(
                z, upper_edges, z.size(0), x=x, edge_index=edge_index,
                intent_vector=intent_vector, rec_head=rec_head)
            self._cand_edges[cand_slot_idx] = edges
            self._cand_targets[cand_slot_idx] = targets
            self._cand_scores[cand_slot_idx] = scores
        cand_edges = self._cand_edges[cand_slot_idx]
        cand_targets = self._cand_targets[cand_slot_idx]
        cand_scores = self._cand_scores[cand_slot_idx]
        if cand_edges.size(1) > 0:
            cand_edge_logits = rec_head(z, cand_edges, intent_vector)
        else:
            cand_edge_logits = z.new_zeros((0, 1))
            cand_targets = z.new_zeros(0)
            cand_scores = z.new_zeros(0)

        return {
            'upper_edge_logits': upper_edge_logits,
            'lower_edge_logits': lower_edge_logits,
            'upper_edge_fea': upper_edge_fea,
            'lower_edge_fea': lower_edge_fea,
            'cand_edges': cand_edges,
            'cand_edge_logits': cand_edge_logits,
            'cand_edge_targets': cand_targets,
            'cand_edge_scores': cand_scores,
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
                self.edge_model_adv[r], self.edge_model_rec[r], r, x=x))
        return infos

    def refresh_candidate_edges_multi(self, x, edge_index_list,
                                      edge_weight_list, intent_vector):
        """逐关系刷新候选新边缓存(各自嵌入空间内重选)。"""
        zs, _, _ = self.encoder.encode_per_relation(
            x, edge_index_list, edge_weight_list, intent_vector)
        for r in range(self.num_relations):
            upper = self.filter_upper_edges(edge_index_list[r])
            edges, targets, scores = self.generate_candidate_edges(
                zs[r], upper, x.size(0), x=x, edge_index=edge_index_list[r],
                intent_vector=intent_vector, rec_head=self.edge_model_rec[r])
            self._cand_edges[r] = edges
            self._cand_targets[r] = targets
            self._cand_scores[r] = scores


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

    def _chunked_semi_loss(self, z1, z2, already_normalized=False):
        if not already_normalized:
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)
        num_nodes = z1.size(0)
        # 动态 chunk size: 大图 (100K+) 用更小的 chunk 避免 OOM
        if num_nodes > 100000:
            chunk_size = 128
        elif num_nodes > 50000:
            chunk_size = 256
        else:
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
            # 显式释放中间张量
            del refl_sim, between_sim, pos_sim, refl_diag
        return torch.cat(losses, dim=0)

    def semi_loss(self, z1, z2):
        return self._chunked_semi_loss(z1, z2)

    def contrastive_loss(self, z1, z2):
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        l1 = self._chunked_semi_loss(h1, h2, already_normalized=True)
        l2 = self._chunked_semi_loss(h2, h1, already_normalized=True)
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

    def candidate_reconstruction_bce_loss(self, cand_logits, cand_targets):
        if cand_logits is None or cand_targets is None or cand_logits.numel() == 0:
            device = self.fc1.weight.device
            return torch.tensor(0.0, device=device)
        logits = cand_logits.squeeze(-1)
        targets = cand_targets.to(logits.device).float().view_as(logits)
        return F.binary_cross_entropy_with_logits(logits, targets)

    def intent_guided_qc_loss(self, z, q_idx, intent_batch, pos_idx, neg_idx,
                              gate=True, tau_gate=0.2):
        """意图引导的查询中心对比 (IGQC): 自监督抬高 oracle 排序天花板。

        对每个查询 q, 在编码器输出空间 z(= oracle 排序空间)里把 q 的全量图
        邻居(结构正样本)拉近、随机非邻居(负样本)推远; 正样本按"意图对齐度"
        门控加权, 让意图相关的邻居主导拉拢。全程不使用标签 y。

        z:            [N, H]  编码器输出(传 z_rec)
        q_idx:        [B]     查询节点索引
        intent_batch: [B, D]  每个查询各自生成的意图
        pos_idx:      [B, m]  每个查询的全量图邻居(结构正样本)
        neg_idx:      [B, n]  每个查询的随机非邻居
        gate:         True=按意图对齐度加权正样本; False=等权(纯结构消融)
        """
        zc = F.normalize(z, dim=-1)                       # [N, H]
        zq = zc[q_idx]                                     # [B, H]
        zpos = zc[pos_idx]                                 # [B, m, H]
        zneg = zc[neg_idx]                                 # [B, n, H]

        pos_sim = torch.einsum('bh,bmh->bm', zq, zpos) / self.tau   # [B, m]
        neg_sim = torch.einsum('bh,bnh->bn', zq, zneg) / self.tau   # [B, n]

        # 多正样本 InfoNCE: 每个正样本一项, 分母 = 该正 + 全部负
        #   per_pos_i = -log( exp(pos_i) / (exp(pos_i) + Σ_j exp(neg_j)) )
        neg_lse = torch.logsumexp(neg_sim, dim=1, keepdim=True)     # [B, 1]
        denom = torch.logaddexp(pos_sim, neg_lse.expand_as(pos_sim))  # [B, m]
        per_pos = denom - pos_sim                                   # [B, m]

        if gate:
            ip = F.normalize(self.intent_proj(zpos), dim=-1)        # [B, m, D]
            iq = F.normalize(intent_batch, dim=-1).unsqueeze(1)     # [B, 1, D]
            gate_sim = (ip * iq).sum(dim=-1) / tau_gate             # [B, m]
            w = torch.softmax(gate_sim, dim=1)                      # [B, m]
        else:
            w = torch.full_like(per_pos, 1.0 / per_pos.size(1))

        return (w * per_pos).sum(dim=1).mean()

    def intent_conditioned_spnm_loss(self, z, q_idx, intent_batch,
                                     pos_idx, neg_idx, pos_weight=None,
                                     neg_weight=None, tau_gate=0.2):
        zc = F.normalize(z, dim=-1)
        zq = zc[q_idx]
        zpos = zc[pos_idx]
        zneg = zc[neg_idx]

        pos_sim = torch.einsum('bh,bmh->bm', zq, zpos) / self.tau
        neg_sim = torch.einsum('bh,bnh->bn', zq, zneg) / self.tau
        if neg_weight is not None:
            nw = neg_weight.to(neg_sim.device).float().clamp_min(1e-8)
            neg_sim = neg_sim + torch.log(nw)

        neg_lse = torch.logsumexp(neg_sim, dim=1, keepdim=True)
        denom = torch.logaddexp(pos_sim, neg_lse.expand_as(pos_sim))
        per_pos = denom - pos_sim

        if pos_weight is not None:
            w = pos_weight.to(per_pos.device).float()
            w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)
        else:
            ip = F.normalize(self.intent_proj(zpos), dim=-1)
            iq = F.normalize(intent_batch, dim=-1).unsqueeze(1)
            gate_sim = (ip * iq).sum(dim=-1) / tau_gate
            w = torch.softmax(gate_sim, dim=1)

        return (w * per_pos).sum(dim=1).mean()

    def intent_local_seed_struct_loss(self, z, q_idx, intent_batch,
                                      seed_idx, neg_idx, seed_weight=None,
                                      proto_alpha=0.5, tau_gate=0.2,
                                      gate_mode='prior'):
        zc = F.normalize(z, dim=-1)
        zq = zc[q_idx]
        zseed = zc[seed_idx]
        zneg = zc[neg_idx]

        seed_sim = torch.einsum('bh,bsh->bs', zq, zseed) / self.tau
        neg_sim = torch.einsum('bh,bkh->bk', zq, zneg) / self.tau
        neg_lse = torch.logsumexp(neg_sim, dim=1, keepdim=True)
        denom = torch.logaddexp(seed_sim, neg_lse.expand_as(seed_sim))
        per_seed = denom - seed_sim

        gate_mode = str(gate_mode).lower()
        prior = None
        if seed_weight is not None:
            prior = seed_weight.to(per_seed.device).float()
            prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(1e-8)

        if gate_mode == 'prior' and prior is not None:
            w = prior
        else:
            ip = F.normalize(self.intent_proj(zseed), dim=-1)
            iq = F.normalize(intent_batch, dim=-1).unsqueeze(1)
            gate_sim = (ip * iq).sum(dim=-1) / tau_gate
            if gate_mode == 'mix' and prior is not None:
                gate_sim = gate_sim + torch.log(prior.clamp_min(1e-8))
            w = torch.softmax(gate_sim, dim=1)

        node_loss = (w * per_seed).sum(dim=1).mean()
        proto = F.normalize((w.unsqueeze(-1) * zseed).sum(dim=1), dim=-1)
        proto_sim = (zq * proto).sum(dim=-1, keepdim=True) / self.tau
        proto_logits = torch.cat([proto_sim, neg_sim], dim=1)
        proto_loss = torch.logsumexp(proto_logits, dim=1) - proto_sim.squeeze(1)
        proto_alpha = min(1.0, max(0.0, float(proto_alpha)))
        return (1.0 - proto_alpha) * node_loss + proto_alpha * proto_loss.mean()

    def total_loss(self, z_adv, z_rec, intent_vector, reg_loss,
                   reg_lambda=0.5, adv_lambda=1.0, edge_fea_adv=None,
                   edge_fea_rec=None, suspicious_idx=None, lambda_rec=0.1,
                   igqc_args=None, lambda_igqc=0.0, ic_spnm_args=None,
                   lambda_ic_spnm=0.0, ilssc_args=None, lambda_ilssc=0.0,
                   cand_rec_args=None, lambda_cand_bce=0.0):
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

        l_igqc = torch.tensor(0.0, device=z_adv.device)
        if lambda_igqc > 0 and igqc_args is not None:
            l_igqc = self.intent_guided_qc_loss(**igqc_args)
            loss = loss + lambda_igqc * l_igqc

        l_ic_spnm = torch.tensor(0.0, device=z_adv.device)
        if lambda_ic_spnm > 0 and ic_spnm_args is not None:
            l_ic_spnm = self.intent_conditioned_spnm_loss(**ic_spnm_args)
            loss = loss + lambda_ic_spnm * l_ic_spnm

        l_ilssc = torch.tensor(0.0, device=z_adv.device)
        if lambda_ilssc > 0 and ilssc_args is not None:
            l_ilssc = self.intent_local_seed_struct_loss(**ilssc_args)
            loss = loss + lambda_ilssc * l_ilssc

        l_cand_bce = torch.tensor(0.0, device=z_adv.device)
        num_cand_edges = 0
        if lambda_cand_bce > 0 and cand_rec_args is not None:
            cand_logits = cand_rec_args.get('logits')
            cand_targets = cand_rec_args.get('targets')
            if cand_logits is not None:
                num_cand_edges = int(cand_logits.numel())
            l_cand_bce = self.candidate_reconstruction_bce_loss(
                cand_logits, cand_targets)
            loss = loss + lambda_cand_bce * l_cand_bce

        return loss, {
            'contrastive': l_contrastive.item(),
            'intent': l_intent.item(),
            'reg': reg_loss.item(),
            'reconstruction': l_rec.item(),
            'cand_bce': l_cand_bce.item(),
            'num_cand_edges': num_cand_edges,
            'igqc': l_igqc.item(),
            'ic_spnm': l_ic_spnm.item(),
            'ilssc': l_ilssc.item(),
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
