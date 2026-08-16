"""
意图引导的对抗-重构双视图对比学习 训练脚本

创新点一 (IG-ESAA): 意图向量注入边权重学习
创新点二 (AR-DVCL): 互补的对抗视图(减边)与重构视图(加边/恢复)

沿用 EDA-GCL 的节点分类台架 (cora_lcc 等) 验证机制能否跑通、不掉点。
意图向量可插拔: --intent_source encoder 用真实意图编码器,
否则回退为固定随机向量,保证开箱即跑。
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 缓解 Windows 上双份 OpenMP 运行时(libiomp5md.dll)在重度 MKL 矩阵运算时的 segfault。
# 必须在 import numpy/torch 之前设置; 外部已指定则尊重外部值。
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import math
import os.path as osp
import random
import sys
from datetime import datetime
from time import perf_counter as t

from utils import set_everything, get_dataset, get_cs_dataset, CS_DATASETS

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected

from model import Encoder
from hii_gnn import HierarchicalIntentInjectedGNN
from multi_relation_fusion import MultiRelationEncoder
from edge_importance import SuspiciousNodeIdentifier
from ig_model import IntentGuidedAdversarialModel, IntentContrastiveModel, QueryIntentGenerator
from eval import (label_classification, community_search, community_search_greedy,
                  community_search_dynamic, community_search_greedy_dynamic,
                  community_search_rl, build_fixed_queries, _build_adj_list,
                  _cs_edge_index)
from actor_critic import ActorCriticCommunityBuilder, train_actor_critic

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class _Tee:
    """把写入同时分发到多个流(控制台 + 日志文件),实现 tee 效果。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def generate_ar_edge_weight(edge_info, temperature=1.0, bias=0.0001):
    """
    生成对抗视图(减边)与重构视图(加边/恢复)的互补边权重。

    对抗视图: 在现有边上做稀疏化(减边)。
    重构视图: 现有边互补子集 + 候选新边(加边),两部分合并。
    """
    device = edge_info['upper_edge_logits'].device
    logits_shape = edge_info['upper_edge_logits'].size()

    # ---- 对抗视图权重 (上三角 logits) ----
    eps_adv = (bias - (1 - bias)) * torch.rand(logits_shape) + (1 - bias)
    gate_adv = (torch.log(eps_adv) - torch.log(1 - eps_adv)).to(device)
    adv_edge_weight = torch.sigmoid(
        (gate_adv + edge_info['upper_edge_logits']) / temperature
    ).squeeze()

    # ---- 重构视图权重 (下三角 logits, 现有边部分) ----
    eps_rec = (bias - (1 - bias)) * torch.rand(logits_shape) + (1 - bias)
    gate_rec = (torch.log(eps_rec) - torch.log(1 - eps_rec)).to(device)
    rec_edge_weight = torch.sigmoid(
        (gate_rec + edge_info['lower_edge_logits']) / temperature
    ).squeeze()

    # ---- 对抗正则化: 鼓励/控制两视图差异 ----
    reg = F.l1_loss(adv_edge_weight, rec_edge_weight)

    # ---- 互补硬阈值: 每条现有边只归属一个视图 ----
    noise = torch.randn_like(adv_edge_weight) * 1e-7
    mask = (adv_edge_weight + noise) > (rec_edge_weight + noise)
    adv_edge_weight = torch.where(mask, adv_edge_weight, torch.tensor(0.).to(device))
    rec_edge_weight = torch.where(~mask, rec_edge_weight, torch.tensor(0.).to(device))

    # ---- 候选新边权重 (加边) ----
    cand_logits = edge_info['cand_edge_logits']
    if cand_logits.numel() > 0:
        cand_shape = cand_logits.size()
        eps_c = (bias - (1 - bias)) * torch.rand(cand_shape) + (1 - bias)
        gate_c = (torch.log(eps_c) - torch.log(1 - eps_c)).to(device)
        cand_edge_weight = torch.sigmoid(
            (gate_c + cand_logits) / temperature
        ).squeeze(-1)
        if cand_edge_weight.dim() == 0:
            cand_edge_weight = cand_edge_weight.unsqueeze(0)
    else:
        cand_edge_weight = torch.zeros(0, device=device)

    return adv_edge_weight, rec_edge_weight, reg, cand_edge_weight


def build_intent_vector(source, query, intent_dim, device, seed,
                        encoder_name='paraphrase-multilingual-MiniLM-L12-v2',
                        library_path=None):
    """构建意图向量。encoder 不可用时回退为固定随机向量。"""
    if source == 'encoder':
        try:
            from adversarial_intent_encoder import SimpleIntentEncoder
            enc = SimpleIntentEncoder(intent_dim=intent_dim,
                                      library_path=library_path,
                                      encoder_name=encoder_name)
            with torch.no_grad():
                iv, _ = enc(query, top_k_patterns=5)
            iv = iv.squeeze(0).float().to(device)
            print(f"[intent] source=encoder ({encoder_name}), dim={intent_dim}")
            return F.normalize(iv, dim=-1)
        except Exception as e:
            print(f"[intent] 编码器不可用 ({e}); 回退为随机意图向量")

    g = torch.Generator().manual_seed(seed)
    iv = torch.randn(intent_dim, generator=g)
    return F.normalize(iv, dim=-1).to(device)


def filter_upper_edges(edges):
    u, v = edges[0], edges[1]
    mask = u < v
    return torch.stack([u[mask], v[mask]], dim=0)


def parse_topk_arg(value):
    parsed = []
    for item in value.split(','):
        item = item.strip()
        if not item:
            continue
        parsed.append('oracle' if item.lower() == 'oracle' else int(item))
    return tuple(parsed)


def parse_float_list_arg(value):
    return tuple(float(item.strip()) for item in value.split(',') if item.strip())


def perturb_edge_index(edge_index, num_nodes, mode='none', rate=0.0, seed=0):
    if mode == 'none' or rate <= 0:
        return edge_index

    device = edge_index.device
    upper = filter_upper_edges(edge_index)
    num_edges = upper.size(1)
    if num_edges == 0:
        return edge_index

    g = torch.Generator(device='cpu').manual_seed(seed)
    keep = torch.ones(num_edges, dtype=torch.bool)
    num_change = min(num_edges, max(1, int(num_edges * rate)))

    if mode in ('drop', 'rewire'):
        drop_idx = torch.randperm(num_edges, generator=g)[:num_change]
        keep[drop_idx] = False
    kept = upper[:, keep].cpu()

    add_edges = []
    if mode in ('add', 'rewire'):
        existing = set((int(a), int(b)) for a, b in zip(upper[0].cpu(), upper[1].cpu()))
        attempts = 0
        max_attempts = max(1000, num_change * 50)
        while len(add_edges) < num_change and attempts < max_attempts:
            attempts += 1
            u = int(torch.randint(0, num_nodes, (1,), generator=g).item())
            v = int(torch.randint(0, num_nodes, (1,), generator=g).item())
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in existing:
                continue
            existing.add((a, b))
            add_edges.append((a, b))

    if add_edges:
        add_t = torch.tensor(add_edges, dtype=torch.long).t()
        kept = torch.cat([kept, add_t], dim=1)
    if kept.numel() == 0:
        kept = upper.cpu()

    rev = kept.flip(0)
    return torch.cat([kept, rev], dim=1).to(device)


def build_ic_spnm_args(z_rec, data, q_epoch, intent_vector, intent_generator,
                       contrastive_model, qc_adj, node_score, args, rng_q):
    if qc_adj is None or args.lambda_ic_spnm <= 0:
        return None, None

    N = data.num_nodes
    dev = data.x.device
    B = max(1, int(args.ic_spnm_num_queries))
    m = max(1, int(args.ic_spnm_pos))
    n = max(1, int(args.ic_spnm_neg))
    hard_pool = max(n, int(args.ic_spnm_hard_pool))
    pos_mode = getattr(args, 'ic_spnm_pos_mode', 'neighbor')
    frontier_hops = max(1, int(getattr(args, 'ic_spnm_frontier_hops', 2)))
    frontier_ratio = min(1.0, max(0.0, float(getattr(args, 'ic_spnm_frontier_ratio', 0.5))))
    frontier_pool = max(1, int(getattr(args, 'ic_spnm_frontier_pool', 128)))
    frontier_conn_beta = max(0.0, float(getattr(args, 'ic_spnm_frontier_conn_beta', 0.2)))
    frontier_min_align = float(getattr(args, 'ic_spnm_frontier_min_align', -1.0))

    with torch.no_grad():
        z_norm = F.normalize(z_rec.detach(), dim=-1)
        z_align = F.normalize(contrastive_model.intent_proj(z_rec.detach()), dim=-1)
        susp = None
        if (node_score is not None and args.ic_spnm_suspicious_alpha > 0
                and not args.no_ic_spnm_suspicious):
            susp = node_score.detach().float()
            susp = (susp - susp.min()) / (susp.max() - susp.min() + 1e-8)

    def _score_nodes(cands, iq_n, visited=None, frontier_weight=False):
        if not cands:
            return [], [], [], []
        cand_list = list(dict.fromkeys(int(c) for c in cands))
        cand_t = torch.tensor(cand_list, device=dev)
        align_t = z_align[cand_t] @ iq_n
        conn_vals = []
        if visited is None:
            conn_vals = [0.0] * len(cand_list)
        else:
            for cand in cand_list:
                c_nbrs = qc_adj[int(cand)]
                conn_vals.append(len(c_nbrs & visited) / max(1, len(c_nbrs)))
        conn_t = torch.tensor(conn_vals, dtype=torch.float, device=dev)
        if frontier_weight:
            score_t = 0.5 * align_t + frontier_conn_beta * conn_t
        else:
            score_t = align_t.clone()
        if susp is not None:
            score_t = score_t + args.ic_spnm_suspicious_alpha * susp[cand_t]
        return cand_list, align_t, conn_t, score_t

    def _top_neighbor_positives(nbrs, iq_n, k):
        cand_list, align_t, conn_t, score_t = _score_nodes(nbrs, iq_n)
        if not cand_list or k <= 0:
            return []
        top_k = min(k, len(cand_list))
        rank = torch.topk(score_t, top_k).indices.cpu().tolist()
        items = []
        for idx in rank:
            items.append({
                'node': cand_list[idx],
                'align': float(align_t[idx].item()),
                'conn': float(conn_t[idx].item()),
                'score': float(score_t[idx].item()),
                'hop': 1,
            })
        return items

    def _frontier_positives(qi, iq_n):
        visited = {int(qi)}
        frontier = set(qc_adj[int(qi)])
        selected = []
        per_step = max(1, int(np.ceil(frontier_pool / frontier_hops)))
        for hop in range(1, frontier_hops + 1):
            cand = [c for c in frontier if c not in visited and c != qi]
            if len(cand) > frontier_pool:
                cand = np.random.choice(np.array(cand, dtype=np.int64),
                                        size=frontier_pool, replace=False).tolist()
            cand_list, align_t, conn_t, score_t = _score_nodes(
                cand, iq_n, visited=visited, frontier_weight=True)
            if not cand_list:
                break
            keep = (align_t >= frontier_min_align).nonzero(as_tuple=False).view(-1)
            if keep.numel() == 0:
                break
            top_k = min(per_step, int(keep.numel()))
            local_rank = torch.topk(score_t[keep], top_k).indices
            rank = keep[local_rank].cpu().tolist()
            new_nodes = []
            for idx in rank:
                node = cand_list[idx]
                selected.append({
                    'node': node,
                    'align': float(align_t[idx].item()),
                    'conn': float(conn_t[idx].item()),
                    'score': float(score_t[idx].item()),
                    'hop': hop,
                })
                new_nodes.append(node)
            visited.update(new_nodes)
            next_frontier = set()
            for node in new_nodes:
                next_frontier.update(qc_adj[int(node)])
            frontier = (frontier | next_frontier) - visited
        return selected[:frontier_pool]

    q_first = int(q_epoch) if intent_generator is not None else int(
        torch.randint(0, N, (1,), generator=rng_q).item())
    extra = torch.randint(0, N, (max(0, B - 1),), generator=rng_q).tolist()
    q_list = [q_first] + extra

    pos_rows, neg_rows, pos_weight_rows, neg_weight_rows = [], [], [], []
    valid_q, intent_list = [], []
    pos_align_stats, pos_align_std_stats, neg_sim_stats = [], [], []
    neg_intent_stats, neg_struct_stats = [], []
    frontier_frac_stats, pos_hop_stats, pos_unique_stats = [], [], []

    for qi in q_list:
        qi = int(qi)
        nbrs = list(qc_adj[qi])
        nbr_set = set(nbrs)
        if len(nbrs) == 0:
            continue

        if intent_generator is not None:
            iq = intent_generator(data.x[qi])
        else:
            iq = intent_vector
        with torch.no_grad():
            iq_n = F.normalize(iq.detach(), dim=-1)
            if pos_mode == 'neighbor':
                pos_items = _top_neighbor_positives(nbrs, iq_n, m)
            else:
                frontier_items = _frontier_positives(qi, iq_n)
                boundary_items = [item for item in frontier_items if item['node'] not in nbr_set]
                first_hop_items = [item for item in frontier_items if item['node'] in nbr_set]
                if pos_mode == 'mix':
                    frontier_k = int(round(m * frontier_ratio))
                    neighbor_k = max(0, m - frontier_k)
                    pos_items = _top_neighbor_positives(nbrs, iq_n, neighbor_k)
                    pos_items.extend(boundary_items[:frontier_k])
                    pos_items.extend(first_hop_items[:max(0, m - len(pos_items))])
                else:
                    pos_items = boundary_items[:m]
                    pos_items.extend(first_hop_items[:max(0, m - len(pos_items))])
                if len(pos_items) == 0:
                    pos_items = _top_neighbor_positives(nbrs, iq_n, m)

            dedup, seen = [], set()
            for item in pos_items:
                node = int(item['node'])
                if node == qi or node in seen:
                    continue
                seen.add(node)
                dedup.append(item)
                if len(dedup) >= m:
                    break
            if len(dedup) == 0:
                continue
            unique_count = len(dedup)
            if len(dedup) < m:
                pad_idx = torch.randint(0, len(dedup), (m - len(dedup),), generator=rng_q).tolist()
                dedup.extend([dedup[i].copy() for i in pad_idx])

            pos_ids = [int(item['node']) for item in dedup[:m]]
            pos_align = torch.tensor([item['align'] for item in dedup[:m]],
                                     dtype=torch.float, device=dev)
            pos_hops = torch.tensor([item['hop'] for item in dedup[:m]],
                                    dtype=torch.float, device=dev)
            if pos_mode == 'neighbor':
                pos_weight_score = torch.tensor([item['score'] for item in dedup[:m]],
                                                dtype=torch.float, device=dev)
                pos_weight = torch.softmax(pos_weight_score / args.ic_spnm_tau_gate, dim=0)
            else:
                pos_weight_score = torch.tensor([
                    0.5 * item['align'] + frontier_conn_beta * item['conn']
                    + (args.ic_spnm_suspicious_alpha * float(susp[item['node']].item())
                       if susp is not None else 0.0)
                    for item in dedup[:m]
                ], dtype=torch.float, device=dev)
                pos_weight = torch.softmax(
                    pos_weight_score / max(float(args.ic_spnm_tau_gate), 0.5), dim=0)
            pos_sel = torch.tensor(pos_ids, device=dev)

            excluded = set(nbrs)
            excluded.update(pos_ids)
            excluded.add(qi)
            allowed = np.array([i for i in range(N) if i not in excluded], dtype=np.int64)
            if allowed.size == 0:
                continue
            pool_size = min(hard_pool, allowed.size)
            pool_np = np.random.choice(allowed, size=pool_size, replace=False)
            pool_t = torch.tensor(pool_np, device=dev)

            emb_sim = z_norm[pool_t] @ z_norm[qi]
            neg_intent = z_align[pool_t] @ iq_n
            q_nbrs = qc_adj[qi]
            cn_scores = []
            q_den = len(q_nbrs) + 1
            for cand in pool_np.tolist():
                c_nbrs = qc_adj[int(cand)]
                cn = len(q_nbrs & c_nbrs) / np.sqrt(q_den * (len(c_nbrs) + 1))
                cn_scores.append(cn)
            cn_t = torch.tensor(cn_scores, dtype=torch.float, device=dev)

            neg_score = (emb_sim
                         - args.ic_spnm_intent_beta * neg_intent
                         - args.ic_spnm_struct_beta * cn_t)
            k_neg = min(n, pool_size)
            neg_rank = torch.topk(neg_score, k_neg).indices
            neg_sel = pool_t[neg_rank]
            neg_w = torch.ones(k_neg, device=dev)
            if k_neg < n:
                pad_idx = torch.randint(0, k_neg, (n - k_neg,), generator=rng_q).to(dev)
                neg_sel = torch.cat([neg_sel, neg_sel[pad_idx]], dim=0)
                neg_w = torch.cat([neg_w, neg_w[pad_idx]], dim=0)

        pos_rows.append(pos_sel.cpu().numpy())
        neg_rows.append(neg_sel.cpu().numpy())
        pos_weight_rows.append(pos_weight.detach())
        neg_weight_rows.append(neg_w.detach())
        valid_q.append(qi)
        intent_list.append(iq)
        pos_align_stats.append(float(pos_align.mean().item()))
        pos_align_std_stats.append(float(pos_align.std(unbiased=False).item()))
        neg_sim_stats.append(float(emb_sim[neg_rank].mean().item()))
        neg_intent_stats.append(float(neg_intent[neg_rank].mean().item()))
        neg_struct_stats.append(float(cn_t[neg_rank].mean().item()))
        frontier_frac_stats.append(float(np.mean([node not in nbr_set for node in pos_ids])))
        pos_hop_stats.append(float(pos_hops.mean().item()))
        pos_unique_stats.append(float(unique_count))

    if not valid_q:
        return None, None

    ic_spnm_args = dict(
        z=z_rec,
        q_idx=torch.tensor(valid_q, device=dev),
        intent_batch=torch.stack(intent_list),
        pos_idx=torch.tensor(np.stack(pos_rows), device=dev),
        neg_idx=torch.tensor(np.stack(neg_rows), device=dev),
        pos_weight=torch.stack(pos_weight_rows).to(dev),
        neg_weight=torch.stack(neg_weight_rows).to(dev),
        tau_gate=args.ic_spnm_tau_gate,
    )
    stats = {
        'valid_q': len(valid_q),
        'pos_align': float(np.mean(pos_align_stats)),
        'pos_align_std': float(np.mean(pos_align_std_stats)),
        'neg_sim': float(np.mean(neg_sim_stats)),
        'neg_intent': float(np.mean(neg_intent_stats)),
        'neg_struct': float(np.mean(neg_struct_stats)),
        'frontier_frac': float(np.mean(frontier_frac_stats)),
        'pos_hop': float(np.mean(pos_hop_stats)),
        'pos_unique': float(np.mean(pos_unique_stats)),
    }
    return ic_spnm_args, stats


def effective_ilssc_lambda(args, epoch):
    base = float(args.lambda_ilssc)
    if base <= 0:
        return 0.0
    warmup = max(0, int(args.ilssc_warmup_epochs))
    ramp = max(0, int(args.ilssc_ramp_epochs))
    if epoch <= warmup:
        return 0.0
    if ramp > 0:
        progress = min(1.0, max(0.0, (epoch - warmup) / float(ramp)))
        return base * progress
    return base


class IntentDistributionMemory:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.proto = None
        self.dist_norm = None
        self.confidence = None
        self.last_update_epoch = -1
        self.updated_this_epoch = False

    def _candidate_index(self, N, node_score, rng_q):
        pool = int(getattr(self.args, 'intent_dist_anchor_pool', 0))
        mode = getattr(self.args, 'intent_dist_proto_mode', 'random')
        if pool <= 0 or pool >= N:
            return torch.arange(N, device=self.device)
        pool = max(2, pool)
        if mode == 'score' and node_score is not None:
            return torch.topk(node_score.detach().float(), min(pool, N)).indices.to(self.device)
        return torch.randperm(N, generator=rng_q)[:pool].to(self.device)

    def _sample_prototypes(self, z_norm, node_score, rng_q):
        N = z_norm.size(0)
        K = min(N, max(2, int(getattr(self.args, 'intent_dist_k', 16))))
        cand_idx = self._candidate_index(N, node_score, rng_q)
        cand_z = z_norm[cand_idx]
        if cand_idx.numel() <= K:
            return F.normalize(cand_z[:K], dim=-1)

        mode = getattr(self.args, 'intent_dist_proto_mode', 'random')
        if mode == 'score' and node_score is not None:
            first = int(torch.argmax(node_score.detach().float()[cand_idx]).item())
        else:
            first = int(torch.randint(0, cand_idx.numel(), (1,), generator=rng_q).item())
        selected = [first]
        selected_mask = torch.zeros(cand_idx.numel(), dtype=torch.bool, device=self.device)
        selected_mask[first] = True
        min_dist = 1.0 - (cand_z @ cand_z[first])
        min_dist = min_dist.masked_fill(selected_mask, float('-inf'))
        for _ in range(1, K):
            next_idx = int(torch.argmax(min_dist).item())
            selected.append(next_idx)
            selected_mask[next_idx] = True
            dist = 1.0 - (cand_z @ cand_z[next_idx])
            min_dist = torch.minimum(min_dist, dist).masked_fill(selected_mask, float('-inf'))
        return F.normalize(cand_z[torch.tensor(selected, device=self.device)], dim=-1)

    def maybe_update(self, epoch, z_rec, node_score, rng_q):
        self.updated_this_epoch = False
        warmup = max(0, int(getattr(self.args, 'intent_dist_memory_warmup', 0)))
        if epoch <= warmup:
            return False
        interval = max(1, int(getattr(self.args, 'intent_dist_update_interval', 10)))
        if self.proto is not None and (epoch - self.last_update_epoch) < interval:
            with torch.no_grad():
                z_norm = F.normalize(z_rec.detach(), dim=-1)
                tau = max(1e-6, float(getattr(self.args, 'intent_dist_tau', 0.2)))
                dist = torch.softmax((z_norm @ self.proto.t()) / tau, dim=-1)
                self.dist_norm = F.normalize(dist, dim=-1).detach()
                top2 = torch.topk(dist, min(2, dist.size(1)), dim=-1).values
                if top2.size(1) == 1:
                    self.confidence = top2[:, 0].detach()
                else:
                    self.confidence = (top2[:, 0] - top2[:, 1]).detach()
            return False

        with torch.no_grad():
            z_norm = F.normalize(z_rec.detach(), dim=-1)
            new_proto = self._sample_prototypes(z_norm, node_score, rng_q)
            if self.proto is not None and self.proto.shape == new_proto.shape:
                ema = min(0.999, max(0.0, float(getattr(self.args, 'intent_dist_ema', 0.7))))
                new_proto = F.normalize(ema * self.proto + (1.0 - ema) * new_proto, dim=-1)
            tau = max(1e-6, float(getattr(self.args, 'intent_dist_tau', 0.2)))
            dist = torch.softmax((z_norm @ new_proto.t()) / tau, dim=-1)
            self.proto = new_proto.detach()
            self.dist_norm = F.normalize(dist, dim=-1).detach()
            top2 = torch.topk(dist, min(2, dist.size(1)), dim=-1).values
            if top2.size(1) == 1:
                self.confidence = top2[:, 0].detach()
            else:
                self.confidence = (top2[:, 0] - top2[:, 1]).detach()
            self.last_update_epoch = int(epoch)
            self.updated_this_epoch = True
        return True


def _build_intent_dist_context(z_norm, node_score, args, rng_q, dev):
    if not getattr(args, 'ilssc_use_intent_dist', False):
        return None
    N = z_norm.size(0)
    K = min(N, max(2, int(args.intent_dist_k)))
    tau = max(1e-6, float(args.intent_dist_tau))
    if getattr(args, 'intent_dist_proto_mode', 'random') == 'score' and node_score is not None:
        proto_idx = torch.topk(node_score.detach().float(), K).indices.to(dev)
    else:
        proto_idx = torch.randperm(N, generator=rng_q)[:K].to(dev)
    proto = F.normalize(z_norm[proto_idx], dim=-1)
    dist = torch.softmax((z_norm @ proto.t()) / tau, dim=-1)
    dist_norm = F.normalize(dist, dim=-1)
    return dist_norm


def build_ilssc_args(z_rec, data, q_epoch, intent_vector, intent_generator,
                     contrastive_model, qc_adj, node_score, args, rng_q,
                     intent_dist_memory=None):
    if qc_adj is None or args.lambda_ilssc <= 0:
        return None, None

    N = data.num_nodes
    dev = data.x.device
    B = max(1, int(args.ilssc_num_queries))
    S = max(1, int(args.ilssc_seed_size))
    K = max(1, int(args.ilssc_neg))
    hops = max(1, int(args.ilssc_hops))
    frontier_pool = max(1, int(args.ilssc_frontier_pool))
    hard_pool = max(K, int(args.ilssc_hard_pool))
    min_align = float(args.ilssc_min_align)
    tau_gate = max(1e-6, float(args.ilssc_tau_gate))

    with torch.no_grad():
        z_norm = F.normalize(z_rec.detach(), dim=-1)
        z_align = F.normalize(contrastive_model.intent_proj(z_rec.detach()), dim=-1)
        if intent_dist_memory is not None and intent_dist_memory.dist_norm is not None:
            intent_dist = intent_dist_memory.dist_norm
            intent_conf = intent_dist_memory.confidence
        else:
            intent_dist = _build_intent_dist_context(z_norm, node_score, args, rng_q, dev)
            intent_conf = None
    dist_beta = float(getattr(args, 'intent_dist_beta', 0.0))
    high_beta = float(getattr(args, 'ilssc_high_order_beta', 0.0))
    conf_tau = max(1e-6, float(getattr(args, 'intent_dist_conf_tau', 0.05)))
    min_conf = float(getattr(args, 'intent_dist_min_conf', 0.0))

    def _confidence_gate(qi, node_t):
        if intent_conf is None:
            return torch.ones(node_t.size(0), dtype=torch.float, device=dev)
        conf_q = intent_conf[int(qi)]
        conf_t = intent_conf[node_t]
        return torch.sigmoid((conf_q * conf_t - min_conf) / conf_tau)

    def _high_order_scores(qi, cand_list):
        if high_beta <= 0:
            return torch.zeros(len(cand_list), dtype=torch.float, device=dev)
        q_nbrs = qc_adj[int(qi)]
        q_deg = max(1, len(q_nbrs))
        vals = []
        for cand in cand_list:
            c_nbrs = qc_adj[int(cand)]
            vals.append(len(q_nbrs & c_nbrs) / math.sqrt(q_deg * max(1, len(c_nbrs))))
        return torch.tensor(vals, dtype=torch.float, device=dev)

    def _score_seed_candidates(cands, qi, iq_n, visited):
        cand_list = list(dict.fromkeys(int(c) for c in cands
                                       if int(c) != int(qi) and int(c) not in visited))
        if not cand_list:
            return [], None, None, None, None, None, None, None
        cand_t = torch.tensor(cand_list, device=dev)
        align_t = z_align[cand_t] @ iq_n
        sim_t = z_norm[cand_t] @ z_norm[int(qi)]
        if intent_dist is not None:
            dist_t = intent_dist[cand_t] @ intent_dist[int(qi)]
            gate_t = _confidence_gate(qi, cand_t)
        else:
            dist_t = torch.zeros_like(sim_t)
            gate_t = torch.ones_like(sim_t)
        if intent_conf is not None:
            conf_t = intent_conf[cand_t]
        else:
            conf_t = torch.zeros_like(sim_t)
        high_t = _high_order_scores(qi, cand_list)
        conn_vals = []
        for cand in cand_list:
            c_nbrs = qc_adj[int(cand)]
            conn_vals.append(len(c_nbrs & visited) / max(1, len(c_nbrs)))
        conn_t = torch.tensor(conn_vals, dtype=torch.float, device=dev)
        score_t = (align_t
                   + args.ilssc_sim_beta * sim_t
                   + args.ilssc_conn_beta * conn_t
                   + high_beta * high_t
                   + dist_beta * gate_t * dist_t)
        return cand_list, align_t, sim_t, conn_t, dist_t, gate_t, conf_t, high_t, score_t

    def _mine_seed(qi, iq_n):
        visited = {int(qi)}
        frontier = set(qc_adj[int(qi)])
        selected = []
        for hop in range(1, hops + 1):
            if len(selected) >= S or not frontier:
                break
            cand = [c for c in frontier if c not in visited and c != qi]
            if len(cand) > frontier_pool:
                cand = np.random.choice(np.array(cand, dtype=np.int64),
                                        size=frontier_pool, replace=False).tolist()
            cand_list, align_t, sim_t, conn_t, dist_t, gate_t, conf_t, high_t, score_t = _score_seed_candidates(
                cand, qi, iq_n, visited)
            if not cand_list:
                break
            keep = (align_t >= min_align).nonzero(as_tuple=False).view(-1)
            if keep.numel() == 0:
                break
            take = min(S - len(selected), int(keep.numel()))
            rank = keep[torch.topk(score_t[keep], take).indices].cpu().tolist()
            new_nodes = []
            for idx in rank:
                node = int(cand_list[idx])
                selected.append({
                    'node': node,
                    'align': float(align_t[idx].item()),
                    'sim': float(sim_t[idx].item()),
                    'conn': float(conn_t[idx].item()),
                    'dist': float(dist_t[idx].item()),
                    'gate': float(gate_t[idx].item()),
                    'conf': float(conf_t[idx].item()),
                    'high': float(high_t[idx].item()),
                    'score': float(score_t[idx].item()),
                    'hop': hop,
                })
                new_nodes.append(node)
            visited.update(new_nodes)
            next_frontier = set()
            for node in new_nodes:
                next_frontier.update(qc_adj[int(node)])
            frontier = (frontier | next_frontier) - visited
        return selected

    q_first = int(q_epoch) if intent_generator is not None else int(
        torch.randint(0, N, (1,), generator=rng_q).item())
    extra = torch.randint(0, N, (max(0, B - 1),), generator=rng_q).tolist()
    q_list = [q_first] + extra

    seed_rows, neg_rows, seed_weight_rows = [], [], []
    valid_q, intent_list = [], []
    seed_align_stats, seed_sim_stats, seed_conn_stats, seed_dist_stats = [], [], [], []
    seed_high_stats = []
    seed_conf_stats, seed_gate_stats, q_conf_stats = [], [], []
    seed_unique_stats, neg_sim_stats, neg_dist_stats = [], [], []
    neg_high_stats = []
    neg_conf_stats, neg_gate_stats = [], []

    for qi in q_list:
        qi = int(qi)
        if len(qc_adj[qi]) == 0:
            continue
        if intent_generator is not None:
            iq = intent_generator(data.x[qi])
        else:
            iq = intent_vector

        with torch.no_grad():
            iq_n = F.normalize(iq.detach(), dim=-1)
            seed_items = _mine_seed(qi, iq_n)
            if len(seed_items) == 0:
                cand_list, align_t, sim_t, conn_t, dist_t, gate_t, conf_t, high_t, score_t = _score_seed_candidates(
                    qc_adj[qi], qi, iq_n, {qi})
                if not cand_list:
                    continue
                take = min(S, len(cand_list))
                rank = torch.topk(score_t, take).indices.cpu().tolist()
                seed_items = [{
                    'node': int(cand_list[idx]),
                    'align': float(align_t[idx].item()),
                    'sim': float(sim_t[idx].item()),
                    'conn': float(conn_t[idx].item()),
                    'dist': float(dist_t[idx].item()),
                    'gate': float(gate_t[idx].item()),
                    'conf': float(conf_t[idx].item()),
                    'high': float(high_t[idx].item()),
                    'score': float(score_t[idx].item()),
                    'hop': 1,
                } for idx in rank]
            if len(seed_items) == 0:
                continue
            unique_count = len({int(item['node']) for item in seed_items})
            if len(seed_items) < S:
                pad_idx = torch.randint(0, len(seed_items), (S - len(seed_items),),
                                        generator=rng_q).tolist()
                seed_items.extend([seed_items[i].copy() for i in pad_idx])
            seed_items = seed_items[:S]
            seed_ids = [int(item['node']) for item in seed_items]
            seed_score = torch.tensor([item['score'] for item in seed_items],
                                      dtype=torch.float, device=dev)
            seed_weight = torch.softmax(seed_score / tau_gate, dim=0)

            excluded = set(seed_ids)
            excluded.update(qc_adj[qi])
            excluded.add(qi)
            allowed = np.array([i for i in range(N) if i not in excluded], dtype=np.int64)
            if allowed.size == 0:
                allowed = np.array([i for i in range(N) if i != qi and i not in set(seed_ids)],
                                   dtype=np.int64)
            if allowed.size == 0:
                continue
            pool_size = min(hard_pool, allowed.size)
            pool_np = np.random.choice(allowed, size=pool_size, replace=False)
            pool_t = torch.tensor(pool_np, device=dev)
            emb_sim = z_norm[pool_t] @ z_norm[qi]
            neg_intent = z_align[pool_t] @ iq_n
            seed_set = set(seed_ids)
            struct_scores = []
            for cand in pool_np.tolist():
                c_nbrs = qc_adj[int(cand)]
                struct_scores.append(len(c_nbrs & seed_set) / max(1, len(c_nbrs)))
            struct_t = torch.tensor(struct_scores, dtype=torch.float, device=dev)
            neg_high = _high_order_scores(qi, pool_np.tolist())
            if intent_dist is not None:
                neg_dist = intent_dist[pool_t] @ intent_dist[qi]
                neg_gate = _confidence_gate(qi, pool_t)
            else:
                neg_dist = torch.zeros_like(emb_sim)
                neg_gate = torch.ones_like(emb_sim)
            if intent_conf is not None:
                neg_conf = intent_conf[pool_t]
                q_conf = intent_conf[qi]
            else:
                neg_conf = torch.zeros_like(emb_sim)
                q_conf = torch.tensor(0.0, device=dev)
            neg_score = emb_sim - args.ilssc_struct_beta * struct_t - high_beta * neg_high
            if args.ilssc_neg_mode == 'hard':
                neg_score = neg_score + args.ilssc_intent_beta * neg_intent + dist_beta * neg_gate * neg_dist
            else:
                neg_score = neg_score - args.ilssc_intent_beta * neg_intent - dist_beta * neg_gate * neg_dist

            k_neg = min(K, pool_size)
            neg_rank = torch.topk(neg_score, k_neg).indices
            neg_sel = pool_t[neg_rank]
            if k_neg < K:
                pad_idx = torch.randint(0, k_neg, (K - k_neg,), generator=rng_q).to(dev)
                neg_sel = torch.cat([neg_sel, neg_sel[pad_idx]], dim=0)

        seed_rows.append(np.array(seed_ids, dtype=np.int64))
        neg_rows.append(neg_sel.cpu().numpy())
        seed_weight_rows.append(seed_weight.detach())
        valid_q.append(qi)
        intent_list.append(iq)
        seed_align_stats.append(float(np.mean([item['align'] for item in seed_items])))
        seed_sim_stats.append(float(np.mean([item['sim'] for item in seed_items])))
        seed_conn_stats.append(float(np.mean([item['conn'] for item in seed_items])))
        seed_dist_stats.append(float(np.mean([item['dist'] for item in seed_items])))
        seed_high_stats.append(float(np.mean([item['high'] for item in seed_items])))
        seed_conf_stats.append(float(np.mean([item['conf'] for item in seed_items])))
        seed_gate_stats.append(float(np.mean([item['gate'] for item in seed_items])))
        q_conf_stats.append(float(q_conf.item()))
        seed_unique_stats.append(float(unique_count))
        neg_sim_stats.append(float(emb_sim[neg_rank].mean().item()))
        neg_dist_stats.append(float(neg_dist[neg_rank].mean().item()))
        neg_high_stats.append(float(neg_high[neg_rank].mean().item()))
        neg_conf_stats.append(float(neg_conf[neg_rank].mean().item()))
        neg_gate_stats.append(float(neg_gate[neg_rank].mean().item()))

    if not valid_q:
        return None, None

    ilssc_args = dict(
        z=z_rec,
        q_idx=torch.tensor(valid_q, device=dev),
        intent_batch=torch.stack(intent_list),
        seed_idx=torch.tensor(np.stack(seed_rows), device=dev),
        neg_idx=torch.tensor(np.stack(neg_rows), device=dev),
        seed_weight=torch.stack(seed_weight_rows).to(dev),
        proto_alpha=args.ilssc_proto_alpha,
        tau_gate=args.ilssc_tau_gate,
        gate_mode=args.ilssc_gate_mode,
    )
    stats = {
        'valid_q': len(valid_q),
        'seed_align': float(np.mean(seed_align_stats)),
        'seed_sim': float(np.mean(seed_sim_stats)),
        'seed_conn': float(np.mean(seed_conn_stats)),
        'seed_dist': float(np.mean(seed_dist_stats)),
        'seed_high': float(np.mean(seed_high_stats)),
        'seed_conf': float(np.mean(seed_conf_stats)),
        'seed_gate': float(np.mean(seed_gate_stats)),
        'q_conf': float(np.mean(q_conf_stats)),
        'seed_unique': float(np.mean(seed_unique_stats)),
        'neg_sim': float(np.mean(neg_sim_stats)),
        'neg_dist': float(np.mean(neg_dist_stats)),
        'neg_high': float(np.mean(neg_high_stats)),
        'neg_conf': float(np.mean(neg_conf_stats)),
        'neg_gate': float(np.mean(neg_gate_stats)),
        'id_updated': int(bool(intent_dist_memory is not None and intent_dist_memory.updated_this_epoch)),
    }
    return ilssc_args, stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora_lcc')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--learning_rate_train', type=float, default=0.001)
    parser.add_argument('--learning_rate_adv', type=float, default=0.0005)
    parser.add_argument('--num_hidden', type=int, default=1024)
    parser.add_argument('--num_proj_hidden', type=int, default=1024)
    parser.add_argument('--num_edge_hidden', type=int, default=64)
    parser.add_argument('--activation', type=str, default='prelu')
    parser.add_argument('--base_model', type=str, default='GCNConv', choices=['GCNConv'])
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--tau', type=float, default=0.4)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--wd_train', type=float, default=1e-5)
    parser.add_argument('--wd_adv', type=float, default=1e-5)
    parser.add_argument('--reg_lambda', type=float, default=0.5)
    parser.add_argument('--adv_lambda', type=float, default=1.0)
    # 意图相关
    parser.add_argument('--intent_dim', type=int, default=256)
    parser.add_argument('--lambda_intent', type=float, default=0.3)
    parser.add_argument('--intent_source', type=str, default='random',
                        choices=['random', 'encoder', 'dynamic'],
                        help='random/encoder=全图固定意图; dynamic=按查询节点动态生成意图')
    parser.add_argument('--intent_num_queries', type=int, default=100,
                        help='动态意图社区搜索时采样的查询数 (每个查询重编码一次)')
    parser.add_argument('--query', type=str,
                        default='找出在社交网络上通过隐蔽连接协同的群体')
    parser.add_argument('--intent_encoder_name', type=str,
                        default='paraphrase-multilingual-MiniLM-L12-v2',
                        help='Text encoder for --intent_source encoder; no LLM is called')
    parser.add_argument('--intent_library_path', type=str, default=None,
                        help='Adversarial pattern library JSON path; None=default bundled library')
    parser.add_argument('--adv_temp', type=float, default=1.0)
    parser.add_argument('--bias', type=float, default=0.0001)
    parser.add_argument('--meta_path', type=str, default=None,
                        help='For ACM/DBLP/IMDB: single meta-path file (e.g. pap.npz). None=merge all.')
    parser.add_argument('--num_cand_per_node', type=int, default=5,
                        help='Final candidate reconstruction edges per node')
    parser.add_argument('--cand_sources', type=str, default='embed',
                        help='Comma-separated candidate sources: embed,twohop,semantic,common,intent,dist')
    parser.add_argument('--cand_source_topk', type=int, default=None,
                        help='Per-source candidate top-k before merging; None=derived from num_cand_per_node')
    parser.add_argument('--cand_intent_dist_k', type=int, default=16,
                        help='Prototype count for dist/intent_dist candidate-edge source')
    parser.add_argument('--cand_intent_dist_tau', type=float, default=0.2,
                        help='Temperature for dist/intent_dist candidate-edge assignments')
    parser.add_argument('--cand_label_mode', type=str, default='soft',
                        choices=['soft', 'hard', 'consensus'],
                        help='Pseudo-label mode for candidate reconstruction BCE')
    parser.add_argument('--cand_hard_threshold', type=float, default=0.5,
                        help='Threshold used by hard candidate pseudo labels')
    parser.add_argument('--lambda_cand_bce', type=float, default=0.0,
                        help='Candidate-edge reconstruction BCE loss weight')
    parser.add_argument('--disable_candidate_edges', action='store_true',
                        help='Disable reconstruction candidate edges')
    parser.add_argument('--cand_refresh_interval', type=int, default=20,
                        help='Refresh candidate edges every N epochs (0=only once)')
    # 创新点三/四
    parser.add_argument('--encoder', type=str, default='gcn',
                        choices=['gcn', 'hii'],
                        help='gcn=vanilla GCN baseline; hii=层次化意图注入GNN')
    parser.add_argument('--hii_heads', type=int, default=4,
                        help='HII-GNN 注意力头数')
    parser.add_argument('--lambda_rec', type=float, default=0.1,
                        help='对抗特征保持(重构)损失权重')
    parser.add_argument('--top_k_suspicious', type=int, default=50,
                        help='可疑节点 Top-K')
    parser.add_argument('--suspicious_boost', type=float, default=1.5,
                        help='社区搜索时可疑节点相似度加权倍数')
    parser.add_argument('--cs_num_queries', type=int, default=40,
                        help='社区搜索固定查询数 (对齐 CLUHCS 40 查询协议)')
    parser.add_argument('--query_file', type=str, default=None,
                        help='固定查询节点文件(每行一个节点id); None 则自动生成并保存')
    parser.add_argument('--use_actor_critic', action='store_true',
                        help='启用 Actor-Critic 对抗图生成器(§7.2 Step4); 默认关闭')
    parser.add_argument('--ac_epochs', type=int, default=100,
                        help='Actor-Critic 自监督训练轮数')
    parser.add_argument('--ac_lr', type=float, default=1e-3,
                        help='Actor-Critic 学习率')
    parser.add_argument('--ac_max_size', type=int, default=200,
                        help='Actor-Critic 生成社区的最大规模')
    parser.add_argument('--ac_size_sweep', type=str, default=None,
                        help='逗号分隔 max_size 列表(如 200,400,600,800,1000,1200,1400); '
                             '设置后一次评测扫出整条 P-R 曲线')
    # 断点续训 (checkpoint / resume)
    parser.add_argument('--resume', action='store_true',
                        help='存在检查点则从中断处继续训练')
    parser.add_argument('--eval_only', action='store_true',
                        help='Load ckpt_path and skip training; useful for sweeping CS/search params')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Name used in default checkpoint path; separates reusable trained models')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='检查点路径; None=checkpoints/ckpt_{model_name}_{dataset}_{encoder}.pt or ckpt_{dataset}_{encoder}.pt')
    parser.add_argument('--ckpt_interval', type=int, default=1,
                        help='每 N 轮保存一次最新检查点(覆盖旧的); 0=不保存')
    # 多关系 + ICRA 融合 (创新点: 意图条件化关系注意力)
    parser.add_argument('--icra_heads', type=int, default=4,
                        help='ICRA 关系注意力头数')
    parser.add_argument('--icra_dim', type=int, default=128,
                        help='ICRA 注意力投影维度')
    parser.add_argument('--relation_fusion', type=str, default='icra',
                        choices=['icra', 'transformer'],
                        help='多关系 meta-path 融合方式: icra 或 transformer')
    parser.add_argument('--cs_relations', type=str, default=None,
                        help='逗号分隔 meta-path 名(如 pap,psp); None=全部关系')
    parser.add_argument('--lambda_rel_entropy', type=float, default=0.0,
                        help='ICRA 关系熵正则权重(0=关; >0 防止塌缩到单一关系)')
    parser.add_argument('--sparsify_topk', type=int, default=None,
                        help='稠密 meta-path top-k 稀疏化(每节点保留k个最强邻居); '
                             'None=不稀疏。ACM-PSP/DBLP-APCPA 等超稠密关系必需(否则OOM)')
    parser.add_argument('--cs_full_graph', action='store_true', default=True,
                        help='社区搜索在全量合并 meta-path 图上扩展(补齐 PSP 等稠密'
                             '主题社区); 默认开。编码器仍用稀疏图, 互不影响')
    parser.add_argument('--no_cs_full_graph', dest='cs_full_graph',
                        action='store_false',
                        help='关闭全量图, CS 退回默认稀疏 edge_index(旧行为)')
    # 阶段B: 意图引导查询中心对比 (IGQC), 抬高 oracle 表示天花板
    parser.add_argument('--lambda_igqc', type=float, default=0.0,
                        help='IGQC 损失权重; 0=关(默认, 向后兼容), 实验设 0.5')
    parser.add_argument('--igqc_pos', type=int, default=20,
                        help='每个查询采样的全量图正邻居数')
    parser.add_argument('--igqc_neg', type=int, default=512,
                        help='每个查询采样的随机非邻居负样本数')
    parser.add_argument('--igqc_num_queries', type=int, default=8,
                        help='每轮采样的查询节点数(单q信号太稀疏)')
    parser.add_argument('--igqc_intent_gate', dest='igqc_intent_gate',
                        action='store_true', default=True,
                        help='IGQC 正样本按意图对齐度门控加权(默认开)')
    parser.add_argument('--no_igqc_intent_gate', dest='igqc_intent_gate',
                        action='store_false',
                        help='关意图门控=纯结构查询对比(消融对照)')
    parser.add_argument('--lambda_ic_spnm', type=float, default=0.0,
                        help='IC-SPNM loss weight; 0=off by default')
    parser.add_argument('--ic_spnm_pos', type=int, default=20,
                        help='IC-SPNM positives per query from CS graph neighbors')
    parser.add_argument('--ic_spnm_neg', type=int, default=256,
                        help='IC-SPNM hard negatives per query')
    parser.add_argument('--ic_spnm_num_queries', type=int, default=8,
                        help='IC-SPNM sampled query nodes per epoch')
    parser.add_argument('--ic_spnm_hard_pool', type=int, default=2048,
                        help='Candidate pool size for hard-negative mining')
    parser.add_argument('--ic_spnm_tau_gate', type=float, default=0.2,
                        help='Temperature for IC-SPNM positive weighting')
    parser.add_argument('--ic_spnm_intent_beta', type=float, default=1.0,
                        help='Penalty for intent-aligned hard negatives')
    parser.add_argument('--ic_spnm_struct_beta', type=float, default=0.5,
                        help='Penalty for structurally aligned hard negatives')
    parser.add_argument('--ic_spnm_suspicious_alpha', type=float, default=0.0,
                        help='Positive score boost from suspicious node score; 0 disables')
    parser.add_argument('--no_ic_spnm_suspicious', action='store_true',
                        help='Disable suspicious-score term in IC-SPNM positive mining')
    parser.add_argument('--ic_spnm_pos_mode', type=str, default='neighbor',
                        choices=['neighbor', 'frontier', 'mix'],
                        help='IC-SPNM positive mining mode: old neighbors, frontier positives, or mixed')
    parser.add_argument('--ic_spnm_frontier_hops', type=int, default=2,
                        help='Expansion steps for frontier-aware IC-SPNM positives')
    parser.add_argument('--ic_spnm_frontier_ratio', type=float, default=0.5,
                        help='Fraction of positives reserved for frontier nodes in mix mode')
    parser.add_argument('--ic_spnm_frontier_pool', type=int, default=128,
                        help='Max frontier candidates kept per query during IC-SPNM mining')
    parser.add_argument('--ic_spnm_frontier_conn_beta', type=float, default=0.2,
                        help='Connectivity reward for frontier-aware IC-SPNM positives')
    parser.add_argument('--ic_spnm_frontier_min_align', type=float, default=-1.0,
                        help='Minimum intent alignment for frontier-aware IC-SPNM positives')
    parser.add_argument('--lambda_ilssc', type=float, default=0.0,
                        help='ILSSC loss weight; 0=off by default')
    parser.add_argument('--ilssc_seed_size', type=int, default=8,
                        help='ILSSC connected local seed nodes per query')
    parser.add_argument('--ilssc_neg', type=int, default=256,
                        help='ILSSC hard negatives per query')
    parser.add_argument('--ilssc_hard_pool', type=int, default=2048,
                        help='Candidate pool size for ILSSC hard-negative mining')
    parser.add_argument('--ilssc_num_queries', type=int, default=8,
                        help='ILSSC sampled query nodes per epoch')
    parser.add_argument('--ilssc_hops', type=int, default=2,
                        help='ILSSC local seed expansion hops')
    parser.add_argument('--ilssc_frontier_pool', type=int, default=128,
                        help='Max frontier candidates scored during ILSSC seed mining')
    parser.add_argument('--ilssc_conn_beta', type=float, default=0.3,
                        help='Connectivity reward for ILSSC seed mining')
    parser.add_argument('--ilssc_sim_beta', type=float, default=0.5,
                        help='Embedding similarity reward for ILSSC seed mining')
    parser.add_argument('--ilssc_intent_beta', type=float, default=1.0,
                        help='Penalty for intent-aligned ILSSC hard negatives')
    parser.add_argument('--ilssc_struct_beta', type=float, default=0.5,
                        help='Penalty for structurally close ILSSC hard negatives')
    parser.add_argument('--ilssc_high_order_beta', type=float, default=0.0,
                        help='IDBR-inspired sparse high-order common-neighbor prior for ILSSC mining')
    parser.add_argument('--ilssc_proto_alpha', type=float, default=0.5,
                        help='Weight of ILSSC seed-prototype contrast term')
    parser.add_argument('--ilssc_tau_gate', type=float, default=0.2,
                        help='Temperature for ILSSC seed weighting')
    parser.add_argument('--ilssc_gate_mode', type=str, default='prior',
                        choices=['prior', 'intent', 'mix'],
                        help='ILSSC seed weighting in loss: prior=v1 stable, intent=differentiable, mix=both')
    parser.add_argument('--ilssc_neg_mode', type=str, default='conservative',
                        choices=['conservative', 'hard'],
                        help='ILSSC negative mining: conservative avoids intent-aligned false negatives; hard mines intent-similar negatives')
    parser.add_argument('--ilssc_warmup_epochs', type=int, default=0,
                        help='Disable ILSSC for first N epochs')
    parser.add_argument('--ilssc_ramp_epochs', type=int, default=0,
                        help='Linearly ramp ILSSC weight after warmup for N epochs')
    parser.add_argument('--ilssc_min_align', type=float, default=-1.0,
                        help='Minimum intent alignment for ILSSC seed candidates')
    parser.add_argument('--ilssc_use_intent_dist', action='store_true',
                        help='Enable ID-ILSSC: intent-distribution-aware seed and negative mining')
    parser.add_argument('--intent_dist_k', type=int, default=16,
                        help='Number of online intent distribution prototypes for ID-ILSSC')
    parser.add_argument('--intent_dist_tau', type=float, default=0.2,
                        help='Temperature for assigning nodes to intent distribution prototypes')
    parser.add_argument('--intent_dist_beta', type=float, default=0.5,
                        help='Weight of intent distribution similarity in ID-ILSSC mining')
    parser.add_argument('--intent_dist_proto_mode', type=str, default='random',
                        choices=['random', 'score'],
                        help='Prototype node selection for ID-ILSSC: random or suspicious-score top nodes')
    parser.add_argument('--intent_dist_stable', action='store_true',
                        help='Enable stable EMA prototype memory and confidence gate for SCID-ILSSC')
    parser.add_argument('--intent_dist_update_interval', type=int, default=10,
                        help='Update interval of stable intent distribution memory')
    parser.add_argument('--intent_dist_memory_warmup', type=int, default=0,
                        help='Disable stable intent distribution memory updates for first N epochs')
    parser.add_argument('--intent_dist_ema', type=float, default=0.7,
                        help='EMA coefficient for stable intent distribution prototypes')
    parser.add_argument('--intent_dist_anchor_pool', type=int, default=0,
                        help='Candidate anchor pool size for farthest-point prototype sampling; 0=all nodes')
    parser.add_argument('--intent_dist_conf_tau', type=float, default=0.05,
                        help='Temperature for confidence-gated intent distribution similarity')
    parser.add_argument('--intent_dist_min_conf', type=float, default=0.0,
                        help='Minimum confidence product before intent distribution similarity is trusted')
    parser.add_argument('--intent_rerank_alpha', type=float, default=0.0,
                        help='推理阶段意图 rerank 系数; 0=关(默认), 实验设 0.1~0.3')
    parser.add_argument('--no_intent_loss', action='store_true',
                        help='Ablation: disable intent consistency loss')
    parser.add_argument('--no_suspicious_kl', action='store_true',
                        help='Ablation: disable suspicious-node KL reconstruction loss')
    parser.add_argument('--no_suspicious_boost', action='store_true',
                        help='Ablation: disable suspicious-node boost during community search')
    parser.add_argument('--no_edge_feature_loss', action='store_true',
                        help='Ablation: disable edge feature consistency loss')
    parser.add_argument('--cs_topk', type=str, default='10,20,50,oracle',
                        help='Comma-separated community-search top-k values')
    parser.add_argument('--cs_w_list', type=str,
                        default='0.0,0.1,0.2,0.3,0.5,0.7,1.0,1.5,2.0',
                        help='Comma-separated greedy community-search w values')
    parser.add_argument('--compute_structure_metrics', action='store_true',
                        help='Compute density/conductance/diameter in greedy CS')
    parser.add_argument('--greedy_patience', type=int, default=0,
                        help='Greedy CS density-stop patience; 0 keeps old first-drop behavior')
    parser.add_argument('--greedy_min_gain_tol', type=float, default=0.0,
                        help='Greedy CS density-drop tolerance; 0 keeps old behavior')
    parser.add_argument('--greedy_size_penalty', type=float, default=0.0,
                        help='Penalty for oversized greedy communities; 0 keeps old behavior')
    parser.add_argument('--greedy_balance_alpha', type=float, default=0.15,
                        help='Mean-similarity support bonus for greedy trace stop and prefix selection')
    parser.add_argument('--greedy_max_size', type=int, default=0,
                        help='Hard cap on greedy community size; 0 disables the cap')
    parser.add_argument('--greedy_adaptive_cap_alpha', type=float, default=0.0,
                        help='Recall-biased adaptive greedy cap scaling factor; 0 disables adaptation')
    parser.add_argument('--greedy_adaptive_cap_floor', type=int, default=0,
                        help='Minimum adaptive greedy cap before scaling')
    parser.add_argument('--greedy_trace_cap_ratio', type=float, default=1.5,
                        help='Trace exploration budget as a multiple of the final greedy cap when multiple w values are evaluated')
    parser.add_argument('--frontier_batch_size', type=int, default=1,
                        help='Greedy CS top-b frontier expansion size; 1 keeps old behavior')
    parser.add_argument('--greedy_connectivity_boost', type=float, default=0.0,
                        help='Optional frontier connectivity boost during greedy expansion; 0 disables')
    parser.add_argument('--greedy_select_mode', type=str, default='first_drop',
                        choices=['first_drop', 'global'],
                        help='Greedy CS density selection mode')
    parser.add_argument('--greedy_init_seed_size', type=int, default=1,
                        help='Initial connected seed size for greedy CS; 1 keeps old behavior')
    parser.add_argument('--greedy_init_seed_hops', type=int, default=1,
                        help='Max hops used to form initial greedy seed')
    parser.add_argument('--greedy_init_seed_conn_beta', type=float, default=0.3,
                        help='Connectivity reward for initial greedy seed')
    parser.add_argument('--greedy_init_seed_min_sim', type=float, default=None,
                        help='Minimum similarity for initial greedy seed candidates')
    parser.add_argument('--greedy_high_order_beta', type=float, default=0.0,
                        help='HSE-Greedy query-candidate high-order reachability weight')
    parser.add_argument('--greedy_comm_cohesion_beta', type=float, default=0.0,
                        help='HSE-Greedy candidate-to-community high-order cohesion weight')
    parser.add_argument('--greedy_comm_direct_beta', type=float, default=0.0,
                        help='HSE-Greedy candidate-to-current-community direct cohesion weight')
    parser.add_argument('--greedy_boundary_gamma', type=float, default=0.0,
                        help='HSE-Greedy boundary expansion penalty weight')
    parser.add_argument('--greedy_hse_pool_size', type=int, default=0,
                        help='Top-K frontier candidates scored by HSE; 0 uses the full frontier')
    parser.add_argument('--greedy_hse_normalize', action='store_true',
                        help='Normalize each HSE structural term inside the current candidate pool')
    parser.add_argument('--greedy_hse_density', action='store_true',
                        help='Use HSE-adjusted selected utility for greedy density selection and early stop')
    parser.add_argument('--greedy_recall_expand_size', type=int, default=0,
                        help='Add up to N high-order frontier nodes after HSE core community selection')
    parser.add_argument('--greedy_recall_min_sim_delta', type=float, default=0.0,
                        help='Minimum fallback candidate similarity as avg_sim + delta')
    parser.add_argument('--include_query_in_pred', action='store_true',
                        help='Include query node in both predicted and truth communities during greedy CS evaluation')
    parser.add_argument('--eval_perturb_mode', type=str, default='none',
                        choices=['none', 'drop', 'add', 'rewire'],
                        help='Evaluation-only graph perturbation mode')
    parser.add_argument('--eval_perturb_rate', type=float, default=0.0,
                        help='Ratio of edges to perturb during evaluation')
    parser.add_argument('--eval_perturb_seed', type=int, default=123,
                        help='Seed for evaluation-only graph perturbation')
    args = parser.parse_args()
    cs_topk = parse_topk_arg(args.cs_topk)
    cs_w_list = parse_float_list_arg(args.cs_w_list)
    trace_early_stop_w = (
        float(np.median(cs_w_list))
        if cs_w_list and str(args.greedy_select_mode).lower() == 'first_drop' and len(cs_w_list) == 1
        else None
    )
    effective_num_cand_per_node = 0 if args.disable_candidate_edges else args.num_cand_per_node
    effective_lambda_intent = 0.0 if args.no_intent_loss else args.lambda_intent
    effective_adv_lambda = 0.0 if args.no_edge_feature_loss else args.adv_lambda
    effective_lambda_rec = 0.0 if args.no_suspicious_kl else args.lambda_rec
    effective_lambda_cand_bce = 0.0 if args.disable_candidate_edges else args.lambda_cand_bce

    # ========== 运行日志: 全部 print/stderr 同时写入文件 (tee) ==========
    os.makedirs("log", exist_ok=True)
    run_log_path = osp.join(
        "log", f"run_{args.dataset}_{datetime.now():%Y%m%d_%H%M%S}.log")
    _run_log_fh = open(run_log_path, "a", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _run_log_fh)
    sys.stderr = _Tee(sys.__stderr__, _run_log_fh)
    print(f"[log] 运行日志写入: {run_log_path}")
    print(f"[config] cand_sources={args.cand_sources} "
          f"num_cand_per_node={effective_num_cand_per_node} "
          f"cand_label_mode={args.cand_label_mode} "
          f"cand_dist_k={args.cand_intent_dist_k} "
          f"cand_dist_tau={args.cand_intent_dist_tau} "
          f"lambda_cand_bce={effective_lambda_cand_bce} "
          f"lambda_ic_spnm={args.lambda_ic_spnm} "
          f"ic_spnm_pos={args.ic_spnm_pos} "
          f"ic_spnm_neg={args.ic_spnm_neg} "
          f"ic_spnm_hard_pool={args.ic_spnm_hard_pool} "
          f"ic_spnm_pos_mode={args.ic_spnm_pos_mode} "
          f"ic_spnm_frontier_ratio={args.ic_spnm_frontier_ratio} "
          f"ic_spnm_frontier_hops={args.ic_spnm_frontier_hops} "
          f"lambda_ilssc={args.lambda_ilssc} "
          f"ilssc_seed_size={args.ilssc_seed_size} "
          f"ilssc_neg={args.ilssc_neg} "
          f"ilssc_hard_pool={args.ilssc_hard_pool} "
          f"ilssc_hops={args.ilssc_hops} "
          f"ilssc_neg_mode={args.ilssc_neg_mode} "
          f"ilssc_gate_mode={args.ilssc_gate_mode} "
          f"ilssc_high_order_beta={args.ilssc_high_order_beta} "
          f"ilssc_warmup={args.ilssc_warmup_epochs} "
          f"ilssc_ramp={args.ilssc_ramp_epochs} "
          f"ilssc_use_intent_dist={args.ilssc_use_intent_dist} "
          f"intent_dist_k={args.intent_dist_k} "
          f"intent_dist_tau={args.intent_dist_tau} "
          f"intent_dist_beta={args.intent_dist_beta} "
          f"intent_dist_proto_mode={args.intent_dist_proto_mode} "
          f"intent_dist_stable={args.intent_dist_stable} "
          f"intent_dist_update_interval={args.intent_dist_update_interval} "
          f"intent_dist_ema={args.intent_dist_ema} "
          f"intent_dist_anchor_pool={args.intent_dist_anchor_pool} "
          f"intent_dist_conf_tau={args.intent_dist_conf_tau} "
          f"intent_dist_min_conf={args.intent_dist_min_conf} "
          f"relation_fusion={args.relation_fusion} "
          f"cs_topk={cs_topk} cs_w_list={cs_w_list} "
          f"greedy_patience={args.greedy_patience} "
          f"greedy_min_gain_tol={args.greedy_min_gain_tol} "
          f"greedy_size_penalty={args.greedy_size_penalty} "
          f"greedy_balance_alpha={args.greedy_balance_alpha} "
          f"greedy_max_size={args.greedy_max_size} "
          f"greedy_adaptive_cap_alpha={args.greedy_adaptive_cap_alpha} "
          f"greedy_adaptive_cap_floor={args.greedy_adaptive_cap_floor} "
          f"greedy_trace_cap_ratio={args.greedy_trace_cap_ratio} "
          f"trace_early_stop_w={trace_early_stop_w} "
          f"frontier_batch_size={args.frontier_batch_size} "
          f"greedy_connectivity_boost={args.greedy_connectivity_boost} "
          f"greedy_select_mode={args.greedy_select_mode} "
          f"greedy_init_seed_size={args.greedy_init_seed_size} "
          f"greedy_init_seed_hops={args.greedy_init_seed_hops} "
          f"greedy_init_seed_conn_beta={args.greedy_init_seed_conn_beta} "
          f"greedy_high_order_beta={args.greedy_high_order_beta} "
          f"greedy_comm_cohesion_beta={args.greedy_comm_cohesion_beta} "
          f"greedy_comm_direct_beta={args.greedy_comm_direct_beta} "
          f"greedy_boundary_gamma={args.greedy_boundary_gamma} "
          f"greedy_hse_pool_size={args.greedy_hse_pool_size} "
          f"greedy_hse_normalize={args.greedy_hse_normalize} "
          f"greedy_hse_density={args.greedy_hse_density} "
          f"greedy_recall_expand_size={args.greedy_recall_expand_size} "
          f"greedy_recall_min_sim_delta={args.greedy_recall_min_sim_delta} "
          f"include_query_in_pred={args.include_query_in_pred} "
          f"eval_perturb={args.eval_perturb_mode}:{args.eval_perturb_rate} "
          f"model_name={args.model_name} "
          f"seed={args.seed}")

    print("Using CPU")
    program_start = t()  # 总运行时间起点 (含数据加载/建模/训练/评估)
    set_everything(args.seed)

    activation = ({
        'relu': F.relu,
        'prelu': nn.PReLU(),
        'rrelu': nn.RReLU(),
        'leakyrelu': nn.LeakyReLU(),
        'gelu': nn.GELU()
    })[args.activation]
    base_model = ({'GCNConv': GCNConv})[args.base_model]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {'GPU: ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    # ========== 数据 ==========
    cs_rel_list = args.cs_relations.split(',') if args.cs_relations else None
    if args.dataset in CS_DATASETS:
        dataset = get_cs_dataset('./datasets/', args.dataset,
                                 meta_path=args.meta_path,
                                 multi_relation=(args.encoder == 'hii'),
                                 cs_relations=cs_rel_list,
                                 sparsify_topk=args.sparsify_topk,
                                 cs_full_graph=args.cs_full_graph)
    else:
        dataset = get_dataset('./datasets/', args.dataset)
    data = dataset[0]

    # 兼容: 没有 edge_index_list 的数据集补齐
    if not hasattr(data, 'edge_index_list'):
        data.edge_index_list = None
        data.num_relations = 1
    use_multi = (args.encoder == 'hii') and getattr(data, 'num_relations', 1) > 1

    data.edge_index = to_undirected(data.edge_index)
    num_features = data.x.shape[1]

    upper_edges = filter_upper_edges(data.edge_index)
    lower_edges = torch.stack([upper_edges[1], upper_edges[0]], dim=0)
    data.edge_index = torch.cat([upper_edges, lower_edges], dim=1)

    # 多关系: 逐关系做同样的 filter_upper + 双向拼接
    if use_multi:
        new_list = []
        for ei in data.edge_index_list:
            ei = to_undirected(ei)
            up = filter_upper_edges(ei)
            lo = torch.stack([up[1], up[0]], dim=0)
            new_list.append(torch.cat([up, lo], dim=1))
        data.edge_index_list = new_list
        ones_list = [torch.ones(ei.size(1)).to(device) for ei in data.edge_index_list]
        rel_info = ', '.join(f"{n}({ei.size(1)} edges)"
                             for n, ei in zip(data.relation_names, data.edge_index_list))
        print(f"[data] 多关系 R={data.num_relations}: {rel_info}")

    data = data.to(device)

    # ========== 意图 ==========
    rng_q = torch.Generator().manual_seed(args.seed)  # 动态意图采样查询节点用
    if args.intent_source == 'dynamic':
        intent_generator = QueryIntentGenerator(num_features, args.intent_dim).to(device)
        intent_vector = None  # 训练循环中每轮动态生成
        print(f"[intent] dynamic query-based, dim={args.intent_dim}")
    else:
        intent_generator = None
        intent_vector = build_intent_vector(
            args.intent_source, args.query, args.intent_dim, device, args.seed,
            encoder_name=args.intent_encoder_name,
            library_path=args.intent_library_path
        )
        print(f"[intent] source={args.intent_source}, dim={intent_vector.shape[0]}")

    # ========== 模型 (共享 encoder, 可切换 GCN / HII-GNN / 多关系) ==========
    if use_multi:
        encoder = MultiRelationEncoder(
            num_features, args.num_hidden, activation, args.intent_dim,
            num_relations=data.num_relations, num_layers=args.num_layers,
            heads=args.hii_heads, icra_heads=args.icra_heads,
            icra_dim=args.icra_dim, relation_fusion=args.relation_fusion
        ).to(device)
        print(f"[encoder] MultiRelation+{args.relation_fusion.upper()}  R={data.num_relations} "
              f"hii_heads={args.hii_heads} icra_heads={args.icra_heads} "
              f"icra_dim={args.icra_dim}")
    elif args.encoder == 'hii':
        encoder = HierarchicalIntentInjectedGNN(
            num_features, args.num_hidden, activation, args.intent_dim,
            num_layers=args.num_layers, heads=args.hii_heads
        ).to(device)
        print(f"[encoder] HII-GNN  layers={args.num_layers} heads={args.hii_heads}")
    else:
        encoder = Encoder(
            num_features, args.num_hidden, activation,
            base_model=base_model, num_layers=args.num_layers
        ).to(device)
        print("[encoder] vanilla GCN")

    contrastive_model = IntentContrastiveModel(
        encoder, args.num_hidden, args.num_proj_hidden, args.intent_dim,
        tau=args.tau, lambda_intent=effective_lambda_intent
    ).to(device)

    # 创新点四: 可疑节点识别器 (随主模型一起优化)
    suspicious_identifier = SuspiciousNodeIdentifier(
        args.num_hidden, args.intent_dim, top_k=args.top_k_suspicious
    ).to(device)

    train_params = (list(contrastive_model.parameters())
                    + list(suspicious_identifier.parameters()))
    if intent_generator is not None:
        train_params += list(intent_generator.parameters())
    optimizer_train = torch.optim.Adam(
        train_params,
        lr=args.learning_rate_train, weight_decay=args.wd_train
    )

    adv_model = IntentGuidedAdversarialModel(
        encoder, args.num_hidden, args.intent_dim, args.num_edge_hidden,
        num_cand_per_node=effective_num_cand_per_node,
        num_relations=(data.num_relations if use_multi else 1),
        cand_sources=args.cand_sources,
        cand_source_topk=args.cand_source_topk,
        cand_label_mode=args.cand_label_mode,
        cand_hard_threshold=args.cand_hard_threshold,
        cand_intent_dist_k=args.cand_intent_dist_k,
        cand_intent_dist_tau=args.cand_intent_dist_tau
    ).to(device)
    optimizer_adv = torch.optim.Adam(
        adv_model.parameters(), lr=args.learning_rate_adv,
        weight_decay=args.wd_adv
    )

    # ========== 断点续训 (checkpoint / resume) ==========
    # 每轮把最新状态原子写入单个文件(覆盖旧的); 崩溃后 --resume 从中断处继续。
    ckpt_name = (f'ckpt_{args.model_name}_{args.dataset}_{args.encoder}.pt'
                 if args.model_name else f'ckpt_{args.dataset}_{args.encoder}.pt')
    ckpt_path = args.ckpt_path or osp.join('checkpoints', ckpt_name)
    os.makedirs(osp.dirname(ckpt_path) or '.', exist_ok=True)

    def _save_ckpt(path, epoch, train_elapsed):
        ckpt = {
            'epoch': epoch,
            'train_elapsed': train_elapsed,
            'contrastive_model': contrastive_model.state_dict(),
            'suspicious_identifier': suspicious_identifier.state_dict(),
            'adv_model': adv_model.state_dict(),
            'optimizer_train': optimizer_train.state_dict(),
            'optimizer_adv': optimizer_adv.state_dict(),
            'intent_generator': (intent_generator.state_dict()
                                 if intent_generator is not None else None),
            'rng_q': rng_q.get_state(),
            'torch_rng': torch.get_rng_state(),
            'numpy_rng': np.random.get_state(),
            'python_rng': random.getstate(),
        }
        tmp = path + '.tmp'
        torch.save(ckpt, tmp)
        os.replace(tmp, path)   # 原子替换: 即使写盘时崩溃也不损坏旧存档

    def _load_ckpt(path):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        contrastive_model.load_state_dict(ckpt['contrastive_model'])
        suspicious_identifier.load_state_dict(ckpt['suspicious_identifier'])
        adv_model.load_state_dict(ckpt['adv_model'])
        optimizer_train.load_state_dict(ckpt['optimizer_train'])
        optimizer_adv.load_state_dict(ckpt['optimizer_adv'])
        if intent_generator is not None and ckpt.get('intent_generator'):
            intent_generator.load_state_dict(ckpt['intent_generator'])
        rng_q.set_state(ckpt['rng_q'])
        torch.set_rng_state(ckpt['torch_rng'])
        if ckpt.get('numpy_rng') is not None:
            np.random.set_state(ckpt['numpy_rng'])
        if ckpt.get('python_rng') is not None:
            random.setstate(ckpt['python_rng'])
        return ckpt['epoch'] + 1, ckpt.get('train_elapsed', 0.0)

    start_epoch, train_elapsed_prev = 1, 0.0
    if args.eval_only:
        if not osp.exists(ckpt_path):
            raise FileNotFoundError(f'--eval_only 需要已有检查点: {ckpt_path}')
        loaded_next_epoch, train_elapsed_prev = _load_ckpt(ckpt_path)
        start_epoch = args.num_epochs + 1
        print(f'[ckpt] eval-only 从 {ckpt_path} 加载, '
              f'检查点已完成 {loaded_next_epoch - 1} 轮, 跳过训练')
    elif args.resume and osp.exists(ckpt_path):
        start_epoch, train_elapsed_prev = _load_ckpt(ckpt_path)
        print(f'[ckpt] 从 {ckpt_path} 恢复, 已完成 {start_epoch - 1} 轮, '
              f'从第 {start_epoch} 轮继续')
    elif args.resume:
        print(f'[ckpt] 未找到存档 {ckpt_path}, 从头开始训练')

    # ========== 日志 ==========
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = osp.join(log_dir, "run_ig_results.txt")
    with open(log_file, "a") as f:
        f.write("########################################\n")
        f.write(str(vars(args)) + "\n")

    ones = torch.ones(data.num_edges).to(device)

    def _build_views_single(adv_m, cont_m, intent):
        """单关系模式: 原有逻辑——合并图上一次扰动。返回 (z_adv, z_rec, reg, fea_up, fea_lo, edge_info)。"""
        info = adv_m(data.x, data.edge_index, ones, intent)
        aw, rw, rg, cw = generate_ar_edge_weight(info, args.adv_temp, args.bias)
        za = cont_m(data.x, data.edge_index,
                    torch.cat([aw, aw], dim=0), intent)
        ce = info['cand_edges']
        if cw.numel() > 0:
            cb = torch.cat([ce, ce.flip(0)], dim=1)
            cwb = torch.cat([cw, cw], dim=0)
            rei = torch.cat([data.edge_index, cb], dim=1)
            rew = torch.cat([torch.cat([rw, rw], dim=0), cwb], dim=0)
        else:
            rei = data.edge_index
            rew = torch.cat([rw, rw], dim=0)
        zr = cont_m(data.x, rei, rew, intent)
        edge_aux = {
            'alpha': None,
            'cand_logits': info['cand_edge_logits'],
            'cand_targets': info['cand_edge_targets'],
            'num_cand': info['cand_edges'].size(1),
        }
        return za, zr, rg, info['upper_edge_fea'], info['lower_edge_fea'], edge_aux

    def _build_views_multi(adv_m, cont_m, intent):
        """多关系模式: per-relation 扰动 → 各用各的 adv/rec 边 → ICRA 融合。"""
        infos = adv_m.forward_multi(data.x, data.edge_index_list, ones_list, intent)
        adv_ws, rec_ei_list, rec_ew_list = [], [], []
        regs, fea_ups, fea_los = [], [], []
        cand_logits, cand_targets = [], []
        num_cand = 0
        for r, info in enumerate(infos):
            aw, rw, rg, cw = generate_ar_edge_weight(info, args.adv_temp, args.bias)
            adv_ws.append(torch.cat([aw, aw], dim=0))
            ei_r = data.edge_index_list[r]
            ce = info['cand_edges']
            if cw.numel() > 0:
                cb = torch.cat([ce, ce.flip(0)], dim=1)
                cwb = torch.cat([cw, cw], dim=0)
                rec_ei_list.append(torch.cat([ei_r, cb], dim=1))
                rec_ew_list.append(torch.cat([torch.cat([rw, rw]), cwb]))
            else:
                rec_ei_list.append(ei_r)
                rec_ew_list.append(torch.cat([rw, rw]))
            regs.append(rg)
            fea_ups.append(info['upper_edge_fea'])
            fea_los.append(info['lower_edge_fea'])
            cand_logits.append(info['cand_edge_logits'])
            cand_targets.append(info['cand_edge_targets'])
            num_cand += info['cand_edges'].size(1)
        za = cont_m(data.x, data.edge_index_list, adv_ws, intent)
        _, zr, alpha = cont_m.encoder.encode_per_relation(
            data.x, rec_ei_list, rec_ew_list, intent)
        reg_mean = torch.stack(regs).mean()
        edge_aux = {
            'alpha': alpha,
            'cand_logits': torch.cat(cand_logits, 0),
            'cand_targets': torch.cat(cand_targets, 0),
            'num_cand': num_cand,
        }
        return (za, zr, reg_mean, torch.cat(fea_ups, 0),
                torch.cat(fea_los, 0), edge_aux)

    build_views = _build_views_multi if use_multi else _build_views_single

    # ========== 训练 (Min-Max) ==========
    start = t() - train_elapsed_prev    # 续训时把已训耗时算进总时间
    # 阶段B: query-centric 结构采样使用与 CS 评测一致的完整图
    qc_adj = None
    if args.lambda_igqc > 0 or args.lambda_ic_spnm > 0 or args.lambda_ilssc > 0:
        qc_adj = _build_adj_list(_cs_edge_index(data), data.num_nodes)
    if args.lambda_igqc > 0:
        print(f"[IGQC] 全量图邻接表就绪, lambda={args.lambda_igqc}, "
              f"pos={args.igqc_pos} neg={args.igqc_neg} "
              f"B={args.igqc_num_queries} gate={args.igqc_intent_gate}")
    if args.lambda_ic_spnm > 0:
        print(f"[IC-SPNM] 全量图邻接表就绪, lambda={args.lambda_ic_spnm}, "
              f"pos={args.ic_spnm_pos} neg={args.ic_spnm_neg} "
              f"B={args.ic_spnm_num_queries} pool={args.ic_spnm_hard_pool} "
              f"pos_mode={args.ic_spnm_pos_mode} "
              f"frontier_ratio={args.ic_spnm_frontier_ratio} "
              f"frontier_hops={args.ic_spnm_frontier_hops} "
              f"frontier_pool={args.ic_spnm_frontier_pool} "
              f"intent_beta={args.ic_spnm_intent_beta} "
              f"struct_beta={args.ic_spnm_struct_beta} "
              f"susp_alpha={args.ic_spnm_suspicious_alpha}")
    if args.lambda_ilssc > 0:
        print(f"[ILSSC] 全量图邻接表就绪, lambda={args.lambda_ilssc}, "
              f"seed={args.ilssc_seed_size} neg={args.ilssc_neg} "
              f"B={args.ilssc_num_queries} pool={args.ilssc_hard_pool} "
              f"hops={args.ilssc_hops} "
              f"frontier_pool={args.ilssc_frontier_pool} "
              f"conn_beta={args.ilssc_conn_beta} "
              f"sim_beta={args.ilssc_sim_beta} "
              f"high_order_beta={args.ilssc_high_order_beta} "
              f"proto_alpha={args.ilssc_proto_alpha} "
              f"use_intent_dist={args.ilssc_use_intent_dist} "
              f"dist_k={args.intent_dist_k} "
              f"dist_tau={args.intent_dist_tau} "
              f"dist_beta={args.intent_dist_beta} "
              f"dist_proto_mode={args.intent_dist_proto_mode} "
              f"dist_stable={args.intent_dist_stable} "
              f"dist_update_interval={args.intent_dist_update_interval} "
              f"dist_ema={args.intent_dist_ema} "
              f"dist_conf_tau={args.intent_dist_conf_tau} "
              f"dist_min_conf={args.intent_dist_min_conf}")
    intent_dist_memory = None
    if args.ilssc_use_intent_dist and args.intent_dist_stable:
        intent_dist_memory = IntentDistributionMemory(args, device)
        print(f"[SCID-ILSSC] stable intent distribution memory enabled, "
              f"K={args.intent_dist_k}, interval={args.intent_dist_update_interval}, "
              f"ema={args.intent_dist_ema}, anchor_pool={args.intent_dist_anchor_pool}")

    prev = t()
    epoch = start_epoch - 1             # 续训跳过循环时兜底: 最后已完成的轮次
    for epoch in range(start_epoch, args.num_epochs + 1):
        # 动态意图: 每轮采样一个查询节点, 由其特征生成意图(端到端训练生成器)
        if intent_generator is not None:
            q_epoch = torch.randint(0, data.num_nodes, (1,),
                                    generator=rng_q).item()
            intent_vector = intent_generator(data.x[q_epoch])
        # Phase1 把意图当固定输入(detach), 仅 Phase2 训练意图生成器
        intent_p1 = intent_vector.detach()

        # 周期性刷新候选新边(用当前嵌入重选, 第 1 轮在 forward 内部自动生成)
        if args.cand_refresh_interval > 0 and epoch > 1 and (epoch - 1) % args.cand_refresh_interval == 0:
            with torch.no_grad():
                if use_multi:
                    adv_model.refresh_candidate_edges_multi(
                        data.x, data.edge_index_list, ones_list, intent_p1)
                else:
                    adv_model.refresh_candidate_edges(
                        data.x, data.edge_index, ones, intent_p1)

        # ----- Phase 1: 对抗生成器最大化损失 -----
        adv_model.train()
        adv_model.zero_grad()
        contrastive_model.eval()

        z_adv, z_rec, reg, fea_up, fea_lo, _ = build_views(
            adv_model, contrastive_model, intent_p1)

        loss, _ = contrastive_model.total_loss(
            z_adv, z_rec, intent_p1, reg,
            reg_lambda=args.reg_lambda, adv_lambda=effective_adv_lambda,
            edge_fea_adv=fea_up, edge_fea_rec=fea_lo
        )
        (-loss).backward()
        optimizer_adv.step()

        # ----- Phase 2: 主模型(+意图生成器)最小化损失 -----
        contrastive_model.train()
        optimizer_train.zero_grad()
        adv_model.eval()

        # 动态模式: 重新从同一个查询节点生成意图(这次保留梯度, 训练生成器)
        if intent_generator is not None:
            intent_vector = intent_generator(data.x[q_epoch])

        z_adv, z_rec, reg, fea_up, fea_lo, aux = build_views(
            adv_model, contrastive_model, intent_vector)
        alpha = aux.get('alpha') if aux is not None else None

        # 创新点四: 识别可疑节点(用重构视图表示, 保留梯度以训练识别器)
        susp_idx, node_score = suspicious_identifier(
            z_rec, data.edge_index, intent_vector
        )
        # 训练识别器: 可疑分对齐两视图发散度(被篡改节点发散更大)
        with torch.no_grad():
            divergence = 1.0 - F.cosine_similarity(z_adv, z_rec, dim=-1)
            divergence = divergence / (divergence.max() + 1e-8)
        l_susp = F.mse_loss(node_score, divergence)

        # 阶段B IGQC: 采样查询批, 意图对齐 top-k 正邻居 + 随机负样本, 逐q生成意图
        igqc_args = None
        if qc_adj is not None and args.lambda_igqc > 0:
            B, m, n = args.igqc_num_queries, args.igqc_pos, args.igqc_neg
            N = data.num_nodes
            dev = data.x.device
            # 采样阶段用于筛正样本的意图对齐空间(detach, 不回传到筛选)
            with torch.no_grad():
                z_align = F.normalize(
                    contrastive_model.intent_proj(z_rec), dim=-1)
            q_first = q_epoch if intent_generator is not None else int(
                torch.randint(0, N, (1,), generator=rng_q).item())
            q_list = [q_first] + torch.randint(
                0, N, (B - 1,), generator=rng_q).tolist()
            pos_rows, neg_rows, valid_q, intent_list = [], [], [], []
            for qi in q_list:
                nbrs = list(qc_adj[qi])
                if len(nbrs) == 0:
                    continue
                # 逐q意图(保留梯度以端到端训练生成器)
                if intent_generator is not None:
                    iq = intent_generator(data.x[qi])
                else:
                    iq = intent_vector
                # 意图对齐 top-m: 门控开且邻居够多时按对齐度筛掉跨类邻居
                if args.igqc_intent_gate and len(nbrs) > m:
                    nbr_t = torch.tensor(nbrs, device=dev)
                    with torch.no_grad():
                        iq_n = F.normalize(iq.detach(), dim=-1)
                        align = z_align[nbr_t] @ iq_n
                        top = torch.topk(align, m).indices
                    pos_rows.append(nbr_t[top].cpu().numpy())
                else:
                    pos_rows.append(np.random.choice(
                        nbrs, size=m, replace=len(nbrs) < m))
                neg_rows.append(np.random.randint(0, N, size=n))
                valid_q.append(qi)
                intent_list.append(iq)
            if valid_q:
                intent_batch = torch.stack(intent_list)
                igqc_args = dict(
                    z=z_rec,
                    q_idx=torch.tensor(valid_q, device=dev),
                    intent_batch=intent_batch,
                    pos_idx=torch.tensor(np.stack(pos_rows), device=dev),
                    neg_idx=torch.tensor(np.stack(neg_rows), device=dev),
                    gate=args.igqc_intent_gate,
                )

        ic_spnm_args = None
        ic_spnm_stats = None
        if qc_adj is not None and args.lambda_ic_spnm > 0:
            ic_spnm_args, ic_spnm_stats = build_ic_spnm_args(
                z_rec=z_rec,
                data=data,
                q_epoch=(q_epoch if intent_generator is not None else 0),
                intent_vector=intent_vector,
                intent_generator=intent_generator,
                contrastive_model=contrastive_model,
                qc_adj=qc_adj,
                node_score=node_score,
                args=args,
                rng_q=rng_q,
            )

        ilssc_args = None
        ilssc_stats = None
        eff_lambda_ilssc = effective_ilssc_lambda(args, epoch)
        if qc_adj is not None and eff_lambda_ilssc > 0:
            if intent_dist_memory is not None:
                intent_dist_memory.maybe_update(epoch, z_rec, node_score, rng_q)
            ilssc_args, ilssc_stats = build_ilssc_args(
                z_rec=z_rec,
                data=data,
                q_epoch=(q_epoch if intent_generator is not None else 0),
                intent_vector=intent_vector,
                intent_generator=intent_generator,
                contrastive_model=contrastive_model,
                qc_adj=qc_adj,
                node_score=node_score,
                args=args,
                rng_q=rng_q,
                intent_dist_memory=intent_dist_memory,
            )

        cand_rec_args = None
        if aux is not None:
            cand_rec_args = {
                'logits': aux.get('cand_logits'),
                'targets': aux.get('cand_targets'),
            }

        model_loss, loss_info = contrastive_model.total_loss(
            z_adv, z_rec, intent_vector, reg,
            reg_lambda=args.reg_lambda, adv_lambda=effective_adv_lambda,
            edge_fea_adv=fea_up, edge_fea_rec=fea_lo,
            suspicious_idx=(None if args.no_suspicious_kl else susp_idx),
            lambda_rec=effective_lambda_rec,
            igqc_args=igqc_args, lambda_igqc=args.lambda_igqc,
            ic_spnm_args=ic_spnm_args,
            lambda_ic_spnm=args.lambda_ic_spnm,
            ilssc_args=ilssc_args,
            lambda_ilssc=eff_lambda_ilssc,
            cand_rec_args=cand_rec_args,
            lambda_cand_bce=effective_lambda_cand_bce
        )
        model_loss = model_loss + effective_lambda_rec * l_susp

        # ICRA 关系熵正则: 最大化关系权重熵, 防止融合塌缩到单一 meta-path
        if use_multi and args.lambda_rel_entropy > 0 and alpha is not None:
            ent = -(alpha * (alpha + 1e-9).log()).sum(dim=0).mean()
            model_loss = model_loss - args.lambda_rel_entropy * ent

        model_loss.backward()
        optimizer_train.step()

        now = t()
        msg = (
            f'(T) | Epoch={epoch:03d}, loss={model_loss:.4f}, '
            f'con={loss_info["contrastive"]:.4f}, '
            f'intent={loss_info["intent"]:.4f}, '
            f'rec={loss_info["reconstruction"]:.4f}, '
            + (f'cand_bce={loss_info["cand_bce"]:.4f}, '
               f'n_cand={loss_info["num_cand_edges"]}, '
               if effective_lambda_cand_bce > 0 else '')
            + (f'igqc={loss_info["igqc"]:.4f}, ' if args.lambda_igqc > 0 else '')
            + (f'ic_spnm={loss_info["ic_spnm"]:.4f}, '
               if args.lambda_ic_spnm > 0 else '')
            + (f'ic_q={ic_spnm_stats["valid_q"]}, '
               f'ic_pos={ic_spnm_stats["pos_align"]:.3f}, '
               f'ic_neg={ic_spnm_stats["neg_sim"]:.3f}, '
               f'ic_frontier={ic_spnm_stats["frontier_frac"]:.2f}, '
               f'ic_hop={ic_spnm_stats["pos_hop"]:.2f}, '
               f'ic_pos_std={ic_spnm_stats["pos_align_std"]:.3f}, '
               f'ic_uniq={ic_spnm_stats["pos_unique"]:.1f}, '
               if args.lambda_ic_spnm > 0 and ic_spnm_stats is not None else '')
            + (f'ilssc={loss_info["ilssc"]:.4f}, '
               f'il_lam={eff_lambda_ilssc:.4f}, '
               if args.lambda_ilssc > 0 else '')
            + (f'il_q={ilssc_stats["valid_q"]}, '
               f'il_seed={ilssc_stats["seed_align"]:.3f}, '
               f'il_sim={ilssc_stats["seed_sim"]:.3f}, '
               f'il_conn={ilssc_stats["seed_conn"]:.3f}, '
               f'il_dist={ilssc_stats["seed_dist"]:.3f}, '
               f'il_high={ilssc_stats["seed_high"]:.3f}, '
               f'il_conf_q={ilssc_stats["q_conf"]:.3f}, '
               f'il_conf_seed={ilssc_stats["seed_conf"]:.3f}, '
               f'il_gate_seed={ilssc_stats["seed_gate"]:.3f}, '
               f'il_uniq={ilssc_stats["seed_unique"]:.1f}, '
               f'il_neg={ilssc_stats["neg_sim"]:.3f}, '
               f'il_neg_dist={ilssc_stats["neg_dist"]:.3f}, '
               f'il_neg_high={ilssc_stats["neg_high"]:.3f}, '
               f'il_conf_neg={ilssc_stats["neg_conf"]:.3f}, '
               f'il_gate_neg={ilssc_stats["neg_gate"]:.3f}, '
               f'id_upd={ilssc_stats["id_updated"]}, '
               if args.lambda_ilssc > 0 and ilssc_stats is not None else '')
            + f'this epoch {now - prev:.4f}, total {now - start:.4f}'
        )
        # 多关系: 周期性打印 ICRA 各关系平均权重, 监控是否塌缩
        if use_multi and alpha is not None and (epoch % 10 == 0 or epoch == 1):
            with torch.no_grad():
                rel_w = alpha.mean(dim=(1, 2))
            msg += '  alpha=[' + ', '.join(
                f'{n}:{w:.3f}' for n, w in
                zip(data.relation_names, rel_w.tolist())) + ']'
        print(msg)
        prev = now

        # 每轮保存最新检查点(覆盖旧的), 供中断后 --resume 续训
        if args.ckpt_interval > 0 and epoch % args.ckpt_interval == 0:
            _save_ckpt(ckpt_path, epoch, now - start)

    if not args.eval_only and epoch >= start_epoch:
        _save_ckpt(ckpt_path, epoch, t() - start)
        print(f'[ckpt] final checkpoint saved -> {ckpt_path}')

    train_time = t() - start            # 训练总耗时
    eval_start = t()                    # 评估(测试)起点

    # ========== 评估 (节点分类台架) ==========
    # 节点分类用 "平均意图" 编码一次 (非查询驱动任务)
    with torch.no_grad():
        if intent_generator is not None:
            avg_intent = intent_generator(data.x.mean(dim=0))
        else:
            avg_intent = intent_vector
        if use_multi:
            emb = contrastive_model(data.x, data.edge_index_list, ones_list,
                                    avg_intent)
        else:
            emb = contrastive_model(data.x, data.edge_index, ones, avg_intent)
        _, node_boost = suspicious_identifier(emb, data.edge_index, avg_intent)
    node_boost_eval = None if args.no_suspicious_boost else node_boost

    # ========== Actor-Critic 对抗图生成器 (§7.2 Step4, 自监督) ==========
    # 主编码器收敛后单独训练, 用冻结的 emb; 不动上面的 Min-Max 主循环。
    builder = None
    if args.use_actor_critic:
        # AC 训练与评测用同一张图(全量合并 meta-path), 避免 train/eval 图不一致
        ac_adj = _build_adj_list(_cs_edge_index(data), data.x.size(0))
        builder = ActorCriticCommunityBuilder(
            emb.size(1), avg_intent.size(0), max_size=args.ac_max_size
        ).to(device)
        print(f'[AC] 训练 Actor-Critic 对抗图生成器 ({args.ac_epochs} 轮)...')
        train_actor_critic(
            builder, emb.detach(), ac_adj, avg_intent.detach(),
            node_boost=node_boost_eval, epochs=args.ac_epochs, lr=args.ac_lr,
            seed=args.seed
        )

    micro_f1_mean, micro_f1_std, macro_f1_mean, macro_f1_std, acc_mean, acc_std = \
        label_classification(emb, data, args.dataset, ratio=0.1)

    formatted_result = (
        f"micro_f1: {micro_f1_mean:.2f}±{micro_f1_std:.2f}, "
        f"macro_f1: {macro_f1_mean:.2f}±{macro_f1_std:.2f}, "
        f"acc: {acc_mean:.2f}±{acc_std:.2f}"
    )
    print(formatted_result)

    # ========== 评估 (社区搜索指标) ==========
    # 构建/加载固定查询节点 (对齐 CLUHCS 40 查询协议)
    qf = args.query_file
    if qf is None:
        qf = osp.join(log_dir, f"queries_{args.dataset}_{args.cs_num_queries}.txt")
    fixed_queries = build_fixed_queries(
        data, num_queries=args.cs_num_queries, seed=args.seed, query_file=qf
    )

    cs_eval_edge_index = _cs_edge_index(data)
    if args.eval_perturb_mode != 'none' and args.eval_perturb_rate > 0:
        cs_eval_edge_index = perturb_edge_index(
            cs_eval_edge_index, data.num_nodes,
            mode=args.eval_perturb_mode,
            rate=args.eval_perturb_rate,
            seed=args.eval_perturb_seed
        )
        data.cs_edge_index = cs_eval_edge_index
        print(f"[eval-perturb] mode={args.eval_perturb_mode} "
              f"rate={args.eval_perturb_rate} seed={args.eval_perturb_seed} "
              f"edges={cs_eval_edge_index.size(1)}")

    if intent_generator is not None and args.encoder == 'hii':
        contrastive_model.eval()
        intent_generator.eval()
        # 多关系: 编码器吃 edge_index_list + ones_list; 单图扰动评估时吃扰动后的合并图
        if use_multi:
            ew_arg = ones_list
            ei_arg = data.edge_index_list
        elif args.eval_perturb_mode != 'none' and args.eval_perturb_rate > 0:
            ew_arg = torch.ones(cs_eval_edge_index.size(1), device=device)
            ei_arg = cs_eval_edge_index
        else:
            ew_arg = ones
            ei_arg = None
        cs_results = community_search_dynamic(
            contrastive_model, intent_generator, data, ew_arg,
            topk=cs_topk,
            num_queries=args.cs_num_queries, seed=args.seed,
            node_boost=node_boost_eval, boost_factor=args.suspicious_boost,
            queries=fixed_queries, edge_index=ei_arg,
            intent_proj_fn=contrastive_model.intent_proj,
            intent_rerank_alpha=args.intent_rerank_alpha
        )
        cs_greedy = community_search_greedy_dynamic(
            contrastive_model, intent_generator, data, ew_arg,
            w_list=cs_w_list,
            num_queries=args.cs_num_queries, seed=args.seed,
            compute_structure=args.compute_structure_metrics,
            node_boost=node_boost_eval, boost_factor=args.suspicious_boost,
            queries=fixed_queries, edge_index=ei_arg,
            intent_proj_fn=contrastive_model.intent_proj,
            intent_rerank_alpha=args.intent_rerank_alpha,
            greedy_patience=args.greedy_patience,
            greedy_min_gain_tol=args.greedy_min_gain_tol,
            greedy_size_penalty=args.greedy_size_penalty,
            balance_alpha=args.greedy_balance_alpha,
            greedy_max_size=args.greedy_max_size,
            greedy_adaptive_cap_alpha=args.greedy_adaptive_cap_alpha,
            greedy_adaptive_cap_floor=args.greedy_adaptive_cap_floor,
            greedy_trace_cap_ratio=args.greedy_trace_cap_ratio,
            frontier_batch_size=args.frontier_batch_size,
            include_query_in_pred=args.include_query_in_pred,
            greedy_connectivity_boost=args.greedy_connectivity_boost,
            greedy_select_mode=args.greedy_select_mode,
            trace_early_stop_w=trace_early_stop_w,
            greedy_init_seed_size=args.greedy_init_seed_size,
            greedy_init_seed_hops=args.greedy_init_seed_hops,
            greedy_init_seed_conn_beta=args.greedy_init_seed_conn_beta,
            greedy_init_seed_min_sim=args.greedy_init_seed_min_sim,
            hse_high_order_beta=args.greedy_high_order_beta,
            hse_comm_cohesion_beta=args.greedy_comm_cohesion_beta,
            hse_boundary_gamma=args.greedy_boundary_gamma,
            hse_pool_size=args.greedy_hse_pool_size,
            hse_comm_direct_beta=args.greedy_comm_direct_beta,
            hse_normalize=args.greedy_hse_normalize,
            hse_density=args.greedy_hse_density,
            recall_expand_size=args.greedy_recall_expand_size,
            recall_expand_min_sim_delta=args.greedy_recall_min_sim_delta
        )
    else:
        cs_results = community_search(emb, data, topk=cs_topk,
                                      node_boost=node_boost_eval,
                                      boost_factor=args.suspicious_boost,
                                      queries=fixed_queries)
        cs_greedy = community_search_greedy(emb, data,
                                            w_list=cs_w_list,
                                            seed=args.seed,
                                            compute_structure=args.compute_structure_metrics,
                                            node_boost=node_boost_eval,
                                            boost_factor=args.suspicious_boost,
                                            queries=fixed_queries,
                                            greedy_patience=args.greedy_patience,
                                            greedy_min_gain_tol=args.greedy_min_gain_tol,
                                            greedy_size_penalty=args.greedy_size_penalty,
                                            balance_alpha=args.greedy_balance_alpha,
                                            greedy_max_size=args.greedy_max_size,
                                            greedy_adaptive_cap_alpha=args.greedy_adaptive_cap_alpha,
                                            greedy_adaptive_cap_floor=args.greedy_adaptive_cap_floor,
                                            greedy_trace_cap_ratio=args.greedy_trace_cap_ratio,
                                            frontier_batch_size=args.frontier_batch_size,
                                            include_query_in_pred=args.include_query_in_pred,
                                            greedy_connectivity_boost=args.greedy_connectivity_boost,
                                            greedy_select_mode=args.greedy_select_mode,
                                            trace_early_stop_w=trace_early_stop_w,
                                            greedy_init_seed_size=args.greedy_init_seed_size,
                                            greedy_init_seed_hops=args.greedy_init_seed_hops,
                                            greedy_init_seed_conn_beta=args.greedy_init_seed_conn_beta,
                                            greedy_init_seed_min_sim=args.greedy_init_seed_min_sim,
                                            hse_high_order_beta=args.greedy_high_order_beta,
                                            hse_comm_cohesion_beta=args.greedy_comm_cohesion_beta,
                                            hse_boundary_gamma=args.greedy_boundary_gamma,
                                            hse_pool_size=args.greedy_hse_pool_size,
                                            hse_comm_direct_beta=args.greedy_comm_direct_beta,
                                            hse_normalize=args.greedy_hse_normalize,
                                            hse_density=args.greedy_hse_density,
                                            recall_expand_size=args.greedy_recall_expand_size,
                                            recall_expand_min_sim_delta=args.greedy_recall_min_sim_delta)

    # Actor-Critic 社区搜索评测 (启用时)
    cs_rl = None
    if args.use_actor_critic and builder is not None:
        sweep_sizes = None
        if args.ac_size_sweep:
            sweep_sizes = [int(s) for s in args.ac_size_sweep.split(',')]
        cs_rl = community_search_rl(
            builder, emb, data, fixed_queries,
            node_boost=node_boost_eval, intent=avg_intent, max_sizes=sweep_sizes,
            oracle_size=True
        )

    eval_time = t() - eval_start        # 评估(测试)总耗时
    total_time = t() - program_start    # 总运行时间

    timing_result = (
        f"[timing] 训练时间={train_time:.2f}s, "
        f"测试时间={eval_time:.2f}s, "
        f"总运行时间={total_time:.2f}s "
        f"(每轮均值={train_time / max(1, args.num_epochs):.3f}s)"
    )
    print(timing_result)

    with open(log_file, 'a') as f:
        f.write('epoch: ' + str(epoch) + '\n')
        f.write(formatted_result + '\n')
        for k, metrics in cs_results.items():
            f.write(f"  CS@{k}: P={metrics['precision']:.2f} "
                    f"R={metrics['recall']:.2f} "
                    f"F1={metrics['f1']:.2f} "
                    f"Jaccard={metrics['jaccard']:.2f}\n")
        for w, metrics in cs_greedy.items():
            line = (f"  CS-greedy@w={w}: P={metrics['precision']:.2f} "
                    f"R={metrics['recall']:.2f} "
                    f"F1={metrics['f1']:.2f} "
                    f"Jaccard={metrics['jaccard']:.2f} "
                    f"size={metrics['avg_size']:.1f}")
            if metrics.get('density', 0) > 0:
                line += (f" den={metrics['density']:.3f}"
                         f" cond={metrics['conductance']:.3f}"
                         f" diam={metrics['diameter']:.2f}")
            f.write(line + '\n')
        if cs_rl is not None:
            oracle_m = cs_rl.pop('oracle', None)          # oracle-size 对照线
            if 'precision' in cs_rl:                     # 单 size: 扁平 dict
                f.write(f"  CS-rl: P={cs_rl['precision']:.2f} "
                        f"R={cs_rl['recall']:.2f} "
                        f"F1={cs_rl['f1']:.2f} "
                        f"Jaccard={cs_rl['jaccard']:.2f} "
                        f"size={cs_rl['avg_size']:.1f}\n")
            else:                                         # 扫描: {size: dict}
                for ms in sorted(cs_rl):
                    m = cs_rl[ms]
                    f.write(f"  CS-rl@max_size={ms}: P={m['precision']:.2f} "
                            f"R={m['recall']:.2f} "
                            f"F1={m['f1']:.2f} "
                            f"Jaccard={m['jaccard']:.2f} "
                            f"size={m['avg_size']:.1f}\n")
            if oracle_m is not None:
                f.write(f"  CS-rl@oracle-size: P={oracle_m['precision']:.2f} "
                        f"R={oracle_m['recall']:.2f} "
                        f"F1={oracle_m['f1']:.2f} "
                        f"Jaccard={oracle_m['jaccard']:.2f} "
                        f"size={oracle_m['avg_size']:.1f}\n")
        f.write(timing_result + '\n')
    print('-----------------')
