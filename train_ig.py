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
                  community_search_rl, build_fixed_queries, _build_adj_list)
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
                        encoder_name='paraphrase-multilingual-MiniLM-L12-v2'):
    """构建意图向量。encoder 不可用时回退为固定随机向量。"""
    if source == 'encoder':
        try:
            from adversarial_intent_encoder import SimpleIntentEncoder
            enc = SimpleIntentEncoder(intent_dim=intent_dim,
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
    parser.add_argument('--adv_temp', type=float, default=1.0)
    parser.add_argument('--bias', type=float, default=0.0001)
    parser.add_argument('--meta_path', type=str, default=None,
                        help='For ACM/DBLP/IMDB: single meta-path file (e.g. pap.npz). None=merge all.')
    parser.add_argument('--num_cand_per_node', type=int, default=5,
                        help='Candidate new edges per node for reconstruction view')
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
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='检查点路径; None=checkpoints/ckpt_{dataset}_{encoder}.pt')
    parser.add_argument('--ckpt_interval', type=int, default=1,
                        help='每 N 轮保存一次最新检查点(覆盖旧的); 0=不保存')
    # 多关系 + ICRA 融合 (创新点: 意图条件化关系注意力)
    parser.add_argument('--icra_heads', type=int, default=4,
                        help='ICRA 关系注意力头数')
    parser.add_argument('--icra_dim', type=int, default=128,
                        help='ICRA 注意力投影维度')
    parser.add_argument('--cs_relations', type=str, default=None,
                        help='逗号分隔 meta-path 名(如 pap,psp); None=全部关系')
    parser.add_argument('--lambda_rel_entropy', type=float, default=0.0,
                        help='ICRA 关系熵正则权重(0=关; >0 防止塌缩到单一关系)')
    parser.add_argument('--sparsify_topk', type=int, default=None,
                        help='稠密 meta-path top-k 稀疏化(每节点保留k个最强邻居); '
                             'None=不稀疏。ACM-PSP/DBLP-APCPA 等超稠密关系必需(否则OOM)')
    args = parser.parse_args()

    # ========== 运行日志: 全部 print/stderr 同时写入文件 (tee) ==========
    os.makedirs("log", exist_ok=True)
    run_log_path = osp.join(
        "log", f"run_{args.dataset}_{datetime.now():%Y%m%d_%H%M%S}.log")
    _run_log_fh = open(run_log_path, "a", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _run_log_fh)
    sys.stderr = _Tee(sys.__stderr__, _run_log_fh)
    print(f"[log] 运行日志写入: {run_log_path}")

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
                                 sparsify_topk=args.sparsify_topk)
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
            args.intent_source, args.query, args.intent_dim, device, args.seed
        )
        print(f"[intent] source={args.intent_source}, dim={intent_vector.shape[0]}")

    # ========== 模型 (共享 encoder, 可切换 GCN / HII-GNN / 多关系) ==========
    if use_multi:
        encoder = MultiRelationEncoder(
            num_features, args.num_hidden, activation, args.intent_dim,
            num_relations=data.num_relations, num_layers=args.num_layers,
            heads=args.hii_heads, icra_heads=args.icra_heads,
            icra_dim=args.icra_dim
        ).to(device)
        print(f"[encoder] MultiRelation+ICRA  R={data.num_relations} "
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
        tau=args.tau, lambda_intent=args.lambda_intent
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
        num_cand_per_node=args.num_cand_per_node,
        num_relations=(data.num_relations if use_multi else 1)
    ).to(device)
    optimizer_adv = torch.optim.Adam(
        adv_model.parameters(), lr=args.learning_rate_adv,
        weight_decay=args.wd_adv
    )

    # ========== 断点续训 (checkpoint / resume) ==========
    # 每轮把最新状态原子写入单个文件(覆盖旧的); 崩溃后 --resume 从中断处继续。
    ckpt_path = args.ckpt_path or osp.join(
        'checkpoints', f'ckpt_{args.dataset}_{args.encoder}.pt')
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
    if args.resume and osp.exists(ckpt_path):
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
        return za, zr, rg, info['upper_edge_fea'], info['lower_edge_fea'], None

    def _build_views_multi(adv_m, cont_m, intent):
        """多关系模式: per-relation 扰动 → 各用各的 adv/rec 边 → ICRA 融合。"""
        infos = adv_m.forward_multi(data.x, data.edge_index_list, ones_list, intent)
        adv_ws, rec_ei_list, rec_ew_list = [], [], []
        regs, fea_ups, fea_los = [], [], []
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
        za = cont_m(data.x, data.edge_index_list, adv_ws, intent)
        _, zr, alpha = cont_m.encoder.encode_per_relation(
            data.x, rec_ei_list, rec_ew_list, intent)
        reg_mean = torch.stack(regs).mean()
        return (za, zr, reg_mean, torch.cat(fea_ups, 0),
                torch.cat(fea_los, 0), alpha)

    build_views = _build_views_multi if use_multi else _build_views_single

    # ========== 训练 (Min-Max) ==========
    start = t() - train_elapsed_prev    # 续训时把已训耗时算进总时间
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
            reg_lambda=args.reg_lambda, adv_lambda=args.adv_lambda,
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

        z_adv, z_rec, reg, fea_up, fea_lo, alpha = build_views(
            adv_model, contrastive_model, intent_vector)

        # 创新点四: 识别可疑节点(用重构视图表示, 保留梯度以训练识别器)
        susp_idx, node_score = suspicious_identifier(
            z_rec, data.edge_index, intent_vector
        )
        # 训练识别器: 可疑分对齐两视图发散度(被篡改节点发散更大)
        with torch.no_grad():
            divergence = 1.0 - F.cosine_similarity(z_adv, z_rec, dim=-1)
            divergence = divergence / (divergence.max() + 1e-8)
        l_susp = F.mse_loss(node_score, divergence)

        model_loss, loss_info = contrastive_model.total_loss(
            z_adv, z_rec, intent_vector, reg,
            reg_lambda=args.reg_lambda, adv_lambda=args.adv_lambda,
            edge_fea_adv=fea_up, edge_fea_rec=fea_lo,
            suspicious_idx=susp_idx,
            lambda_rec=args.lambda_rec
        )
        model_loss = model_loss + args.lambda_rec * l_susp

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
            f'this epoch {now - prev:.4f}, total {now - start:.4f}'
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

    # ========== Actor-Critic 对抗图生成器 (§7.2 Step4, 自监督) ==========
    # 主编码器收敛后单独训练, 用冻结的 emb; 不动上面的 Min-Max 主循环。
    builder = None
    if args.use_actor_critic:
        ac_adj = _build_adj_list(data.edge_index, data.x.size(0))
        builder = ActorCriticCommunityBuilder(
            emb.size(1), avg_intent.size(0), max_size=args.ac_max_size
        ).to(device)
        print(f'[AC] 训练 Actor-Critic 对抗图生成器 ({args.ac_epochs} 轮)...')
        train_actor_critic(
            builder, emb.detach(), ac_adj, avg_intent.detach(),
            node_boost=node_boost, epochs=args.ac_epochs, lr=args.ac_lr,
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

    if intent_generator is not None and args.encoder == 'hii':
        contrastive_model.eval()
        intent_generator.eval()
        # 多关系: 编码器吃 edge_index_list + ones_list; 单图: 合并图
        ew_arg = ones_list if use_multi else ones
        ei_arg = data.edge_index_list if use_multi else None
        cs_results = community_search_dynamic(
            contrastive_model, intent_generator, data, ew_arg,
            topk=(10, 20, 50, 'oracle'),
            num_queries=args.cs_num_queries, seed=args.seed,
            node_boost=node_boost, boost_factor=args.suspicious_boost,
            queries=fixed_queries, edge_index=ei_arg
        )
        cs_greedy = community_search_greedy_dynamic(
            contrastive_model, intent_generator, data, ew_arg,
            w_list=(0.0, 0.1, 0.2, 0.3, 0.5),
            num_queries=args.cs_num_queries, seed=args.seed,
            node_boost=node_boost, boost_factor=args.suspicious_boost,
            queries=fixed_queries, edge_index=ei_arg
        )
    else:
        cs_results = community_search(emb, data, topk=(10, 20, 50, 'oracle'),
                                      node_boost=node_boost,
                                      boost_factor=args.suspicious_boost,
                                      queries=fixed_queries)
        cs_greedy = community_search_greedy(emb, data,
                                            w_list=(0.0, 0.1, 0.2, 0.3, 0.5),
                                            seed=args.seed,
                                            node_boost=node_boost,
                                            boost_factor=args.suspicious_boost,
                                            queries=fixed_queries)

    # Actor-Critic 社区搜索评测 (启用时)
    cs_rl = None
    if args.use_actor_critic and builder is not None:
        sweep_sizes = None
        if args.ac_size_sweep:
            sweep_sizes = [int(s) for s in args.ac_size_sweep.split(',')]
        cs_rl = community_search_rl(
            builder, emb, data, fixed_queries,
            node_boost=node_boost, intent=avg_intent, max_sizes=sweep_sizes
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
        f.write(timing_result + '\n')
    print('-----------------')
