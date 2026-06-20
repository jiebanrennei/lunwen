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

import argparse
import os.path as osp
from time import perf_counter as t

from utils import set_everything, get_dataset

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected

from model import Encoder
from ig_model import IntentGuidedAdversarialModel, IntentContrastiveModel
from eval import label_classification

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def generate_ar_edge_weight(edge_info, temperature=1.0, bias=0.0001):
    """
    生成对抗视图(减边)与重构视图(加边/恢复)的互补边权重。

    与 EDA-GCL 的 generate_aug_edge_weight 同构,但语义不同:
    - adv_edge_weight: 对抗视图,保留在对抗压力下"存活"的子集 (减边/稀疏化)
    - rec_edge_weight: 重构视图,保留互补子集,即意图认为应恢复的连接 (加边)
    二者经硬阈值后边集互斥,实现"互补双视图"。

    注: 真正的"加全新边"需要候选边生成 (O(N^2)),此核心版先在现有边上
    做意图引导的互补划分;新增边作为后续扩展。
    """
    device = edge_info['upper_edge_logits'].device
    logits_shape = edge_info['upper_edge_logits'].size()

    # ---- 对抗视图权重 (上三角 logits) ----
    eps_adv = (bias - (1 - bias)) * torch.rand(logits_shape) + (1 - bias)
    gate_adv = (torch.log(eps_adv) - torch.log(1 - eps_adv)).to(device)
    adv_edge_weight = torch.sigmoid(
        (gate_adv + edge_info['upper_edge_logits']) / temperature
    ).squeeze()

    # ---- 重构视图权重 (下三角 logits) ----
    eps_rec = (bias - (1 - bias)) * torch.rand(logits_shape) + (1 - bias)
    gate_rec = (torch.log(eps_rec) - torch.log(1 - eps_rec)).to(device)
    rec_edge_weight = torch.sigmoid(
        (gate_rec + edge_info['lower_edge_logits']) / temperature
    ).squeeze()

    # ---- 对抗正则化: 鼓励/控制两视图差异 ----
    reg = F.l1_loss(adv_edge_weight, rec_edge_weight)

    # ---- 互补硬阈值: 每条边只归属一个视图 ----
    noise = torch.randn_like(adv_edge_weight) * 1e-7
    mask = (adv_edge_weight + noise) > (rec_edge_weight + noise)
    adv_edge_weight = torch.where(mask, adv_edge_weight, torch.tensor(0.).to(device))
    rec_edge_weight = torch.where(~mask, rec_edge_weight, torch.tensor(0.).to(device))

    return adv_edge_weight, rec_edge_weight, reg


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
                        choices=['random', 'encoder'])
    parser.add_argument('--query', type=str,
                        default='找出在社交网络上通过隐蔽连接协同的群体')
    parser.add_argument('--adv_temp', type=float, default=1.0)
    parser.add_argument('--bias', type=float, default=0.0001)
    args = parser.parse_args()

    print("Using CPU")
    set_everything(args.seed)

    activation = ({
        'relu': F.relu,
        'prelu': nn.PReLU(),
        'rrelu': nn.RReLU(),
        'leakyrelu': nn.LeakyReLU(),
        'gelu': nn.GELU()
    })[args.activation]
    base_model = ({'GCNConv': GCNConv})[args.base_model]

    device = torch.device('cpu')

    # ========== 数据 ==========
    dataset = get_dataset('./datasets/', args.dataset)
    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index)
    num_features = data.x.shape[1]

    upper_edges = filter_upper_edges(data.edge_index)
    lower_edges = torch.stack([upper_edges[1], upper_edges[0]], dim=0)
    data.edge_index = torch.cat([upper_edges, lower_edges], dim=1)
    data = data.to(device)

    # ========== 意图向量 ==========
    intent_vector = build_intent_vector(
        args.intent_source, args.query, args.intent_dim, device, args.seed
    )
    print(f"[intent] source={args.intent_source}, dim={intent_vector.shape[0]}")

    # ========== 模型 (共享 encoder) ==========
    encoder = Encoder(
        num_features, args.num_hidden, activation,
        base_model=base_model, num_layers=args.num_layers
    ).to(device)

    contrastive_model = IntentContrastiveModel(
        encoder, args.num_hidden, args.num_proj_hidden, args.intent_dim,
        tau=args.tau, lambda_intent=args.lambda_intent
    ).to(device)
    optimizer_train = torch.optim.Adam(
        contrastive_model.parameters(), lr=args.learning_rate_train,
        weight_decay=args.wd_train
    )

    adv_model = IntentGuidedAdversarialModel(
        encoder, args.num_hidden, args.intent_dim, args.num_edge_hidden
    ).to(device)
    optimizer_adv = torch.optim.Adam(
        adv_model.parameters(), lr=args.learning_rate_adv,
        weight_decay=args.wd_adv
    )

    # ========== 日志 ==========
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = osp.join(log_dir, "run_ig_results.txt")
    with open(log_file, "a") as f:
        f.write("########################################\n")
        f.write(str(vars(args)) + "\n")

    ones = torch.ones(data.num_edges).to(device)

    # ========== 训练 (Min-Max) ==========
    start = t()
    prev = start
    for epoch in range(1, args.num_epochs + 1):
        # ----- Phase 1: 对抗生成器最大化损失 -----
        adv_model.train()
        adv_model.zero_grad()
        contrastive_model.eval()

        edge_info = adv_model(data.x, data.edge_index, ones, intent_vector)
        adv_w, rec_w, reg = generate_ar_edge_weight(
            edge_info, args.adv_temp, args.bias
        )

        z_adv = contrastive_model(
            data.x, data.edge_index, torch.cat([adv_w, adv_w], dim=0)
        )
        z_rec = contrastive_model(
            data.x, data.edge_index, torch.cat([rec_w, rec_w], dim=0)
        )

        loss, _ = contrastive_model.total_loss(
            z_adv, z_rec, intent_vector, reg,
            reg_lambda=args.reg_lambda, adv_lambda=args.adv_lambda,
            edge_fea_adv=edge_info['upper_edge_fea'],
            edge_fea_rec=edge_info['lower_edge_fea']
        )
        (-loss).backward()
        optimizer_adv.step()

        # ----- Phase 2: 主模型最小化损失 -----
        contrastive_model.train()
        contrastive_model.zero_grad()
        adv_model.eval()

        edge_info = adv_model(data.x, data.edge_index, ones, intent_vector)
        adv_w, rec_w, _ = generate_ar_edge_weight(
            edge_info, args.adv_temp, args.bias
        )

        z_adv = contrastive_model(
            data.x, data.edge_index, torch.cat([adv_w, adv_w], dim=0)
        )
        z_rec = contrastive_model(
            data.x, data.edge_index, torch.cat([rec_w, rec_w], dim=0)
        )

        l_con = contrastive_model.contrastive_loss(z_adv, z_rec)
        l_int = contrastive_model.intent_consistency_loss(
            z_adv, z_rec, intent_vector
        )
        model_loss = l_con + args.lambda_intent * l_int
        model_loss.backward()
        optimizer_train.step()

        now = t()
        print(
            f'(T) | Epoch={epoch:03d}, loss={model_loss:.4f}, '
            f'con={l_con:.4f}, intent={l_int:.4f}, '
            f'this epoch {now - prev:.4f}, total {now - start:.4f}'
        )
        prev = now

    # ========== 评估 (节点分类台架) ==========
    with torch.no_grad():
        emb = contrastive_model(data.x, data.edge_index, ones)

    micro_f1_mean, micro_f1_std, macro_f1_mean, macro_f1_std, acc_mean, acc_std = \
        label_classification(emb, data, args.dataset, ratio=0.1)

    formatted_result = (
        f"micro_f1: {micro_f1_mean:.2f}±{micro_f1_std:.2f}, "
        f"macro_f1: {macro_f1_mean:.2f}±{macro_f1_std:.2f}, "
        f"acc: {acc_mean:.2f}±{acc_std:.2f}"
    )
    print(formatted_result)
    with open(log_file, 'a') as f:
        f.write('epoch: ' + str(epoch) + '\n')
        f.write(formatted_result + '\n')
    print('-----------------')
