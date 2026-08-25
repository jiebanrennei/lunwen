"""
数据集对抗性扰动工具

支持多种扰动类型:
1. 对抗边注入: 添加连接不相似节点的噪声边
2. 边删除: 删除重要边（高中心性或社区内部边）
3. 特征扰动: 高斯噪声 + 特征遮蔽
4. 社区结构扰动: 添加离群节点、删除边界节点

用于评估模型的鲁棒性和对抗性。
"""

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, remove_self_loops
import os
import random


def set_seed(seed):
    """设置随机种子以保证可复现性"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def compute_node_similarity(x, method='cosine'):
    """计算节点间的相似度矩阵"""
    if method == 'cosine':
        x_norm = torch.nn.functional.normalize(x, dim=-1)
        sim_matrix = x_norm @ x_norm.t()
    elif method == 'dot':
        sim_matrix = x @ x.t()
    else:
        raise ValueError(f"Unknown similarity method: {method}")
    return sim_matrix


def perturb_edge_injection(data, noise_ratio=0.1, seed=0, method='dissimilar'):
    """
    对抗边注入: 添加连接不相似节点的噪声边

    Args:
        data: PyG Data 对象
        noise_ratio: 噪声边比例 (相对于原始边数)
        seed: 随机种子
        method: 'dissimilar' (选择低相似度节点对) 或 'random' (完全随机)

    Returns:
        扰动后的 Data 对象
    """
    set_seed(seed)

    num_nodes = data.num_nodes
    num_edges = data.edge_index.size(1)
    num_noise_edges = int(num_edges * noise_ratio)

    if num_noise_edges == 0:
        return data.clone()

    # 构建已有边的集合
    edge_set = set()
    edge_index_np = data.edge_index.cpu().numpy()
    for i in range(num_edges):
        edge_set.add((edge_index_np[0, i], edge_index_np[1, i]))

    if method == 'dissimilar':
        # 计算相似度矩阵，选择低相似度节点对
        sim_matrix = compute_node_similarity(data.x)
        # 将对角线设为 1（避免自环）
        sim_matrix.fill_diagonal_(1.0)

        # 展平并排序，选择相似度最低的
        sim_flat = sim_matrix.view(-1)
        _, sorted_indices = torch.sort(sim_flat)

        noise_edges = []
        for idx in sorted_indices:
            if len(noise_edges) >= num_noise_edges:
                break
            src = (idx // num_nodes).item()
            dst = (idx % num_nodes).item()
            if src != dst and (src, dst) not in edge_set and (dst, src) not in edge_set:
                noise_edges.append([src, dst])
                edge_set.add((src, dst))

    elif method == 'random':
        # 完全随机选择节点对
        noise_edges = []
        attempts = 0
        max_attempts = num_noise_edges * 10

        while len(noise_edges) < num_noise_edges and attempts < max_attempts:
            src = random.randint(0, num_nodes - 1)
            dst = random.randint(0, num_nodes - 1)
            attempts += 1

            if src != dst and (src, dst) not in edge_set and (dst, src) not in edge_set:
                noise_edges.append([src, dst])
                edge_set.add((src, dst))

    if len(noise_edges) == 0:
        return data.clone()

    # 合并原始边和噪声边
    noise_edge_index = torch.tensor(noise_edges, dtype=torch.long).t()
    new_edge_index = torch.cat([data.edge_index, noise_edge_index], dim=1)

    # 创建新的 Data 对象
    new_data = data.clone()
    new_data.edge_index = new_edge_index

    # 添加扰动信息
    new_data.pert_info = {
        'type': 'edge_injection',
        'method': method,
        'noise_ratio': noise_ratio,
        'num_noise_edges': len(noise_edges),
        'seed': seed
    }

    return new_data


def perturb_edge_dropout(data, drop_ratio=0.2, seed=0, method='important'):
    """
    边删除: 删除重要边

    Args:
        data: PyG Data 对象
        drop_ratio: 删除比例
        seed: 随机种子
        method: 'important' (优先删除高中心性边) 或 'random' (随机删除)

    Returns:
        扰动后的 Data 对象
    """
    set_seed(seed)

    num_edges = data.edge_index.size(1)
    num_drop_edges = int(num_edges * drop_ratio)

    if num_drop_edges == 0:
        return data.clone()

    edge_index = data.edge_index.cpu()

    if method == 'important':
        # 计算边的"重要性"：基于节点度
        src, dst = edge_index[0], edge_index[1]
        degree = torch.bincount(src, minlength=data.num_nodes)
        # 边的重要性 = 两端节点度的乘积
        edge_importance = degree[src] * degree[dst]

        # 按重要性排序，优先删除重要的边
        _, sorted_indices = torch.sort(edge_importance, descending=True)
        drop_indices = sorted_indices[:num_drop_edges]

    elif method == 'random':
        # 随机选择要删除的边
        all_indices = torch.arange(num_edges)
        drop_indices = all_indices[torch.randperm(num_edges)[:num_drop_edges]]

    # 保留未被删除的边
    keep_mask = torch.ones(num_edges, dtype=torch.bool)
    keep_mask[drop_indices] = False
    new_edge_index = edge_index[:, keep_mask]

    # 创建新的 Data 对象
    new_data = data.clone()
    new_data.edge_index = new_edge_index

    # 添加扰动信息
    new_data.pert_info = {
        'type': 'edge_dropout',
        'method': method,
        'drop_ratio': drop_ratio,
        'num_drop_edges': num_drop_edges,
        'seed': seed
    }

    return new_data


def perturb_feature_noise(data, noise_std=0.1, seed=0):
    """
    特征扰动: 添加高斯噪声

    Args:
        data: PyG Data 对象
        noise_std: 噪声标准差
        seed: 随机种子

    Returns:
        扰动后的 Data 对象
    """
    set_seed(seed)

    # 添加高斯噪声
    noise = torch.randn_like(data.x) * noise_std
    new_x = data.x + noise

    # 创建新的 Data 对象
    new_data = data.clone()
    new_data.x = new_x

    # 添加扰动信息
    new_data.pert_info = {
        'type': 'feature_noise',
        'noise_std': noise_std,
        'seed': seed
    }

    return new_data


def perturb_feature_mask(data, mask_ratio=0.2, seed=0):
    """
    特征遮蔽: 随机遮蔽部分特征维度

    Args:
        data: PyG Data 对象
        mask_ratio: 遮蔽比例
        seed: 随机种子

    Returns:
        扰动后的 Data 对象
    """
    set_seed(seed)

    # 生成遮蔽掩码
    mask = torch.rand_like(data.x) > mask_ratio
    new_x = data.x * mask.float()

    # 创建新的 Data 对象
    new_data = data.clone()
    new_data.x = new_x

    # 添加扰动信息
    new_data.pert_info = {
        'type': 'feature_mask',
        'mask_ratio': mask_ratio,
        'seed': seed
    }

    return new_data


def perturb_combined(data, edge_noise_ratio=0.1, edge_drop_ratio=0.1,
                     feature_noise_std=0.05, seed=0):
    """
    组合扰动: 同时应用多种扰动

    Args:
        data: PyG Data 对象
        edge_noise_ratio: 边注入比例
        edge_drop_ratio: 边删除比例
        feature_noise_std: 特征噪声标准差
        seed: 随机种子

    Returns:
        扰动后的 Data 对象
    """
    # 依次应用各种扰动
    pert_data = perturb_edge_injection(data, edge_noise_ratio, seed, method='dissimilar')
    pert_data = perturb_edge_dropout(pert_data, edge_drop_ratio, seed, method='important')
    pert_data = perturb_feature_noise(pert_data, feature_noise_std, seed)

    # 更新扰动信息
    pert_data.pert_info = {
        'type': 'combined',
        'edge_noise_ratio': edge_noise_ratio,
        'edge_drop_ratio': edge_drop_ratio,
        'feature_noise_std': feature_noise_std,
        'seed': seed
    }

    return pert_data


def save_perturbed_data(data, filepath):
    """保存扰动后的数据"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(data, filepath)
    print(f"Saved perturbed data to {filepath}")


def load_perturbed_data(filepath):
    """加载扰动后的数据"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Perturbed data not found: {filepath}")
    data = torch.load(filepath)
    print(f"Loaded perturbed data from {filepath}")
    if hasattr(data, 'pert_info'):
        print(f"Perturbation info: {data.pert_info}")
    return data


def generate_perturbation_name(dataset_name, pert_type, level, seed):
    """生成扰动数据集的文件名"""
    return f"{dataset_name}_pert_{pert_type}_{level}_seed{seed}.pt"


def get_perturbation_levels():
    """定义扰动级别"""
    return {
        'light': {
            'edge_noise_ratio': 0.05,
            'edge_drop_ratio': 0.05,
            'feature_noise_std': 0.05
        },
        'medium': {
            'edge_noise_ratio': 0.15,
            'edge_drop_ratio': 0.15,
            'feature_noise_std': 0.10
        },
        'heavy': {
            'edge_noise_ratio': 0.25,
            'edge_drop_ratio': 0.25,
            'feature_noise_std': 0.20
        }
    }
