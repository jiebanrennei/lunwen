"""
批量生成扰动数据集

对 ACM 和 IMDB_NEW 数据集应用不同级别和类型的扰动，
生成用于鲁棒性评估的对抗性数据集。

用法:
    python perturb_datasets.py
"""

import os
# 解决 OpenMP 库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import sys
from torch_geometric.datasets import Amazon, Coauthor, WikiCS
from data_perturbation import (
    perturb_edge_injection,
    perturb_edge_dropout,
    perturb_feature_noise,
    perturb_combined,
    save_perturbed_data,
    generate_perturbation_name,
    get_perturbation_levels
)
from utils import get_dataset


def load_original_dataset(dataset_name):
    """加载原始数据集"""
    print(f"\n{'='*60}")
    print(f"Loading original dataset: {dataset_name}")
    print(f"{'='*60}")

    # ACM 和 IMDB_NEW 使用 get_cs_dataset
    if dataset_name in ['ACM', 'IMDB_NEW', 'DBLP', 'IMDB']:
        from utils import get_cs_dataset
        dataset = get_cs_dataset('./datasets/', dataset_name, cs_full_graph=True)
        data = dataset[0]
        # 从 train_mask 推断类别数
        if hasattr(data, 'train_mask') and data.y is not None:
            num_classes = int(data.y.max().item()) + 1
        else:
            num_classes = 0  # 未知
    else:
        # 其他数据集使用 get_dataset
        from utils import get_dataset
        dataset = get_dataset('./datasets/', dataset_name)
        data = dataset[0]
        if hasattr(data, 'train_mask') and data.y is not None:
            num_classes = int(data.y.max().item()) + 1
        else:
            num_classes = 0

    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.edge_index.size(1)}")
    print(f"Number of features: {data.num_features}")
    print(f"Number of classes: {num_classes}")

    return data, num_classes


def generate_single_perturbation(data, dataset_name, pert_type, level_name, level_config, seed):
    """生成单个扰动数据集"""
    print(f"\nGenerating {pert_type} perturbation ({level_name})...")

    if pert_type == 'edge_injection':
        pert_data = perturb_edge_injection(
            data,
            noise_ratio=level_config['edge_noise_ratio'],
            seed=seed,
            method='dissimilar'
        )
    elif pert_type == 'edge_dropout':
        pert_data = perturb_edge_dropout(
            data,
            drop_ratio=level_config['edge_drop_ratio'],
            seed=seed,
            method='important'
        )
    elif pert_type == 'feature_noise':
        pert_data = perturb_feature_noise(
            data,
            noise_std=level_config['feature_noise_std'],
            seed=seed
        )
    elif pert_type == 'combined':
        pert_data = perturb_combined(
            data,
            edge_noise_ratio=level_config['edge_noise_ratio'],
            edge_drop_ratio=level_config['edge_drop_ratio'],
            feature_noise_std=level_config['feature_noise_std'],
            seed=seed
        )
    else:
        raise ValueError(f"Unknown perturbation type: {pert_type}")

    # 生成文件名
    filename = generate_perturbation_name(dataset_name, pert_type, level_name, seed)
    filepath = os.path.join('data', filename)

    # 保存
    save_perturbed_data(pert_data, filepath)

    # 打印统计信息
    print(f"  Original edges: {data.edge_index.size(1)}")
    print(f"  Perturbed edges: {pert_data.edge_index.size(1)}")
    if hasattr(pert_data, 'pert_info'):
        print(f"  Perturbation info: {pert_data.pert_info}")

    return pert_data


def generate_all_perturbations(dataset_name, seeds=[0, 1234, 111]):
    """为指定数据集生成所有扰动"""
    print(f"\n{'#'*60}")
    print(f"# Generating perturbations for {dataset_name}")
    print(f"{'#'*60}")

    # 加载原始数据
    data, num_classes = load_original_dataset(dataset_name)

    # 定义扰动类型
    pert_types = ['edge_injection', 'edge_dropout', 'feature_noise', 'combined']

    # 获取扰动级别
    levels = get_perturbation_levels()

    # 对每个种子生成扰动
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed: {seed}")
        print(f"{'='*60}")

        for pert_type in pert_types:
            for level_name, level_config in levels.items():
                generate_single_perturbation(
                    data, dataset_name, pert_type, level_name, level_config, seed
                )


def main():
    """主函数"""
    print("="*60)
    print("Dataset Perturbation Generator")
    print("="*60)

    # 确保 data 目录存在
    os.makedirs('data', exist_ok=True)

    # 要处理的数据集
    datasets = ['ACM', 'IMDB_NEW']

    # 随机种子
    seeds = [0, 1234, 111]

    # 生成所有扰动
    for dataset_name in datasets:
        generate_all_perturbations(dataset_name, seeds)

    print("\n" + "="*60)
    print("All perturbations generated successfully!")
    print("="*60)

    # 列出所有生成的文件
    print("\nGenerated files:")
    for filename in sorted(os.listdir('data')):
        if filename.endswith('.pt') and 'pert' in filename:
            filepath = os.path.join('data', filename)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  {filename} ({size_mb:.2f} MB)")


if __name__ == '__main__':
    main()
