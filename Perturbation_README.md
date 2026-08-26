# 数据集对抗性扰动系统

## 概述

本系统用于生成对抗性数据集，评估模型的鲁棒性。通过对原始数据集施加不同类型的扰动，创建更具挑战性的测试场景。

## 扰动类型

### 1. 对抗边注入 (Edge Injection)
- **目的**: 添加连接不相似节点的噪声边
- **方法**: 基于特征相似度，选择低相似度节点对添加边
- **参数**: `noise_ratio` (噪声边比例)
- **效果**: 测试模型对噪声连接的抵抗能力

### 2. 边删除 (Edge Dropout)
- **目的**: 删除重要边（高中心性边）
- **方法**: 优先删除节点度乘积高的边
- **参数**: `drop_ratio` (删除比例)
- **效果**: 测试模型对信息丢失的鲁棒性

### 3. 特征扰动 (Feature Perturbation)
- **目的**: 对节点特征添加噪声或遮蔽
- **方法**: 高斯噪声 或 随机遮蔽
- **参数**: `noise_std` (噪声标准差) 或 `mask_ratio` (遮蔽比例)
- **效果**: 测试模型对特征噪声的抵抗能力

### 4. 组合扰动 (Combined)
- **目的**: 同时应用多种扰动
- **方法**: 边注入 + 边删除 + 特征噪声
- **参数**: 综合配置
- **效果**: 模拟真实的复杂对抗场景

## 扰动级别

| 级别 | 边注入 | 边删除 | 特征噪声 | 说明 |
|------|--------|--------|---------|------|
| **Light** | 1% | 1% | σ=0.01 | 轻微扰动 |
| **Medium** | 3% | 3% | σ=0.03 | 隐蔽扰动 |
| **Heavy** | 5% | 5% | σ=0.05 | 较强但仍克制 |


## 使用方法

### 步骤 1: 生成扰动数据集

```bash
python perturb_datasets.py
```

这将为 ACM 和 IMDB_NEW 生成所有扰动变体（4种类型 × 3个级别 × 3个种子 = 36个文件）。

生成的文件保存在 `data/` 目录：
```
data/ACM_pert_edge_injection_light_seed0.pt
data/ACM_pert_edge_injection_medium_seed0.pt
data/ACM_pert_edge_dropout_heavy_seed1234.pt
...
```

### 步骤 2: 使用扰动数据集训练

```bash
# 使用原始数据集（基线）
python train_ig.py --dataset ACM --encoder gcn --num_hidden 256

# 使用扰动数据集
python train_ig.py --dataset ACM --encoder gcn --num_hidden 256 \
  --perturbed_data data/ACM_pert_combined_medium_seed0.pt
```

### 步骤 3: 对比结果

对比原始数据集和不同扰动级别下的性能，评估模型的鲁棒性。

## 文件命名规则

```
{数据集}_pert_{扰动类型}_{级别}_seed{种子}.pt
```

示例：
- `ACM_pert_edge_injection_light_seed0.pt`
- `IMDB_NEW_pert_combined_heavy_seed1234.pt`

## API 使用

### 在代码中使用扰动函数

```python
from data_perturbation import (
    perturb_edge_injection,
    perturb_edge_dropout,
    perturb_feature_noise,
    perturb_combined,
    save_perturbed_data,
    load_perturbed_data
)

# 加载原始数据
from utils import get_dataset
dataset = get_dataset('./datasets/', 'ACM')
data = dataset[0]

# 应用扰动
perturbed_data = perturb_combined(
    data,
    edge_noise_ratio=0.15,
    edge_drop_ratio=0.15,
    feature_noise_std=0.10,
    seed=0
)

# 保存
save_perturbed_data(perturbed_data, 'data/ACM_custom_pert.pt')

# 加载
loaded_data = load_perturbed_data('data/ACM_custom_pert.pt')
```

## 实验设计建议

### 1. 鲁棒性测试
在原始数据集上训练，在扰动数据集上测试：
```bash
# 训练（原始数据）
python train_ig.py --dataset ACM --num_epochs 200

# 测试（扰动数据）- 需要修改代码支持
python train_ig.py --dataset ACM --perturbed_data data/ACM_pert_combined_medium_seed0.pt
```

### 2. 对抗训练
在扰动数据集上训练，在原始数据集上测试：
```bash
python train_ig.py --dataset ACM --perturbed_data data/ACM_pert_combined_heavy_seed0.pt
```

### 3. 消融实验
对比不同扰动类型的影响：
- 仅边注入
- 仅边删除
- 仅特征噪声
- 组合扰动

## 论文撰写建议

### 实验部分
1. **鲁棒性分析**: 展示模型在不同扰动级别下的性能下降曲线
2. **对比基线**: 与其他方法在扰动数据集上的对比
3. **消融研究**: 四维异常模型在对抗场景下的贡献

### 预期结果
- 四维异常模型在扰动数据集上表现更好
- 随着扰动强度增加，性能下降更缓慢
- 证明模型的鲁棒性和泛化能力

## 注意事项

1. **可复现性**: 每个扰动都使用固定种子，确保结果可复现
2. **数据一致性**: 扰动后的数据保持相同的节点数和标签
3. **性能开销**: 扰动是一次性的，不影响训练速度
4. **存储**: 每个扰动文件约 1-5 MB

## 扩展方向

可以添加更多扰动类型：
- 社区结构扰动（添加离群节点到社区）
- 意图扰动（改变查询意图）
- 时序扰动（改变时间戳）
- 图结构扰动（重连边）

## 相关文件

- `data_perturbation.py`: 扰动工具模块
- `perturb_datasets.py`: 批量生成脚本
- `train_ig.py`: 训练脚本（已支持 `--perturbed_data` 参数）
