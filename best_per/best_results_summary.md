# 最佳运行配置与结果记录

生成时间: 2026-08-25

---

## ACM 数据集

### 最优配置
```bash
nohup python train_ig.py --dataset ACM --encoder gcn --num_hidden 256 --num_epochs 200 > log/acm_gcn_memopt.log 2>&1 &
```

### 关键参数
- `encoder=gcn`
- `num_hidden=256`
- `num_epochs=200`
- `top_k_suspicious=50` (默认)
- `suspicious_boost=1.5` (默认)
- `greedy_anomaly_alpha=0.3` (默认)

### 节点分类结果
```
micro_f1: 90.53±0.51
macro_f1: 90.56±0.55
accuracy: 90.53±0.51
```

### 社区搜索结果 (Greedy)
```
[CS-greedy] w=0.5  P=68.47 R=73.50 F1=69.42 Jaccard=56.98 size=1432.4
```

### 性能对比
| 指标 | Baseline | New | 提升 |
|------|----------|-----|------|
| F1 | 0.6228 | **0.6942** | **+7.14%** |
| IoU | 0.4793 | **0.5698** | **+9.05%** |

### 运行时间
```
训练时间=187.24s, 测试时间=495.64s, 总运行时间=683.86s
每轮均值=0.936s
```

---

## IMDB_NEW 数据集

### 最优配置
```bash
nohup python train_ig.py --dataset IMDB_NEW --encoder gcn --num_hidden 256 --num_epochs 200 > log/imdb_gcn_memopt.log 2>&1 &
```

### 关键参数
- `encoder=gcn`
- `num_hidden=256`
- `num_epochs=200`
- `top_k_suspicious=50` (默认)
- `suspicious_boost=1.5` (默认)
- `greedy_anomaly_alpha=0.3` (默认，但实际无影响)

### 节点分类结果
```
micro_f1: 67.51±0.48
macro_f1: 67.45±0.49
accuracy: 67.51±0.48
```

### 社区搜索结果 (Greedy)
```
[CS-greedy] w=0.0  P=39.32 R=67.36 F1=49.51 Jaccard=33.49 size=2583.4
```

### 性能对比
| 指标 | Baseline | New | 提升 |
|------|----------|-----|------|
| F1 | 0.4023 | **0.4951** | **+9.28%** |
| IoU | 0.2545 | **0.3349** | **+8.04%** |

### 运行时间
```
训练时间=440.60s, 测试时间=451.24s, 总运行时间=892.32s
每轮均值=2.203s
```

### 特殊说明
- IMDB 上 `w=0.0` 最优，说明社区较大，不应有大小惩罚
- `greedy_anomaly_alpha` 对 IMDB 无影响（设为 0.0 结果相同）

---

## DBLP 数据集

### 运行配置
```bash
nohup python train_ig.py --dataset DBLP --encoder gcn --num_hidden 256 --num_epochs 200 > log/dblp_gcn_memopt.log 2>&1 &
```

### 关键参数
- `encoder=gcn`
- `num_hidden=256`
- `num_epochs=200`

### 节点分类结果
```
micro_f1: 80.18±0.64
macro_f1: 79.45±0.74
accuracy: 80.18±0.64
```

### 社区搜索结果 (Greedy)
```
[CS-greedy] w=0.3  P=44.57 R=60.42 F1=50.65 Jaccard=34.53 size=1377.9
```

### 性能对比
| 指标 | Baseline | New | 变化 |
|------|----------|-----|------|
| F1 | 0.7412 | 0.5065 | **-23.47%** 🔴 |
| IoU | 0.6150 | 0.3453 | **-26.97%** 🔴 |

### 问题分析
- DBLP 性能大幅下降，可能原因:
  1. 四维异常模型不适合 DBLP 的图结构
  2. DBLP 图比较规则，可疑节点识别引入噪声
  3. detach 优化影响了编码器训练
- **建议**: 对 DBLP 使用 `--no_suspicious_kl` 禁用可疑节点训练

---

## 调优实验记录

### IMDB 调优实验
| 实验 | 参数 | Best w | F1 | Jaccard | 结论 |
|------|------|--------|---|---|---|
| Original | boost=1.5 | 0.0 | 49.51 | 33.49 | **最优** |
| No boost | --no_suspicious_boost | 0.0 | 49.40 | 33.33 | -0.11 |
| Boost=1.2 | --suspicious_boost 1.2 | 0.0 | 49.47 | 33.42 | -0.04 |
| Alpha=0.0 | --greedy_anomaly_alpha 0.0 | 0.0 | 49.51 | 33.49 | 相同 |

### ACM 调优实验
| 实验 | 参数 | Best w | F1 | Jaccard | 结论 |
|------|------|--------|---|---|---|
| Original | topk=50, boost=1.5 | 0.5 | 69.42 | 56.98 | **最优** |
| TopK=100 | --top_k_suspicious 100 | 0.3 | 69.34 | 56.27 | -0.08 |
| Boost=2.0 | --suspicious_boost 2.0 | 0.3 | 67.97 | 54.75 | -1.45 |
| Alpha=0.5 | --greedy_anomaly_alpha 0.5 | 0.3 | 69.39 | 56.59 | -0.03 |

---

## 显存优化要点

### 关键优化 (解决 16GB OOM 问题)
1. **切断编码器到可疑识别器的梯度图** (`train_ig.py` line 1671)
   - `z_rec.detach()` 和 `intent_vector.detach()` 传入 `suspicious_identifier`
   - 避免 autograd 图持有所有 chunk 中间量 (~16GB)

2. **减小 chunk_size**: 4096 → 1024 (`edge_importance.py`)
   - 降低 forward 阶段峰值显存

3. **节点级 MLP 分块**: sem_anomaly 和 intent_anomaly
   - chunk_size=1024，避免一次性创建大张量

4. **显式张量清理**
   - 训练循环中添加 `del` 和 `torch.cuda.empty_cache()`

### 显存使用
- ACM (4.3M edges): ~10GB
- IMDB_NEW: ~8GB
- DBLP: ~6GB

---

## 总结

### 成功案例
- ✅ ACM: F1 +7.14%, IoU +9.05%
- ✅ IMDB: F1 +9.28%, IoU +8.04%
- ✅ 显存优化成功，10-12GB GPU 可运行

### 待改进
- 🔴 DBLP: 性能下降 23%，需要针对该数据集调整或禁用可疑节点训练

### 最优参数总结
```
ACM:    topk=50, boost=1.5, alpha=0.3, w=0.5
IMDB:   topk=50, boost=1.5, alpha=0.3, w=0.0
DBLP:   需要禁用可疑节点训练 (--no_suspicious_kl)
```

---

## 数据集对抗性扰动系统

### 概述
为评估模型鲁棒性，实现了数据集扰动系统。通过预处理生成对抗性数据集，确保结果可复现。

### 扰动类型
1. **对抗边注入**: 添加连接不相似节点的噪声边
2. **边删除**: 删除高中心性的重要边
3. **特征扰动**: 高斯噪声 + 特征遮蔽
4. **组合扰动**: 同时应用多种扰动

### 扰动级别
| 级别 | 边注入 | 边删除 | 特征噪声 | 说明 |
|------|--------|--------|---------|------|
| Light | 5% | 5% | σ=0.05 | 轻度扰动 |
| Medium | 15% | 15% | σ=0.10 | 中度扰动 |
| Heavy | 25% | 25% | σ=0.20 | 重度扰动 |

### 生成扰动数据集
```bash
# 为 ACM 和 IMDB 生成所有扰动变体
python perturb_datasets.py
```

生成文件保存在 `data/` 目录，命名格式：
```
data/ACM_pert_combined_medium_seed0.pt
data/IMDB_NEW_pert_edge_injection_heavy_seed1234.pt
```

### 使用扰动数据集训练
```bash
# 原始数据集（基线）
python train_ig.py --dataset ACM --encoder gcn --num_hidden 256

# 扰动数据集
python train_ig.py --dataset ACM --encoder gcn --num_hidden 256 \
  --perturbed_data data/ACM_pert_combined_medium_seed0.pt
```

### 相关文件
- `data_perturbation.py`: 扰动工具模块
- `perturb_datasets.py`: 批量生成脚本
- `Perturbation_README.md`: 详细文档

### 实验设计建议
1. **鲁棒性测试**: 在原始数据训练，在扰动数据测试
2. **对抗训练**: 在扰动数据训练，在原始数据测试
3. **消融实验**: 对比不同扰动类型的影响

### 预期贡献
- 证明模型在对抗场景下的鲁棒性
- 提供可复现的对抗性评测基准
- 作为论文的独立贡献点
