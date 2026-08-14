# Robustness in Text-Attributed Graph Learning 引用备忘

## 论文信息

- 论文题目：**Robustness in Text-Attributed Graph Learning: Insights, Trade-offs, and New Defenses**
- 作者：Runlin Lei, Lu Yi, Mingguo He, Pengyu Qiu, Zhewei Wei, Yongchao Liu, Chuntao Hong
- 会议：**ICLR 2026**
- 本地 PDF：`D:/360极速浏览器X下载/4322_Robustness_in_Text_Attrib.pdf`
- 代码出处：论文中给出的项目地址为 `https://github.com/Leirunlin/TGRB`

> 注意：正式写论文或 BibTeX 时，建议再从 OpenReview、DBLP 或论文官方页面核对最终作者顺序、会议年份、BibTeX、DOI/URL。

## 这篇论文和 EDA-GCL 的关系

这篇论文不是图对比学习论文，也不是 EDA-GCL 的直接对比方法。它主要研究 **Text-Attributed Graphs, TAGs** 上 GNN、Robust GNN 和 GraphLLM 在文本扰动、结构扰动、混合扰动下的鲁棒性。

它和 EDA-GCL 的主要关联点在于：

1. **结构扰动会显著影响图模型鲁棒性**  
   论文系统评估了结构攻击对 GNN、RGNN、GraphLLM 的影响，可以用来支撑 EDA-GCL 中“边级扰动 / 结构增强具有重要意义”的动机。

2. **边级结构处理是鲁棒图学习的重要机制**  
   论文重新评估了 GNNGuard 等基于边过滤 / 相似度过滤的鲁棒 GNN，发现这类简单结构防御方法在 TAG 上仍然有效。这可以和 EDA-GCL 的 edge-level self-adversarial augmentation 形成呼应。

3. **文本鲁棒性与结构鲁棒性存在 trade-off**  
   论文指出不同模型在文本扰动和结构扰动下表现出明显权衡：一些模型更抗结构攻击，但更容易受到文本攻击；另一些模型相反。这个结论可以用于未来工作或讨论部分，说明 EDA-GCL 可以进一步扩展到文本属性图，并考虑文本-结构联合增强。

## 建议引用位置

### 1. Introduction / Motivation

用途：说明图结构扰动和边级鲁棒性的重要性。

可以放在介绍图数据增强或结构扰动挑战的段落中，例如：

> Graph neural networks are known to be sensitive to structural perturbations, and recent robustness studies on text-attributed graphs further show that both GNNs and GraphLLMs can suffer substantial performance degradation under adversarial edge or structure modifications. These observations motivate the need for structure-aware augmentation strategies that explicitly account for edge-level vulnerability.

中文含义：

> 图神经网络对结构扰动较敏感，近期关于文本属性图鲁棒性的研究进一步表明，GNN 和 GraphLLM 在对抗性边或结构修改下都会出现明显性能下降。因此，需要显式考虑边级脆弱性的结构感知增强策略。

### 2. Related Work：Robust Graph Learning / Graph Adversarial Robustness

用途：归入鲁棒图学习、图攻击防御、结构扰动评测相关工作。

可以写：

> Robust graph learning has been extensively studied under adversarial structural perturbations. Recent work on text-attributed graph robustness provides a unified evaluation of GNNs, robust GNNs, and GraphLLMs under textual, structural, and hybrid attacks, revealing that models often exhibit a trade-off between text robustness and structure robustness. Different from these defense-oriented studies, our work focuses on improving graph contrastive learning through edge-level self-adversarial augmentation.

中文含义：

> 鲁棒图学习已有大量工作研究对抗性结构扰动。近期 TAG 鲁棒性研究统一评估了 GNN、鲁棒 GNN 和 GraphLLM 在文本、结构和混合攻击下的表现，发现模型通常存在文本鲁棒性和结构鲁棒性的权衡。不同于这些以防御为主的研究，本文关注通过边级自对抗增强提升图对比学习。

### 3. Discussion / Future Work

用途：说明 EDA-GCL 后续可以扩展到 TAG、多模态图或 GraphLLM 场景。

可以写：

> An interesting future direction is to extend edge self-adversarial augmentation to text-attributed graphs. Since recent evidence suggests that text and structure robustness may not be simultaneously improved by existing models, integrating contrastive structural augmentation with textual semantic robustness could be a promising direction.

中文含义：

> 一个有价值的未来方向是将边级自对抗增强扩展到文本属性图。由于近期研究表明现有模型很难同时提升文本鲁棒性和结构鲁棒性，将结构对比增强与文本语义鲁棒性结合可能是一个有前景的方向。

## 推荐引用角度

优先推荐以下三种引用方式：

### 角度 A：结构扰动重要性

适合位置：Introduction。

核心句：

> Recent TAG robustness studies show that structural perturbations can significantly degrade the performance of graph learning models.

为什么适合：EDA-GCL 的核心是 edge-level augmentation，这个角度能自然引出“边结构需要被认真建模”。

### 角度 B：边过滤 / 结构防御与 EDA-GCL 的区别

适合位置：Related Work。

核心句：

> Existing robust GNNs often rely on detecting or filtering suspicious edges, whereas EDA-GCL learns adversarial edge augmentations for contrastive representation learning.

为什么适合：可以把你的方法和 GNNGuard、ProGNN 等结构防御方法区分开。

### 角度 C：TAG 扩展方向

适合位置：Discussion / Future Work。

核心句：

> The observed text-structure robustness trade-off in TAGs suggests that future graph contrastive learning methods may need to jointly consider structural and semantic perturbations.

为什么适合：如果当前实验没有 TAG，这篇论文最好不要放太核心，而是作为未来扩展方向。

## 不建议的引用方式

不要把这篇论文写成：

1. **图对比学习核心文献**  
   它主要不是 GCL 方法。

2. **EDA-GCL 的直接 baseline**  
   它的 SFT-auto 是 LLM-based defense，和 EDA-GCL 的训练目标、任务场景和模型类型都不同。

3. **证明 EDA-GCL 一定能提升 TAG 鲁棒性**  
   这篇论文只能支撑结构扰动重要性和鲁棒性 trade-off，不能直接证明 EDA-GCL 在 TAG 上有效，除非后续做对应实验。

## 可放入论文的中文草稿

### 引言版本

图神经网络对图结构扰动较为敏感，尤其是边的增删可能显著改变节点表示传播路径并影响下游性能。近期关于文本属性图鲁棒性的系统研究进一步表明，GNN、鲁棒 GNN 以及 GraphLLM 在结构攻击、文本攻击和混合攻击下均存在不同程度的脆弱性，并且模型通常难以同时兼顾文本鲁棒性与结构鲁棒性。这些发现说明，设计结构感知的图增强策略，尤其是从边级别刻画图结构脆弱性，对于提升图表示学习的鲁棒性具有重要意义。

### 相关工作版本

与鲁棒图学习相关的工作通常从攻击检测、异常边过滤、鲁棒聚合或图结构恢复等角度提升模型在对抗扰动下的稳定性。近期 TAG 鲁棒性研究统一评估了 GNN、RGNN 和 GraphLLM 在文本、结构及混合扰动下的表现，揭示了文本鲁棒性与结构鲁棒性之间的权衡。不同于这些主要面向防御或攻击恢复的研究，本文从图对比学习角度出发，通过边级自对抗增强构造更具挑战性的结构视图，从而提升图表示的判别性和鲁棒性。

## 临时 BibTeX 草稿

```bibtex
@inproceedings{lei2026robustness,
  title={Robustness in Text-Attributed Graph Learning: Insights, Trade-offs, and New Defenses},
  author={Lei, Runlin and Yi, Lu and He, Mingguo and Qiu, Pengyu and Wei, Zhewei and Liu, Yongchao and Hong, Chuntao},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```

> 这个 BibTeX 是根据 PDF 首页信息整理的临时版本，正式使用前建议核对官方 BibTeX。
