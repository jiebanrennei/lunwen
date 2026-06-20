import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------------
# 1. 配置参数（根据你的数据修改！）
# ------------------------------------------------------------------------------
EMBEDDING_PATH = "PreEmb.npy"  # 你的节点嵌入.npy文件路径
LABEL_PATH = "labels.npy"  # 节点类别标签文件路径（已修改为你的label.npy）
PERPLEXITY = 100    # t-SNE关键参数：困惑度（通常取5-50，数据量小时调小）
LEARNING_RATE = 200  # t-SNE学习率（通常取10-1000）
N_ITER = 1000      # t-SNE迭代次数（最少500，建议1000+保证稳定）
FIGURE_SIZE = (10, 8)  # 可视化图的尺寸
TITLE = "Node Embeddings Visualization (t-SNE)"  # 图表标题
SAVE_PATH = "tsne_visualization.png"  # 图表保存路径（如.png/.pdf）

# ------------------------------------------------------------------------------
# 2. 加载节点嵌入数据
# ------------------------------------------------------------------------------
# 加载.npy文件（假设数据格式：shape=(n_nodes, embedding_dim)，每行一个节点的嵌入）
embeddings = np.load(EMBEDDING_PATH)
print(f"✅ 成功加载嵌入数据：{embeddings.shape[0]}个节点，每个节点的嵌入维度为{embeddings.shape[1]}")

# （可选）加载节点类别标签（若有）
labels = None
if LABEL_PATH is not None:
    if LABEL_PATH.endswith(".npy"):
        labels = np.load(LABEL_PATH)
        print(len(labels))
        print(1111)
    elif LABEL_PATH.endswith(".txt"):
        labels = np.loadtxt(LABEL_PATH, dtype=int)  # 假设标签是整数（如类别ID）

    # 验证标签数量与节点数量一致

    assert len(labels) == embeddings.shape[0], "标签数量与节点数量不匹配！"
    print(f"✅ 成功加载标签数据：共{len(set(labels))}个类别")

# ------------------------------------------------------------------------------
# 3. 用t-SNE将高维嵌入降维到2维（用于可视化）
# ------------------------------------------------------------------------------
print("🔄 开始执行t-SNE降维...")
tsne = TSNE(
    n_components=2,        # 降维到2维
    perplexity=PERPLEXITY,
    learning_rate=LEARNING_RATE,
    n_iter=N_ITER,
    random_state=42,       # 固定随机种子，确保结果可复现
    init="pca",            # 用PCA初始化，加速收敛且更稳定
    verbose=1              # 打印降维过程日志（可选）
)
# 执行降维：输入高维嵌入，输出2维坐标
embeddings_2d = tsne.fit_transform(embeddings)
print(f"✅ t-SNE降维完成：输出2维坐标 shape={embeddings_2d.shape}")

# ------------------------------------------------------------------------------
# 4. 绘制t-SNE可视化图
# ------------------------------------------------------------------------------
plt.figure(figsize=FIGURE_SIZE)
sns.set_style("whitegrid")  # 设置图表风格（白色网格，更清晰）

# 情况1：无标签（仅用单一颜色绘制所有节点）
if labels is None:
    sns.scatterplot(
        x=embeddings_2d[:, 0],  # 2维坐标的x轴
        y=embeddings_2d[:, 1],  # 2维坐标的y轴
        color="steelblue",      # 节点颜色
        s=50,                   # 节点大小
        alpha=0.7               # 节点透明度（避免重叠遮挡）
    )

# 情况2：有标签（按类别着色，自动生成图例）
else:
    sns.scatterplot(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        hue=labels,             # 按标签分组着色
        palette="tab10",        # 颜色 palette（支持多类别，可换"Set2"等）
        s=50,
        alpha=0.7,
        legend="full"           # 显示完整图例
    )
    plt.legend(title="Node Category", bbox_to_anchor=(1.05, 1), loc="upper left")  # 图例位置调整

# 设置图表标题和轴标签
plt.title(TITLE, fontsize=16, fontweight="bold", pad=20)
plt.xlabel("t-SNE Dimension 1", fontsize=12)
plt.ylabel("t-SNE Dimension 2", fontsize=12)

# 调整布局（避免图例被截断）
plt.tight_layout()

# 保存图表（高分辨率）
plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
print(f"✅ 可视化图表已保存到：{SAVE_PATH}")

# 显示图表（本地运行时）
plt.show()
