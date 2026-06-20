import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance

# ------------------------------------------------------------------------------
# 1. 配置参数（根据你的数据修改！）
# ------------------------------------------------------------------------------
EMBEDDING_PATH = "PreEmb.npy"  # 你的节点嵌入.npy文件路径
LABEL_PATH = "labels.npy"  # 节点类别标签文件路径
PERPLEXITY = 100  # t-SNE关键参数：困惑度（通常取5-50，数据量小时调小）
LEARNING_RATE = 200  # t-SNE学习率（通常取10-1000）
N_ITER = 1000  # t-SNE迭代次数（最少500，建议1000+保证稳定）
FIGURE_SIZE = (12, 10)  # 可视化图的尺寸
TITLE = "Node Embeddings Visualization with Clustered Target Nodes (t-SNE)"  # 图表标题
SAVE_PATH = "tsne_visualization_with_clustered_targets.png"  # 图表保存路径
TARGET_NODE_IDS = [1810, 1812, 1820, 1822, 1824, 1827, 1829, 2062, 2066, 2076, 2077, 2083, 2088, 2089, 2096, 2098, 2202,
                   2209, 2211, 2212, 2213, 2221, 2226, 2237, 2242, 2243, 2250, 2251, 2253, 2374, 2376, 2383, 2390, 2391,
                   2392, 2398, 2399, 2405, 2409, 2549, 2564, 2566, 2567, 2571, 2576, 2582, 2584, 2601, 2606, 2607, 2613,
                   2617, 2753, 2755, 2756, 2758, 2759, 2765, 2773, 2775, 2776, 2778, 2781, 2782, 2877, 2884, 2887, 2890,
                   2891, 2893, 2901, 2904, 2909, 2914, 2917, 2920, 2924, 2937, 2947, 2948, 3075, 3079, 3085, 3088, 3091,
                   3098, 3100, 3103, 3109, 3110, 3111, 3118, 3124, 3128, 3142, 3308, 3314, 3318, 3324, 3335, 3340, 3348,
                   3354, 3356, 3360, 3370, 3379, 3384, 3387, 3627, 3634, 3639, 3651, 3653, 3658, 3672, 3673, 3679, 3682,
                   3688, 3692, 3693, 3701, 3702, 3706, 3846, 3853, 3856, 3865, 3867]

# 新增参数：控制目标节点聚集程度的阈值
DISTANCE_THRESHOLD = 10.0  # 距离阈值，可根据实际数据调整
MIN_CLUSTER_SIZE = 5  # 最小聚类大小，小于此值的聚类将被过滤

# ------------------------------------------------------------------------------
# 2. 加载节点嵌入数据
# ------------------------------------------------------------------------------
# 加载.npy文件（假设数据格式：shape=(n_nodes, embedding_dim)，每行一个节点的嵌入）
embeddings = np.load(EMBEDDING_PATH)
print(f"✅ 成功加载嵌入数据：{embeddings.shape[0]}个节点，每个节点的嵌入维度为{embeddings.shape[1]}")

# 验证目标节点ID是否有效
valid_target_ids = []
for node_id in TARGET_NODE_IDS:
    if 0 <= node_id < embeddings.shape[0]:
        valid_target_ids.append(node_id)
    else:
        print(f"⚠️ 目标节点ID {node_id} 超出有效范围，已跳过")

TARGET_NODE_IDS = valid_target_ids
print(f"✅ 有效目标节点ID：共{len(TARGET_NODE_IDS)}个目标节点")

if not TARGET_NODE_IDS:
    raise ValueError("没有有效的目标节点ID，请检查TARGET_NODE_IDS列表")

# （可选）加载节点类别标签（若有）
labels = None
if LABEL_PATH is not None:
    if LABEL_PATH.endswith(".npy"):
        labels = np.load(LABEL_PATH)
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
    n_components=2,  # 降维到2维
    perplexity=PERPLEXITY,
    learning_rate=LEARNING_RATE,
    n_iter=N_ITER,
    random_state=42,  # 固定随机种子，确保结果可复现
    init="pca",  # 用PCA初始化，加速收敛且更稳定
    verbose=1  # 打印降维过程日志（可选）
)
# 执行降维：输入高维嵌入，输出2维坐标
embeddings_2d = tsne.fit_transform(embeddings)
print(f"✅ t-SNE降维完成：输出2维坐标 shape={embeddings_2d.shape}")

# ------------------------------------------------------------------------------
# 4. 过滤出聚集在小区域内的目标节点
# ------------------------------------------------------------------------------
# 获取目标节点的2D坐标
target_coords = embeddings_2d[TARGET_NODE_IDS]

# 计算所有目标节点之间的距离
dist_matrix = distance.cdist(target_coords, target_coords, 'euclidean')

# 找到聚集的目标节点（基于距离阈值）
clusters = []
visited = set()

for i in range(len(TARGET_NODE_IDS)):
    if i not in visited:
        # 找到与当前节点距离小于阈值的所有节点
        cluster_indices = np.where(dist_matrix[i] < DISTANCE_THRESHOLD)[0]
        cluster = [TARGET_NODE_IDS[j] for j in cluster_indices if j not in visited]

        if len(cluster) >= MIN_CLUSTER_SIZE:  # 只保留足够大的聚类
            clusters.append(cluster)

        # 标记已访问的节点
        for j in cluster_indices:
            visited.add(j)

# 选择最大的聚类作为要显示的目标节点
if clusters:
    largest_cluster = max(clusters, key=len)
    filtered_target_ids = largest_cluster
    print(f"✅ 已过滤目标节点，保留最大聚类：{len(filtered_target_ids)}个节点")
else:
    # 如果没有找到符合条件的聚类，使用所有目标节点
    filtered_target_ids = TARGET_NODE_IDS
    print(f"⚠️ 未找到符合条件的聚类，将显示所有目标节点")

# ------------------------------------------------------------------------------
# 5. 绘制t-SNE可视化图
# ------------------------------------------------------------------------------
plt.figure(figsize=FIGURE_SIZE)
sns.set_style("whitegrid")  # 设置图表风格（白色网格，更清晰）

# 分离目标节点和普通节点
target_mask = np.zeros(embeddings.shape[0], dtype=bool)
target_mask[filtered_target_ids] = True
non_target_mask = ~target_mask

# 情况1：无标签（先绘制普通节点，再绘制目标节点以突出显示）
if labels is None:
    # 绘制普通节点
    sns.scatterplot(
        x=embeddings_2d[non_target_mask, 0],
        y=embeddings_2d[non_target_mask, 1],
        color="steelblue",
        s=50,
        alpha=0.7,
        label="Regular Nodes"
    )

    # 绘制目标节点（使用不同颜色和形状）
    sns.scatterplot(
        x=embeddings_2d[target_mask, 0],
        y=embeddings_2d[target_mask, 1],
        color="crimson",
        s=300,
        alpha=0.9,
        marker='*',  # 星形标记
        label=f"Clustered Target Nodes ({len(filtered_target_ids)})"
    )

# 情况2：有标签（先绘制普通节点，再叠加目标节点）
else:
    # 绘制普通节点（按原始标签着色）
    sns.scatterplot(
        x=embeddings_2d[non_target_mask, 0],
        y=embeddings_2d[non_target_mask, 1],
        hue=labels[non_target_mask],
        palette="tab10",
        s=50,
        alpha=0.7,
        legend="full",
        label="Regular Nodes"
    )

    # 绘制目标节点（使用特殊样式）
    sns.scatterplot(
        x=embeddings_2d[target_mask, 0],
        y=embeddings_2d[target_mask, 1],
        color="black",
        s=300,
        alpha=0.9,
        marker='*',  # 星形标记
        label=f"Clustered Target Nodes ({len(filtered_target_ids)})"
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
