import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------- 1. 读取邻接矩阵（关键步骤）----------------------
# 请确保 adj.npz 文件路径正确！如果文件和代码在同一文件夹，直接用文件名即可
import scipy.sparse as sp

npz_file_path = "../adj.npz"  # 可根据实际路径修改，例如 "C:/data/adj.npz"

# 检查文件是否存在
if not Path(npz_file_path).exists():
    raise FileNotFoundError(f"未找到文件：{npz_file_path}\n请检查文件路径是否正确！")

# 读取 NPZ 文件（NPZ 可能包含多个数组，需先查看关键名）
adj_matrix = sp.load_npz(npz_file_path)




# 打印邻接矩阵基本信息（确认读取正确）
print(f"\n邻接矩阵形状：{adj_matrix.shape}")
print("邻接矩阵内容：")
print(adj_matrix)

# 检查是否为方阵（邻接矩阵必须是方阵，否则不是合法图结构）
if adj_matrix.shape[0] != adj_matrix.shape[1]:
    raise ValueError(f"邻接矩阵不是方阵！形状为 {adj_matrix.shape}，无法构建图。")

# ---------------------- 2. 构建图结构 ----------------------
# 创建无向图（若为有向图，将 Graph() 改为 DiGraph()）
G = nx.Graph()

# 获取节点数量（邻接矩阵的行数/列数）
num_nodes = adj_matrix.shape[0]
# 添加所有节点（节点编号为 0, 1, 2, ..., num_nodes-1）
G.add_nodes_from(range(num_nodes))

# 根据邻接矩阵添加边（遍历矩阵，值为 1 表示存在边）
for i in range(num_nodes):
    for j in range(i, num_nodes):  # 无向图只需遍历上三角，避免重复
        if adj_matrix[i, j] == 1:
            G.add_edge(i, j)

# 打印图的基本信息
print(f"\n图的节点数：{G.number_of_nodes()}")
print(f"图的边数：{G.number_of_edges()}")
print(f"所有边：{list(G.edges())}")

# ---------------------- 3. 绘制并保存图 ----------------------
# 设置绘图样式（可根据需求调整）
plt.figure(figsize=(10, 8), dpi=100)  # 图的大小和分辨率

# 选择节点布局（不同布局适合不同图结构，可尝试替换）
# - spring_layout：弹簧布局（默认，适合一般图）
# - circular_layout：环形布局（适合展示节点顺序）
# - spectral_layout：谱布局（适合密集图）
pos = nx.spring_layout(G, seed=42)  # seed 固定布局，避免每次绘制位置不同

# 绘制节点（设置颜色、大小、边框）
nx.draw_networkx_nodes(
    G, pos,
    node_color="#4CAF50",  # 节点颜色（绿色）
    node_size=800,  # 节点大小
    edgecolors="#000000",  # 节点边框颜色（黑色）
    linewidths=1.5  # 节点边框宽度
)

# 绘制边（设置颜色、宽度）
nx.draw_networkx_edges(
    G, pos,
    edge_color="#666666",  # 边颜色（灰色）
    width=2,  # 边宽度
    alpha=0.8  # 边透明度
)

# 绘制节点标签（显示节点编号）
nx.draw_networkx_labels(
    G, pos,
    font_size=12,  # 标签字体大小
    font_color="#FFFFFF",  # 标签颜色（白色，与节点颜色对比）
    font_weight="bold"  # 标签字体加粗
)

# 调整图像样式（去除坐标轴、添加标题）
plt.axis("off")  # 隐藏坐标轴
plt.title(f"Graph from Adjacency Matrix\nNodes: {num_nodes}, Edges: {G.number_of_edges()}",
          fontsize=16, pad=20)  # 标题

# 保存图片（支持 PNG/JPG/SVG 等格式，此处保存为 PNG）
save_path = "graph_from_adj_matrix.png"  # 保存路径和文件名
plt.tight_layout()  # 自动调整布局，避免元素被截断
plt.savefig(save_path, bbox_inches="tight")  # bbox_inches 确保标题不被截断
plt.close()  # 关闭绘图窗口，释放内存

print(f"\n图已成功保存到：{Path(save_path).absolute()}")