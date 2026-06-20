import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse import issparse

# --------------------------
# 关键修复1：强制使用稳定的非交互式后端
# Agg后端无需GUI，直接生成图像文件（避免所有GUI兼容性问题）
# --------------------------
import matplotlib

matplotlib.use('Agg')  # 必须放在所有绘图代码之前


def read_npz_adjacency(file_path, max_nodes=40):
    """读取NPZ稀疏矩阵，截取前N个节点的子矩阵"""
    try:
        # 加载scipy格式的稀疏矩阵
        adj_matrix = sp.load_npz(file_path)

        # 验证方阵（邻接矩阵必须是方阵）
        if adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("邻接矩阵必须是二维方阵")

        total_nodes = adj_matrix.shape[0]
        nodes_to_show = min(total_nodes, max_nodes)
        print(f"✅ 总节点数: {total_nodes} → 截取前 {nodes_to_show} 个节点")

        # 截取前N行N列的子矩阵（稀疏矩阵切片优化）
        adj_submatrix = adj_matrix[:nodes_to_show, :nodes_to_show]
        return adj_submatrix

    except FileNotFoundError:
        print(f"❌ 错误：文件 '{file_path}' 未找到，请检查路径是否正确")
        return None
    except Exception as e:
        print(f"❌ 读取矩阵失败：{str(e)}")
        return None


def visualize_graph(adj_matrix, save_path="graph_visualization.png"):
    """
    简化版可视化：
    1. 用nx.draw()替代复杂的边绘制函数
    2. 移除所有可能冲突的样式参数（箭头、边宽、透明度等）
    3. 直接生成图像文件（不弹出窗口，避免GUI问题）
    """
    if adj_matrix is None:
        print("❌ 无有效邻接矩阵，无法可视化")
        return

    # --------------------------
    # 关键修复2：稀疏矩阵转稠密（40x40规模极小，无性能问题）
    # --------------------------
    if issparse(adj_matrix):
        print("✅ 检测到稀疏矩阵，转换为稠密矩阵...")
        adj_matrix = adj_matrix.toarray()

    node_count = adj_matrix.shape[0]
    print(f"✅ 待可视化节点数: {node_count}")

    # 1. 创建无向图（若需有向图，将nx.Graph()改为nx.DiGraph()，但需注意边方向）
    G = nx.Graph()
    G.add_nodes_from(range(node_count))  # 添加节点（编号0~node_count-1）

    # 2. 添加边（只遍历上三角，避免无向图重复添加边）
    edge_count = 0
    for i in range(node_count):
        for j in range(i, node_count):  # 上三角遍历（i<=j），避免重复边
            if adj_matrix[i][j] != 0:  # 非零值表示有边
                G.add_edge(i, j)
                edge_count += 1
    print(f"✅ 成功添加边数: {edge_count}")

    # --------------------------
    # 关键修复3：用nx.draw()一站式绘图，避开复杂函数
    # nx.draw()内部逻辑更简单，兼容性更强
    # --------------------------
    plt.figure(figsize=(10, 8))  # 固定画布大小

    # 力导向布局（seed确保每次布局一致，k控制节点间距）
    pos = nx.spring_layout(G, seed=42, k=0.6)  # k增大→节点间距更大，减少重叠

    # 一站式绘制节点、边、标签（避免拆分函数引发的冲突）
    nx.draw(
        G,
        pos,
        with_labels=True,  # 显示节点标签（编号0~39）
        node_color='lightblue',  # 节点颜色（简单单色，避免冲突）
        node_size=300,  # 节点大小（40个节点适配）
        font_size=6,  # 标签字体（小字体避免重叠）
        font_color='black',  # 标签颜色（对比清晰）
        edge_color='gray',  # 边颜色（灰色不抢焦点）
        linewidths=0.5  # 节点边框（细边框，避免厚重感）
    )

    # --------------------------
    # 关键修复4：直接保存图像文件，不弹出窗口
    # 避免所有GUI交互导致的错误
    # --------------------------
    plt.title(f"Graph Visualization (Top {node_count} Nodes)", fontsize=12)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')  # 保存高清晰度图像
    plt.close()  # 关闭画布，释放内存
    print(f"✅ 可视化完成！图像已保存至: {save_path}")


def main():
    # 配置参数（根据需要调整）
    file_path = "../adj.npz"  # 你的邻接矩阵文件路径
    max_visualize_nodes = 800  # 最多可视化40个节点
    save_image_path = "top40_graph.png"  # 图像保存路径

    # 读取矩阵并可视化
    adj_submatrix = read_npz_adjacency(file_path, max_visualize_nodes)
    if adj_submatrix is not None:
        print(f"✅ 子矩阵信息：类型={type(adj_submatrix)}, 形状={adj_submatrix.shape}")
        visualize_graph(adj_submatrix, save_image_path)


if __name__ == "__main__":
    main()