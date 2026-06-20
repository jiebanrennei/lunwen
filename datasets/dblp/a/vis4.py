import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse import issparse

# 强制使用稳定后端（避免GUI问题）
import matplotlib

matplotlib.use('Agg')

# --------------------------
# 节点类型范围定义
# --------------------------
NODE_TYPE_RANGES = {
    "author": {"start": 0, "end": 4057},  # 作者节点范围
    "paper": {"start": 4058, "end": 18385},  # 论文节点范围
    "conference": {"start": 18386, "end": 18405},  # 会议节点范围
    "term": {"start": 18406, "end": 26128}  # 术语节点范围
}

# 类型样式映射（颜色、形状、大小）
NODE_STYLES = {
    "author": {"color": "lightcoral", "shape": "o", "size": 300, "label": "作者"},
    "paper": {"color": "lightblue", "shape": "s", "size": 200, "label": "论文"},
    "conference": {"color": "lightgreen", "shape": "d", "size": 400, "label": "会议"},
    "term": {"color": "plum", "shape": "^", "size": 250, "label": "术语"},
    "unknown": {"color": "gray", "shape": "x", "size": 200, "label": "未知类型"}
}

# 查询节点的特殊样式（用于突出显示）
QUERY_NODE_STYLE = {"color": "yellow", "shape": "o", "size": 500, "label": "查询节点"}


def get_node_type(node_id):
    """确定节点类型"""
    for type_name, range_info in NODE_TYPE_RANGES.items():
        if range_info["start"] <= node_id <= range_info["end"]:
            return type_name
    return "unknown"


def read_npz_adjacency(file_path, query_nodes, max_related_nodes=20):
    """读取NPZ矩阵，包含指定节点及其相关节点"""
    try:
        # 加载稀疏矩阵
        adj_matrix = sp.load_npz(file_path)
        if adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("邻接矩阵必须是二维方阵")

        total_nodes = adj_matrix.shape[0]
        print(f"✅ 总节点数: {total_nodes}")

        # 验证查询节点是否在有效范围内
        valid_query_nodes = []
        invalid_nodes = []
        for node in query_nodes:
            if 0 <= node < total_nodes:
                valid_query_nodes.append(node)
            else:
                invalid_nodes.append(node)

        if invalid_nodes:
            print(f"⚠️ 警告：以下节点超出范围将被忽略: {invalid_nodes}")

        if not valid_query_nodes:
            print("❌ 没有有效的查询节点")
            return None, None, None, None

        print(f"✅ 查询节点: {valid_query_nodes}")

        # 找到与查询节点直接相连的节点
        related_nodes = set()
        for node in valid_query_nodes:
            # 找到所有与当前节点相连的节点
            if issparse(adj_matrix):
                neighbors = adj_matrix[node].nonzero()[1]
            else:
                neighbors = np.where(adj_matrix[node] != 0)[0]

            # 添加邻居节点，但排除自环
            for neighbor in neighbors:
                if neighbor != node:  # 排除自环
                    related_nodes.add(neighbor)

        # 限制相关节点数量，避免图过于复杂
        related_nodes = list(related_nodes)
        if len(related_nodes) > max_related_nodes:
            print(f"⚠️ 相关节点过多({len(related_nodes)}个)，将只显示前{max_related_nodes}个")
            related_nodes = related_nodes[:max_related_nodes]

        # 合并查询节点和相关节点，去重并排序
        all_nodes = sorted(list(set(valid_query_nodes + related_nodes)))
        print(f"✅ 总节点数(查询节点+相关节点): {len(all_nodes)}")
        print(f"✅ 相关节点数: {len(related_nodes)}")

        # 截取子矩阵（包含所有选中的节点）
        if issparse(adj_matrix):
            adj_submatrix = adj_matrix[all_nodes][:, all_nodes]
        else:
            adj_submatrix = adj_matrix[all_nodes][:, all_nodes]

        # 生成节点类型映射
        node_type_map = {node: get_node_type(node) for node in all_nodes}

        return adj_submatrix, node_type_map, all_nodes, valid_query_nodes

    except FileNotFoundError:
        print(f"❌ 错误：文件 '{file_path}' 未找到")
        return None, None, None, None
    except Exception as e:
        print(f"❌ 读取失败：{str(e)}")
        return None, None, None, None


def visualize_multi_type_graph(adj_matrix, node_type_map, all_nodes, query_nodes,
                               save_path="extended_graph.png"):
    """可视化包含相关节点的图"""
    if adj_matrix is None or node_type_map is None:
        print("❌ 无有效数据，无法可视化")
        return

    # 稀疏矩阵转稠密
    if issparse(adj_matrix):
        print("✅ 转换稀疏矩阵为稠密矩阵...")
        adj_matrix = adj_matrix.toarray()

    node_count = adj_matrix.shape[0]
    print(f"✅ 待可视化节点数: {node_count}")

    # 创建图
    G = nx.Graph()
    G.add_nodes_from(range(node_count))

    # 添加边（排除自环）
    edge_count = 0
    selfloop_count = 0
    for i in range(node_count):
        for j in range(i, node_count):
            if adj_matrix[i][j] != 0:
                if i != j:  # 排除自环
                    G.add_edge(i, j)
                    edge_count += 1
                else:
                    selfloop_count += 1
    print(f"✅ 成功添加边数: {edge_count}（排除自环 {selfloop_count} 条）")

    # 优化布局
    pos = nx.spring_layout(G, seed=42, k=0.8, iterations=100)

    # 绘制图形
    plt.figure(figsize=(15, 12))

    # 先绘制所有非查询节点
    for type_name, style in NODE_STYLES.items():
        # 筛选当前类型的非查询节点
        type_node_indices = [
            idx for idx, node_id in enumerate(all_nodes)
            if node_type_map[node_id] == type_name and node_id not in query_nodes
        ]
        if not type_node_indices:
            continue  # 该类型无节点，跳过

        # 绘制节点
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=type_node_indices,
            node_color=style["color"],
            node_size=style["size"],
            node_shape=style["shape"],
            edgecolors='black',
            linewidths=0.8,
            label=style["label"]
        )

    # 突出显示查询节点
    query_node_indices = [
        idx for idx, node_id in enumerate(all_nodes)
        if node_id in query_nodes
    ]
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=query_node_indices,
        node_color=QUERY_NODE_STYLE["color"],
        node_size=QUERY_NODE_STYLE["size"],
        node_shape=QUERY_NODE_STYLE["shape"],
        edgecolors='red',  # 红色边框突出显示
        linewidths=2,
        label=QUERY_NODE_STYLE["label"]
    )

    # 绘制边
    if edge_count > 0:
        nx.draw_networkx_edges(
            G, pos,
            edge_color='gray',
            width=0.6,
            alpha=0.7
        )

    # 绘制节点标签（显示原节点ID）
    nx.draw_networkx_labels(
        G, pos,
        labels={idx: str(node_id) for idx, node_id in enumerate(all_nodes)},
        font_size=8,
        font_color='black'
    )

    # 添加图例
    plt.legend(loc='best', fontsize=10)

    # 保存图像
    plt.title("查询节点及其相关节点的网络关系图", fontsize=14, pad=20)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 图像已保存至: {save_path}")


def main():
    # 配置参数
    file_path = "../adj.npz"  # 邻接矩阵文件路径
    # 你指定的查询节点列表
    query_nodes = [8, 9, 40, 42, 49, 64, 105, 124, 132, 145, 151, 159, 164, 194, 217, 229, 397,
    412, 432, 495, 594, 595 , 1412, 1428, 1451, 1474, 1477, 1482, 1491,
    1512, 1533, 1578, 1579, 1649, 1750, 1774, 1783, 1792, 1822, 1824, 1831, 1843,
    1900, 1922, 1923, 1926, 1953, 2254,
    2383, 2399, 2410, 2442, 2453, 2454, 2461, 2498]
    max_related_nodes = 30  # 最多显示的相关节点数量
    save_image_path = "extended_graph.png"  # 保存路径

    # 执行流程
    adj_submatrix, node_type_map, all_nodes, valid_query_nodes = read_npz_adjacency(
        file_path, query_nodes, max_related_nodes)

    if adj_submatrix is not None and node_type_map is not None:
        visualize_multi_type_graph(adj_submatrix, node_type_map, all_nodes, valid_query_nodes, save_image_path)


if __name__ == "__main__":
    main()
