import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse import issparse

# 强制使用稳定后端（避免GUI问题）
import matplotlib

matplotlib.use('Agg')

# --------------------------
# 请根据你的实际数据修改以下节点范围
# 这些值需要与你的节点类型划分完全一致
# --------------------------
NODE_TYPE_RANGES = {
    "author": {"start": 0, "end": 4057},  # 作者节点范围
    "paper": {"start": 4058, "end": 18385},  # 论文节点范围（4058=4057+1）
    "conference": {"start": 18386, "end": 18405},  # 会议节点范围（18386=18385+1）
    "term": {"start": 18406, "end": 26128}  # 术语节点范围（18406=18405+1）
}

# 类型样式映射（颜色、形状、大小）
NODE_STYLES = {
    "author": {"color": "lightcoral", "shape": "o", "size": 300, "label": "作者"},
    "paper": {"color": "lightblue", "shape": "s", "size": 200, "label": "论文"},
    "conference": {"color": "lightgreen", "shape": "d", "size": 400, "label": "会议"},
    "term": {"color": "plum", "shape": "^", "size": 250, "label": "术语"},
    "unknown": {"color": "gray", "shape": "x", "size": 200, "label": "未知类型"}
}


def get_node_type(node_id):
    """自动检测节点类型（修复KeyError的核心函数）"""
    for type_name, range_info in NODE_TYPE_RANGES.items():
        if range_info["start"] <= node_id <= range_info["end"]:
            return type_name
    return "unknown"  # 未匹配到任何类型时返回"unknown"


def read_npz_adjacency(file_path, max_nodes_per_type=10):
    """读取NPZ矩阵，按类型筛选节点并生成类型映射"""
    try:
        # 加载稀疏矩阵
        adj_matrix = sp.load_npz(file_path)
        if adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("邻接矩阵必须是二维方阵")

        total_nodes = adj_matrix.shape[0]
        print(f"✅ 总节点数: {total_nodes}")

        # 按类型筛选节点（每种类型选前max_nodes_per_type个）
        selected_nodes = []
        type_selection = {type_name: [] for type_name in NODE_TYPE_RANGES.keys()}

        # 遍历所有节点，按类型筛选
        for node_id in range(total_nodes):
            node_type = get_node_type(node_id)
            if node_type != "unknown" and len(type_selection[node_type]) < max_nodes_per_type:
                type_selection[node_type].append(node_id)

        # 合并所有选中的节点
        for nodes in type_selection.values():
            selected_nodes.extend(nodes)

        # 打印筛选结果（便于调试）
        print("✅ 按类型筛选的节点数：")
        for type_name, nodes in type_selection.items():
            print(f"  - {type_name}: {len(nodes)}个")
        print(f"  - 总计: {len(selected_nodes)}个")

        # 截取子矩阵（仅包含选中的节点）
        if issparse(adj_matrix):
            adj_submatrix = adj_matrix[selected_nodes][:, selected_nodes]
        else:
            adj_submatrix = adj_matrix[selected_nodes][:, selected_nodes]

        # 生成节点类型映射（确保每个节点都有类型）
        node_type_map = {node: get_node_type(node) for node in selected_nodes}

        return adj_submatrix, node_type_map, selected_nodes

    except FileNotFoundError:
        print(f"❌ 错误：文件 '{file_path}' 未找到")
        return None, None, None
    except Exception as e:
        print(f"❌ 读取失败：{str(e)}")
        return None, None, None


def visualize_multi_type_graph(adj_matrix, node_type_map, save_path="multi_type_graph.png"):
    """可视化多类型节点图（带容错处理）"""
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
    # 节点ID使用子矩阵中的索引（0~node_count-1），同时记录原节点ID
    original_node_ids = list(node_type_map.keys())  # 原节点ID列表
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

    # 优化布局（避免环形）
    pos = nx.spring_layout(G, seed=42, k=0.8, iterations=100)

    # 绘制图形
    plt.figure(figsize=(15, 12))

    # 按类型绘制节点（核心：使用NODE_STYLES区分样式）
    for type_name, style in NODE_STYLES.items():
        # 筛选当前类型的节点（子矩阵中的索引）
        type_node_indices = [
            idx for idx, original_id in enumerate(original_node_ids)
            if node_type_map[original_id] == type_name
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
        labels={idx: str(original_id) for idx, original_id in enumerate(original_node_ids)},
        font_size=6,
        font_color='black'
    )

    # 添加图例
    plt.legend(loc='best', fontsize=10)

    # 保存图像
    plt.title("多类型节点图可视化（作者/论文/会议/术语）", fontsize=14, pad=20)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 图像已保存至: {save_path}")


def main():
    # 配置参数
    file_path = "../adj.npz"  # 邻接矩阵文件路径
    max_nodes_per_type = 10  # 每种类型最多选10个节点
    save_image_path = "multi_type_graph.png"  # 保存路径

    # 执行流程
    adj_submatrix, node_type_map, _ = read_npz_adjacency(file_path, max_nodes_per_type)
    if adj_submatrix is not None and node_type_map is not None:
        visualize_multi_type_graph(adj_submatrix, node_type_map, save_image_path)


if __name__ == "__main__":
    main()
