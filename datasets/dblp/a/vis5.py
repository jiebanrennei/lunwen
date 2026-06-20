import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse import issparse

# 强制使用稳定后端（避免GUI问题）
import matplotlib

matplotlib.use('Agg')

# --------------------------
# 节点类型范围定义（保持不变）
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


def read_npz_adjacency(file_path, query_nodes, max_related_nodes=30):
    """
    修复：读取NPZ矩阵，确保query_nodes之间的连接被完整保留
    步骤：1. 先收集query_nodes的所有邻居（含其他query_nodes）；2. 合并后去重；3. 截取子矩阵
    """
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
        print(f"✅ 有效查询节点: {valid_query_nodes}")

        # --------------------------
        # 核心修复：收集所有相关节点（含query_nodes之间的连接）
        # --------------------------
        related_nodes = set()
        for node in valid_query_nodes:
            # 找到当前节点的所有邻居（包括其他query_nodes）
            if issparse(adj_matrix):
                neighbors = adj_matrix[node].nonzero()[1]
            else:
                neighbors = np.where(adj_matrix[node] != 0)[0]

            # 添加邻居（排除自环）
            for neighbor in neighbors:
                if neighbor != node:
                    related_nodes.add(neighbor)

        # 合并：查询节点 + 所有相关节点（确保query_nodes之间的连接被包含）
        all_nodes = sorted(list(set(valid_query_nodes + list(related_nodes))))

        # 限制总节点数（避免图过于复杂）
        if len(all_nodes) > len(valid_query_nodes) + max_related_nodes:
            # 优先保留query_nodes，再截取前N个相关节点
            extra_nodes = [n for n in all_nodes if n not in valid_query_nodes]
            extra_nodes = extra_nodes[:max_related_nodes]
            all_nodes = sorted(valid_query_nodes + extra_nodes)
            print(f"⚠️ 总节点过多，保留查询节点 + 前{max_related_nodes}个相关节点")

        print(f"✅ 子图总节点数: {len(all_nodes)}")
        print(f"✅ 其中查询节点: {len(valid_query_nodes)}个，相关节点: {len(all_nodes) - len(valid_query_nodes)}个")

        # 截取子矩阵（包含所有选中节点的连接，包括query_nodes之间的连接）
        if issparse(adj_matrix):
            adj_submatrix = adj_matrix[all_nodes][:, all_nodes]
        else:
            adj_submatrix = adj_matrix[all_nodes][:, all_nodes]

        # 生成节点类型映射
        node_type_map = {node: get_node_type(node) for node in all_nodes}

        # 验证：打印query_nodes之间的连接（确保修复生效）
        print("\n🔍 验证查询节点间的连接:")
        query_idx = {node: idx for idx, node in enumerate(all_nodes) if node in valid_query_nodes}
        for node1 in valid_query_nodes:
            idx1 = all_nodes.index(node1)
            connected_queries = []
            for node2 in valid_query_nodes:
                if node1 == node2:
                    continue
                idx2 = all_nodes.index(node2)
                if adj_submatrix[idx1, idx2] != 0:
                    connected_queries.append(node2)
            if connected_queries:
                print(f"   节点 {node1} ↔ 查询节点 {connected_queries}")

        return adj_submatrix, node_type_map, all_nodes, valid_query_nodes

    except FileNotFoundError:
        print(f"❌ 错误：文件 '{file_path}' 未找到")
        return None, None, None, None
    except Exception as e:
        print(f"❌ 读取失败：{str(e)}")
        return None, None, None, None


def visualize_multi_type_graph(adj_matrix, node_type_map, all_nodes, query_nodes,
                               save_path="extended_graph.png"):
    """可视化包含相关节点的图（保持原逻辑，确保边被正确绘制）"""
    if adj_matrix is None or node_type_map is None:
        print("❌ 无有效数据，无法可视化")
        return

    # 稀疏矩阵转稠密（确保边的判断正确）
    if issparse(adj_matrix):
        print("✅ 转换稀疏矩阵为稠密矩阵...")
        adj_matrix = adj_matrix.toarray()

    node_count = adj_matrix.shape[0]
    print(f"✅ 待可视化节点数: {node_count}")

    # 创建图并添加节点
    G = nx.Graph()
    G.add_nodes_from(range(node_count))  # 用索引作为临时节点ID，后续映射原ID

    # 添加边（完整保留子矩阵中的所有边，包括query_nodes之间的边）
    edge_count = 0
    selfloop_count = 0
    for i in range(node_count):
        for j in range(i + 1, node_count):  # 避免重复添加边
            if adj_matrix[i][j] != 0:
                G.add_edge(i, j)
                edge_count += 1
    print(f"✅ 成功添加边数: {edge_count}（已自动排除自环）")

    # 优化布局（调整参数让节点分布更合理）
    pos = nx.spring_layout(G, seed=42, k=1.2, iterations=200)  # 增大k值避免节点重叠

    # 绘制图形
    plt.figure(figsize=(18, 15))  # 增大画布尺寸，避免标签重叠

    # 1. 绘制非查询节点（按类型区分）
    for type_name, style in NODE_STYLES.items():
        # 筛选当前类型的非查询节点
        type_node_indices = [
            idx for idx, node_id in enumerate(all_nodes)
            if node_type_map[node_id] == type_name and node_id not in query_nodes
        ]
        if not type_node_indices:
            continue

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

    # 2. 突出显示查询节点（黄色+红色边框）
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
        edgecolors='red',  # 红色边框强调
        linewidths=2,
        label=QUERY_NODE_STYLE["label"]
    )

    # 3. 绘制边（灰色，半透明）
    if edge_count > 0:
        nx.draw_networkx_edges(
            G, pos,
            edge_color='darkgray',
            width=1.0,  # 加粗边，更清晰
            alpha=0.8
        )

    # 4. 绘制节点标签（显示原节点ID，避免重叠）
    nx.draw_networkx_labels(
        G, pos,
        labels={idx: str(node_id) for idx, node_id in enumerate(all_nodes)},
        font_size=10,  # 增大字体
        font_color='black',
        font_weight='bold'  # 加粗标签
    )

    # 添加图例和标题
    plt.legend(loc='upper right', fontsize=12, bbox_to_anchor=(1.2, 1.0))
    plt.title("查询节点及其完整连接关系图（含查询节点间连接）", fontsize=16, pad=20)

    # 保存图像（高分辨率，避免裁剪）
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ 图像已保存至: {save_path}")


def main():
    # 配置参数
    file_path = "../adj.npz"  # 请确保邻接矩阵路径正确
    # --------------------------
    # 完整的查询节点列表（根据你的连接结果整理，包含所有有连接的节点）
    # --------------------------
    query_nodes = [
        1116, 1120, 1150, 1151, 1186, 1200, 1214, 1230, 1261,
        1268, 1290, 1349, 1354
    ]
    max_related_nodes = 50  # 适当增大，确保外部相关节点不被过多截断
    save_image_path = "complete_connection_graph.png"  # 新的保存路径

    # 执行流程
    adj_submatrix, node_type_map, all_nodes, valid_query_nodes = read_npz_adjacency(
        file_path, query_nodes, max_related_nodes)

    if adj_submatrix is not None and node_type_map is not None:
        visualize_multi_type_graph(adj_submatrix, node_type_map, all_nodes, valid_query_nodes, save_image_path)


if __name__ == "__main__":
    main()