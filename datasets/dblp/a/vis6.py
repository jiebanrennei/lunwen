import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse import issparse

# 强制使用稳定后端（避免GUI问题）
import matplotlib
matplotlib.use('Agg')

# --------------------------
# 1. 核心调整：增大所有节点的图案尺寸（原尺寸×1.8倍）
# --------------------------
NODE_TYPE_RANGES = {
    "author": {"start": 0, "end": 4057},  # 作者节点范围
    "paper": {"start": 4058, "end": 18385},  # 论文节点范围
    "conference": {"start": 18386, "end": 18405},  # 会议节点范围
    "term": {"start": 18406, "end": 26128}  # 术语节点范围
}

# 类型样式映射（重点：增大node_size，原300→550，200→360等）
NODE_STYLES = {
    "author": {"color": "lightcoral", "shape": "o", "size": 1500, "label": "作者"},  # 原300→550
    "paper": {"color": "lightblue", "shape": "s", "size": 1, "label": "论文"},    # 原200→360
    "conference": {"color": "lightgreen", "shape": "d", "size": 720, "label": "会议"},# 原400→720
    "term": {"color": "plum", "shape": "^", "size": 450, "label": "术语"},         # 原250→450
    "unknown": {"color": "gray", "shape": "x", "size": 360, "label": "未知类型"}   # 原200→360
}

# 查询节点样式（重点：增大size，原500→900，突出效果更强）
QUERY_NODE_STYLE = {"color": "yellow", "shape": "o", "size": 900, "label": "查询节点"}


def get_node_type(node_id):
    """确定节点类型（无修改）"""
    for type_name, range_info in NODE_TYPE_RANGES.items():
        if range_info["start"] <= node_id <= range_info["end"]:
            return type_name
    return "unknown"


def read_npz_adjacency(file_path, query_nodes, max_related_nodes=30):
    """读取NPZ矩阵（无修改，确保数据正常加载）"""
    try:
        adj_matrix = sp.load_npz(file_path)
        if adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("邻接矩阵必须是二维方阵")

        total_nodes = adj_matrix.shape[0]
        print(f"✅ 总节点数: {total_nodes}")

        # 验证查询节点有效性
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

        # 收集相关节点（含查询节点间连接）
        related_nodes = set()
        for node in valid_query_nodes:
            if issparse(adj_matrix):
                neighbors = adj_matrix[node].nonzero()[1]
            else:
                neighbors = np.where(adj_matrix[node] != 0)[0]
            for neighbor in neighbors:
                if neighbor != node:
                    related_nodes.add(neighbor)

        # 合并节点并限制数量
        all_nodes = sorted(list(set(valid_query_nodes + list(related_nodes))))
        if len(all_nodes) > len(valid_query_nodes) + max_related_nodes:
            extra_nodes = [n for n in all_nodes if n not in valid_query_nodes][:max_related_nodes]
            all_nodes = sorted(valid_query_nodes + extra_nodes)
            print(f"⚠️ 总节点过多，保留查询节点 + 前{max_related_nodes}个相关节点")

        print(f"✅ 子图总节点数: {len(all_nodes)}")
        print(f"✅ 其中查询节点: {len(valid_query_nodes)}个，相关节点: {len(all_nodes) - len(valid_query_nodes)}个")

        # 截取子矩阵并生成类型映射
        if issparse(adj_matrix):
            adj_submatrix = adj_matrix[all_nodes][:, all_nodes]
        else:
            adj_submatrix = adj_matrix[all_nodes][:, all_nodes]
        node_type_map = {node: get_node_type(node) for node in all_nodes}

        # 验证查询节点间连接
        print("\n🔍 验证查询节点间的连接:")
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
                               save_path="complete_connection_graph.png"):
    """可视化函数（重点：添加去掉边框的代码，适配增大的节点）"""
    if adj_matrix is None or node_type_map is None:
        print("❌ 无有效数据，无法可视化")
        return

    # 稀疏矩阵转稠密
    if issparse(adj_matrix):
        print("✅ 转换稀疏矩阵为稠密矩阵...")
        adj_matrix = adj_matrix.toarray()

    node_count = adj_matrix.shape[0]
    print(f"✅ 待可视化节点数: {node_count}")

    # 创建图并添加节点/边
    G = nx.Graph()
    G.add_nodes_from(range(node_count))
    edge_count = 0
    for i in range(node_count):
        for j in range(i + 1, node_count):
            if adj_matrix[i][j] != 0:
                G.add_edge(i, j)
                edge_count += 1
    print(f"✅ 成功添加边数: {edge_count}（已自动排除自环）")

    # --------------------------
    # 2. 适配节点大小：调整布局参数k（避免增大后的节点重叠）
    # --------------------------
    # k值从1.2增大到1.8，让节点分布更稀疏（k越大，节点间距越大）
    pos = nx.spring_layout(G, seed=42, k=1.8, iterations=200)

    # 创建画布（保持18×15尺寸，足够容纳大节点）
    plt.figure(figsize=(18, 15))

    # --------------------------
    # 3. 关键：去掉图像边框（隐藏top/right/bottom/left四条边框）
    # --------------------------

    # 强制使用稳定后端（避免GUI问题）
    import matplotlib
    matplotlib.use('Agg')

    # 新增：指定中文字体（以SimHei为例，需确保系统已安装该字体）
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常问题


    ax = plt.gca()  # 获取当前坐标轴
    ax.spines['top'].set_visible(False)    # 隐藏上边框
    ax.spines['right'].set_visible(False)  # 隐藏右边框
    ax.spines['bottom'].set_visible(False) # 隐藏下边框
    ax.spines['left'].set_visible(False)   # 隐藏左边框
    ax.set_xticks([])  # 隐藏x轴刻度
    ax.set_yticks([])  # 隐藏y轴刻度

    # 绘制非查询节点（使用增大后的size参数）
    for type_name, style in NODE_STYLES.items():
        type_node_indices = [
            idx for idx, node_id in enumerate(all_nodes)
            if node_type_map[node_id] == type_name and node_id not in query_nodes
        ]
        if not type_node_indices:
            continue
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=type_node_indices,
            node_color=style["color"],
            node_size=style["size"],       # 应用增大的节点尺寸
            node_shape=style["shape"],
            edgecolors='black',            # 保留节点黑色边框（增强轮廓）
            linewidths=1.2,                # 边框加粗（适配大节点，更清晰）
            label=style["label"]
        )

    # 绘制查询节点（同样应用增大的size）
    query_node_indices = [
        idx for idx, node_id in enumerate(all_nodes)
        if node_id in query_nodes
    ]
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=query_node_indices,
        node_color=QUERY_NODE_STYLE["color"],
        node_size=QUERY_NODE_STYLE["size"],  # 查询节点尺寸增大到900
        node_shape=QUERY_NODE_STYLE["shape"],
        edgecolors='red',                    # 红色边框保留（突出查询节点）
        linewidths=1,                      # 边框加粗（适配大节点）
        label=QUERY_NODE_STYLE["label"]
    )

    # 绘制边（适配大节点：边宽从1.0增大到1.2，更清晰）
    if edge_count > 0:
        nx.draw_networkx_edges(
            G, pos,
            edge_color='darkgray',
            width=1.2,    # 边加粗（避免被大节点遮挡）
            alpha=0.8
        )

    # 绘制节点标签（适配大节点：字体从10增大到12，避免标签过小）
    nx.draw_networkx_labels(
        G, pos,
        labels={idx: str(node_id) for idx, node_id in enumerate(all_nodes)},
        font_size=12,     # 标签字体增大
        font_color='black',
        font_weight='bold'

    )

    # 添加图例和标题（适配整体尺寸）
    plt.legend(loc='upper right', fontsize=14, bbox_to_anchor=(1.2, 1.0), frameon=True)
    plt.title("查询节点及其完整连接关系图（大节点+无边框）", fontsize=18, pad=25)

    # 保存图像（保持高分辨率，bbox_inches='tight'避免图例被裁剪）
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')  # 显式设置白色背景（避免透明）
    plt.close()
    print(f"✅ 图像已保存至: {save_path}")


def main():
    # 配置参数（无修改，保持原查询节点和路径）
    file_path = "../adj.npz"  # 确保邻接矩阵路径正确
    query_nodes = [
        1116, 1120, 1150, 1151, 1186, 1200, 1214, 1230, 1261,
        1268, 1290, 1349, 1354
    ]
    max_related_nodes = 50  # 保持原限制（避免节点过多导致重叠）
    save_image_path = "large_node_no_border_graph.png"  # 新文件名（区分原结果）

    # 执行流程
    adj_submatrix, node_type_map, all_nodes, valid_query_nodes = read_npz_adjacency(
        file_path, query_nodes, max_related_nodes)

    if adj_submatrix is not None and node_type_map is not None:
        visualize_multi_type_graph(adj_submatrix, node_type_map, all_nodes, valid_query_nodes, save_image_path)


if __name__ == "__main__":
    main()