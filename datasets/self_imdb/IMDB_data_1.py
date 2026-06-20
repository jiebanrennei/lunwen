import numpy as np
import scipy.sparse as sp
import networkx as nx
from collections import Counter
import torch
from collections import defaultdict

def build_heterogeneous_graph(ma_file, md_file):
    """
    构建异构图，包含 m、a、d类型的节点，并根据提供的边文件添加边。
    :param ma_file: m 和 a 类型节点之间的边文件。
    :param md_file: m 和 d 类型节点之间的边文件。
    :return: 返回构建好的异构图。
    """
    G = nx.Graph()

    # 读取ma.txt文件，表示m类型和a类型的边
    with open(ma_file, 'r') as f:
        for line in f:
            node_m, node_a = map(int, line.strip().split())
            G.add_edge(f"m_{node_m}", f"a_{node_a}", edge_type="m-a")

    # 读取md.txt文件，表示m类型和d类型的边
    with open(md_file, 'r') as f:
        for line in f:
            node_m, node_d = map(int, line.strip().split())
            G.add_edge(f"m_{node_m}", f"d_{node_d}", edge_type="m-d")

    return G

# 特征文件 关键词词袋处理
def generate_m_feat_npy(keyword_file, output_file):

    # 从 keyword.txt 文件生成 m_feat.npz 文件，表示节点的特征向量。

    # :param keyword_file: 输入的 keyword.txt 文件路径。
    # :param output_file: 输出的 m_feat.npz 文件路径。
    node_keywords = {}
    all_keywords = set()

    # 读取 keyword.txt 文件
    with open(keyword_file, 'r') as f:
        for line in f:
            parts = line.strip().split("\t")  # 按制表符分割
            node_id = int(parts[0])  # 节点ID
            keywords = parts[1].split("|")  # 获取关键词列表
            node_keywords[node_id] = keywords
            all_keywords.update(keywords)  # 记录所有出现过的关键词

    # 创建一个关键词到索引的映射
    keyword_to_idx = {keyword: idx for idx, keyword in enumerate(sorted(all_keywords))}
    num_keywords = len(keyword_to_idx)
    print("关键词的长度:", num_keywords)

    # 创建每个节点的特征向量
    rows = []
    cols = []
    data = []

    for node_id, keywords in node_keywords.items():
        keyword_counts = Counter(keywords)  # 计算每个关键词的出现次数
        for keyword, count in keyword_counts.items():
            if keyword in keyword_to_idx:  # 如果该关键词在词汇表中
                rows.append(node_id)
                cols.append(keyword_to_idx[keyword])
                data.append(count)

    # 创建稀疏矩阵（CSR格式）
    m_feat_matrix = sp.csr_matrix((data, (rows, cols)), shape=(max(node_keywords.keys()) + 1, num_keywords))
    # print(m_feat_matrix.shape)
    # print(m_feat_matrix)

    # 保存为 m_feat.npz 文件
    sp.save_npz(output_file, m_feat_matrix)

    print(f"Successfully saved m_feat.npz to {output_file}")

    return m_feat_matrix


def create_mam_npz(G, mam_file):
    """
    生成 mam.npz 文件，保存 m类型节点之间的 m-a-m 元路径实例连接。
    改进方法：遍历 a 节点，预转换 ID，使用集合去重以减少内存占用。
    :param G: 异构图
    :param mam_file: 输出文件路径
    """
    # 预处理所有 m 节点的 ID 映射
    m_id_map = {}
    for node in G.nodes():
        if node.startswith("m_"):
            m_id = int(node.split('_')[1])
            m_id_map[node] = m_id

    # 从m_id_map的值中取最大值
    max_m_id = max(m_id_map.values()) if m_id_map else -1

    # 使用 defaultdict 来统计每一对 (src, dst) 的实例数
    mam_edges = defaultdict(int)

    # 使用集合存储边以去重
    # mam_edges = set()

    # 遍历所有 a 节点
    for a_node in G.nodes():
        if not a_node.startswith("a_"):
            continue

        # 收集该 a 节点连接的 m 节点 ID
        m_neighbors = []
        for neighbor in G.neighbors(a_node):
            if neighbor in m_id_map:
                m_neighbors.append(m_id_map[neighbor])

        # 生成所有有序对 (src, dst)
        for src in m_neighbors:
            for dst in m_neighbors:
                mam_edges[(src, dst)] += 1  # 统计每一对节点之间的元路径实例数

    # 转换为稀疏矩阵的行列索引和权值
    if mam_edges:
        src, dst = zip(*mam_edges.keys())  # 获取源节点和目标节点
        weights = list(mam_edges.values())  # 获取对应的权值（元路径实例数）
    else:
        src, dst, weights = [], [], []

    # 创建 COO 格式稀疏矩阵
    mam_matrix = sp.coo_matrix(
        (weights, (np.array(src), np.array(dst))),
        shape=(max_m_id + 1, max_m_id + 1)  # 根据实际情况调整维度
    )

    # 保存文件
    sp.save_npz(mam_file, mam_matrix)
    print(f"Optimized: Successfully saved mam.npz to {mam_file}")

    return mam_matrix

    # # 筛选出权值大于等于2的边及其数量
    # mask = mam_matrix.data >= 2  # 筛选权值大于等于2的边
    #
    # # 获取符合条件的边的行列索引
    # filtered_src = mam_matrix.row[mask]
    # filtered_dst = mam_matrix.col[mask]
    #
    # # 打印符合条件的边和数量
    # print(f"Edges with weight >= 2:")
    # for src, dst, weight in zip(filtered_src, filtered_dst, mam_matrix.data[mask]):
    #     print(f"({src}, {dst}) with weight {weight}")
    #
    # print(f"Number of edges with weight >= 2: {len(filtered_src)}")

def create_mdm_npz(G, mdm_file):
    """
    生成 mdm.npz 文件，保存 m类型节点之间的 m-d-m 元路径实例连接。
    改进方法：遍历 d 节点，预转换 ID，使用集合去重以减少内存占用。
    :param G: 异构图
    :param mdm_file: 输出文件路径
    """
    # 预处理所有 m 节点的 ID 映射
    m_id_map = {}
    for node in G.nodes():
        if node.startswith("m_"):
            m_id = int(node.split('_')[1])
            m_id_map[node] = m_id

    # 从m_id_map的值中取最大值
    max_m_id = max(m_id_map.values()) if m_id_map else -1

    # 使用 defaultdict 来统计每一对 (src, dst) 的实例数
    mdm_edges = defaultdict(int)

    # 使用集合存储边以去重
    # mam_edges = set()

    # 遍历所有 d 节点
    for d_node in G.nodes():
        if not d_node.startswith("d_"):
            continue

        # 收集该 d 节点连接的 m 节点 ID
        m_neighbors = []
        for neighbor in G.neighbors(d_node):
            if neighbor in m_id_map:
                m_neighbors.append(m_id_map[neighbor])

        # 生成所有有序对 (src, dst)
        for src in m_neighbors:
            for dst in m_neighbors:
                mdm_edges[(src, dst)] += 1  # 统计每一对节点之间的元路径实例数

    # 转换为稀疏矩阵的行列索引和权值
    if mdm_edges:
        src, dst = zip(*mdm_edges.keys())  # 获取源节点和目标节点
        weights = list(mdm_edges.values())  # 获取对应的权值（元路径实例数）
    else:
        src, dst, weights = [], [], []

    # 创建 COO 格式稀疏矩阵
    mdm_matrix = sp.coo_matrix(
        (weights, (np.array(src), np.array(dst))),
        shape=(max_m_id + 1, max_m_id + 1)  # 根据实际情况调整维度
    )

    # 保存文件
    sp.save_npz(mdm_file, mdm_matrix)
    print(f"Optimized: Successfully saved mdm.npz to {mdm_file}")

    return mdm_matrix


def generate_a_feat_npy(G, m_feat_matrix, a_feat_file):
    """
    生成a类型节点的特征向量文件。
    a节点的特征向量是与其直接相连的m节点的特征向量的平均。
    """
    generate_ad_feat_npy(G, m_feat_matrix, a_feat_file, 'a')


def generate_d_feat_npy(G, m_feat_matrix, d_feat_file):
    """
    生成d类型节点的特征向量文件。
    d节点的特征向量是与其直接相连的m节点的特征向量的平均。
    """
    generate_ad_feat_npy(G, m_feat_matrix, d_feat_file, 'd')


def generate_ad_feat_npy(G, m_feat_matrix, output_file, node_type):
    """
    通用函数，生成a或d类型节点的特征向量。
    """
    prefix = f"{node_type}_"
    ad_nodes = [n for n in G.nodes() if n.startswith(prefix)]
    if not ad_nodes:
        print(f"No {node_type} nodes found. Creating empty feature matrix.")
        sp.save_npz(output_file, sp.csr_matrix((0, m_feat_matrix.shape[1])))
        return

    ad_ids = [int(n.split('_')[1]) for n in ad_nodes]
    max_ad_id = max(ad_ids) if ad_ids else 0
    num_features = m_feat_matrix.shape[1]

    rows = []
    cols = []
    data = []

    for node in ad_nodes:
        current_id = int(node.split('_')[1])
        m_neighbors = []
        for neighbor in G.neighbors(node):
            if neighbor.startswith('m_'):
                m_id = int(neighbor.split('_')[1])
                m_neighbors.append(m_id)

        if not m_neighbors:
            continue

        sum_vector = defaultdict(float)
        for m_id in m_neighbors:
            m_row = m_feat_matrix.getrow(m_id).tocoo()
            for col, val in zip(m_row.col, m_row.data):
                sum_vector[col] += val

        num_m = len(m_neighbors)
        avg_vector = {col: (sum_val / num_m) for col, sum_val in sum_vector.items()}

        for col, val in avg_vector.items():
            if val != 0:
                rows.append(current_id)
                cols.append(col)
                data.append(val)

    ad_feat = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(max_ad_id + 1, num_features)
    )

    sp.save_npz(output_file, ad_feat)
    print(f"Saved {output_file} for {node_type} nodes.")


def count_statistics(G):
    """
    统计图中m、a、d节点的数量（基于最大ID+1），以及m-a和m-d边的数量。
    返回包含统计结果的字典。
    """
    stats = {
        'm': 0,
        'a': 0,
        'd': 0,
        'm-a_edges': 0,
        'm-d_edges': 0
    }

    # 统计各类型节点数量（max_id + 1）
    def _get_max_id(prefix):
        nodes = [n for n in G.nodes() if n.startswith(f"{prefix}_")]
        if not nodes:
            return 0
        ids = [int(n.split('_')[1]) for n in nodes]
        return max(ids) + 1  # 关键修改：节点数量 = 最大ID + 1

    stats['m'] = _get_max_id("m")
    stats['a'] = _get_max_id("a")
    stats['d'] = _get_max_id("d")

    # 统计边数量（保持不变）
    for u, v, data in G.edges(data=True):
        et = data.get('edge_type', '')
        if et == 'm-a':
            stats['m-a_edges'] += 1
        elif et == 'm-d':
            stats['m-d_edges'] += 1

    return stats


def load_multi_labels(label_file):
    """加载0-1向量标签文件，返回格式：{节点ID: 标签列表}"""
    node_labels = defaultdict(list)
    all_labels = set()

    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            node_id = int(parts[0])
            flags = list(map(int, parts[1:]))

            # 解析标签索引（值为1的索引位置）
            labels = [i for i, val in enumerate(flags) if val == 1]
            node_labels[node_id] = labels
            all_labels.update(labels)

    return node_labels, sorted(all_labels)


def generate_labels_from_metapath(metapath_matrices, label_file, output_file):
    """
    基于元路径邻接矩阵生成最终标签
    :param metapath_matrices: 元路径邻接矩阵列表（如mam_matrix, mdm_matrix）
    :param label_file: 标签文件路径（格式：节点ID + 0-1向量）
    :param output_file: 输出文件路径
    """
    # 加载标签数据
    node_labels, all_labels = load_multi_labels(label_file)
    global_labels = all_labels if all_labels else [0]  # 全局标签池

    # 确定节点ID范围
    max_node_id = max(
        max(node_labels.keys(), default=0),
        *[max(m.shape[0], m.shape[1]) - 1 for m in metapath_matrices]
    )

    # 初始化标签数组（-1表示未处理）
    labels = np.full(max_node_id + 1, -1, dtype=np.int64)

    # 预处理：标记已知标签（随机选择节点的一个标签作为初始值）
    for node_id, lbls in node_labels.items():
        if node_id > max_node_id:
            continue
        labels[node_id] = np.random.choice(lbls) if lbls else -1

    # 迭代处理所有节点
    for node_id in range(max_node_id + 1):
        if labels[node_id] != -1:
            continue  # 已有标签的跳过

        # 获取元路径邻居
        neighbors = []
        for mat in metapath_matrices:
            if node_id < mat.shape[0]:
                neighbors.extend(mat.getrow(node_id).indices.tolist())
        neighbors = list(set(neighbors))  # 去重

        # 统计邻居标签分布
        counter = defaultdict(int)
        for n in neighbors:
            if n <= max_node_id and labels[n] != -1:
                counter[labels[n]] += 1

        # 获取节点自身的标签候选（可能为空）
        own_labels = node_labels.get(node_id, [])

        # 标签决策逻辑
        if own_labels:
            # 筛选自有标签中出现最多的邻居标签
            valid_counts = {k: v for k, v in counter.items() if k in own_labels}
            if valid_counts:
                max_count = max(valid_counts.values())
                candidates = [k for k, v in valid_counts.items() if v == max_count]
                labels[node_id] = np.random.choice(candidates)
            else:
                labels[node_id] = np.random.choice(own_labels)  # 随机选自有标签
        else:
            # 无自有标签，完全依赖邻居
            if counter:
                max_count = max(counter.values())
                candidates = [k for k, v in counter.items() if v == max_count]
                labels[node_id] = np.random.choice(candidates)
            else:
                labels[node_id] = np.random.choice(global_labels)  # 全无则随机

    # 保存结果
    np.save(output_file, labels)
    print(f"标签文件已生成: {output_file}")


def create_final_metapath(mam_file, mdm_file, output_file):
    """
    合并mam和mdm矩阵生成最终矩阵
    :param mam_file: mam矩阵文件路径
    :param mdm_file: mdm矩阵文件路径
    :param output_file: 输出文件路径
    """
    # 加载两个矩阵
    mam = sp.load_npz(mam_file).tocoo()
    mdm = sp.load_npz(mdm_file).tocoo()

    # 合并所有边的坐标
    rows = np.hstack([mam.row, mdm.row])
    cols = np.hstack([mam.col, mdm.col])

    # 去重并过滤自环
    coords = np.column_stack([rows, cols])
    coords = coords[coords[:, 0] != coords[:, 1]]  # 过滤自环
    unique_coords = np.unique(coords, axis=0)  # 去重

    # 创建新矩阵
    data = np.ones(unique_coords.shape[0], dtype=int)
    shape = (max(mam.shape[0], mdm.shape[0]),
             max(mam.shape[1], mdm.shape[1]))

    final_matrix = sp.coo_matrix(
        (data, (unique_coords[:, 0], unique_coords[:, 1])),
        shape=shape
    )

    # 保存结果
    sp.save_npz(output_file, final_matrix)
    print(f"已生成最终矩阵至 {output_file}")

# 输入的文件
ma_file = "ma.txt"
md_file = "md.txt"
keyword_file = "keyword.txt"
label_file = "label.txt"

# 输出的文件
mam_file = "mam.npz"
mdm_file = "mdm.npz"
m_feat_file = "m_feat.npz"
a_feat_file = "a_feat.npz"
d_feat_file = "d_feat.npz"
labels_file = "labels.npy"
final_meta_path_file = "final_meta_path.npz"


# 构建图
G = build_heterogeneous_graph(ma_file, md_file)

# m词袋向量构建特征    CSR稀疏矩阵
# m_feat_matrix = generate_m_feat_npy(keyword_file, m_feat_file)

# 元路径邻接矩阵构造     COO格式
mam_matrix = create_mam_npz(G, mam_file)
mdm_matrix = create_mdm_npz(G, mdm_file)

# 生成元路径同构矩阵
create_final_metapath(
    mam_file,
    mdm_file,
    final_meta_path_file
)

# 使用函数生成a和d的特征文件
# generate_a_feat_npy(G, m_feat_matrix, a_feat_file)
# generate_d_feat_npy(G, m_feat_matrix, d_feat_file)

# 生成最终标签
# generate_labels_from_metapath(
#     metapath_matrices=[mam_matrix, mdm_matrix],
#     label_file=label_file,
#     output_file=labels_file
# )

# 统计并打印结果
stats = count_statistics(G)
print("\nStatistics:")
print(f"Number of m nodes: {stats['m']}")
print(f"Number of a nodes: {stats['a']}")
print(f"Number of d nodes: {stats['d']}")
print(f"Number of m-a edges: {stats['m-a_edges']}")
print(f"Number of m-d edges: {stats['m-d_edges']}")

