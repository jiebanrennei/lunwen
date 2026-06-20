import os
import numpy as np
import scipy.sparse as sp
import pickle
import torch as th
from collections import defaultdict

# --------------------------
# 配置路径
# --------------------------
edge_files = {
    'ma': 'ma.txt',
    'md': 'md.txt',
}

feat_files = {
    'm': 'm_feat.npz',  # 稀疏特征
    'a': 'a_feat.npz',
    'd': 'd_feat.npz',
}

output_dir = ''

# --------------------------
# Step 1: 验证节点ID连续性
# --------------------------
def validate_and_count_nodes(edge_files):
    node_ids = defaultdict(lambda: defaultdict(set))

    for rel, file in edge_files.items():
        src_type, dst_type = rel[0], rel[1]
        with open(file, 'r') as f:
            for line in f:
                src, dst = map(int, line.strip().split())
                node_ids[src_type][src].add(dst_type)
                node_ids[dst_type][dst].add(src_type)

    node_counts = {}
    for ntype in ['m', 'a', 'd']:
        if ntype not in node_ids:
            node_ids[ntype] = {}
        ids = sorted(node_ids[ntype].keys())
        if not ids:
            node_counts[ntype] = 0
            continue
        min_id, max_id = min(ids), max(ids)
        if min_id != 0:
            raise ValueError(f"{ntype}类型节点ID未从0开始")
        expected = set(range(max_id + 1))
        missing = expected - set(ids)
        # if missing:
          #   raise ValueError(f"{ntype}类型节点ID不连续，缺失ID: {missing}")
        node_counts[ntype] = max_id + 1

    return node_counts


node_counts = validate_and_count_nodes(edge_files)
print("各类型节点数量:", node_counts)

# --------------------------
# Step 2: 全局ID映射
# --------------------------
global_id_offset = {
    'm': 0,
    'a': node_counts['m'],
    'd': node_counts['m'] + node_counts['a'],
}

total_nodes = sum(node_counts.values())
print("全局节点总数:", total_nodes)

# --------------------------
# Step 3: 邻接矩阵构建
# --------------------------
rel2type = {
    'ma': 0, 'am': 2,
    'md': 1, 'dm': 3,
    'mm': 4, 'aa': 5, 'dd': 6         # 新增3种自环边类型
}

rows, cols, edge_types = [], [], []

for rel in edge_files:
    src_type = rel[0]
    dst_type = rel[1]
    with open(edge_files[rel], 'r') as f:
        for line in f:
            src, dst = map(int, line.strip().split())
            global_src = global_id_offset[src_type] + src
            global_dst = global_id_offset[dst_type] + dst
            rows.append(global_src)
            cols.append(global_dst)
            edge_types.append(rel2type[rel])
            rows.append(global_dst)
            cols.append(global_src)
            edge_types.append(rel2type[f"{dst_type}{src_type}"])

# 修改点2: 为每个节点添加自环边
for ntype in ['m', 'a', 'd']:
    num_nodes = node_counts[ntype]
    if num_nodes == 0:
        continue
    rel = f"{ntype}{ntype}"
    edge_type = rel2type[rel]
    global_ids = [global_id_offset[ntype] + i for i in range(num_nodes)]
    rows.extend(global_ids)
    cols.extend(global_ids)
    edge_types.extend([edge_type] * num_nodes)

adj = sp.coo_matrix((np.ones(len(rows)), (rows, cols)),
                    shape=(total_nodes, total_nodes)).tocsr()


# --------------------------
# Step 4: 健壮的特征加载
# --------------------------
def load_feature(file_path):
    """通用特征加载方法"""
    # 尝试作为稀疏矩阵加载
    try:
        feat = sp.load_npz(file_path).astype("float32")
        return feat.todense()
    except:
        pass

    # 尝试作为密集矩阵加载
    try:
        # 处理np.save保存的单数组
        return np.load(file_path)
    except:
        pass

    # 处理np.savez保存的多数组
    try:
        loader = np.load(file_path)
        if 'arr_0' in loader.files:
            return loader['arr_0']
        if 't_feat' in loader.files:
            return loader['t_feat']
    except Exception as e:
        raise ValueError(f"无法识别特征文件格式: {file_path}, 错误: {str(e)}")

    raise ValueError(f"无法解析特征文件: {file_path}")


features_list = []

# 处理m,a,d类型
for ntype in ['m']:
    feat = load_feature(feat_files[ntype])
    features_list.append(th.FloatTensor(feat))

# # 处理a,d,k类型
a_feat = sp.eye(node_counts['a'], format='csr').astype("float32").todense()
features_list.append(th.FloatTensor(a_feat))
#
d_feat = sp.eye(node_counts['d'], format='csr').astype("float32").todense()
features_list.append(th.FloatTensor(d_feat))
#
# k_feat = sp.eye(node_counts['k'], format='csr').astype("float32").todense()
# features_list.append(th.FloatTensor(k_feat))

# --------------------------
# Step 5: 保存文件
# --------------------------
sp.save_npz(os.path.join(output_dir, 'adj.npz'), adj)           # 加了自环

with open(os.path.join(output_dir, 'features.pickle'), 'wb') as f:
    pickle.dump(features_list, f)

edge2type = {(u, v): t for u, v, t in zip(rows, cols, edge_types)}
with open(os.path.join(output_dir, 'edge2type.pickle'), 'wb') as f:
    pickle.dump(edge2type, f)

type2id = {'m': 0, 'a': 1, 'd': 2}
node_types = []
for ntype in ['m', 'a', 'd']:
    node_types.extend([type2id[ntype]] * node_counts[ntype])
np.save(os.path.join(output_dir, 'node_types.npy'), np.array(node_types))

print("数据处理完成！输出目录:", output_dir)