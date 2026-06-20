import pickle
# 1. 读取pickle文件（存储边的字典：键为(起点, 终点)元组，值为边类型）
with open('../edge2type.pickle', 'rb') as pkl_file:
    edge_dict = pickle.load(pkl_file)  # 加载数据，格式：{(u, v): type, ...}

# 2. 过滤双向边（确保每条边只保留一次）
added_edges = set()  # 用于记录已添加的边（标准化为u < v的形式）
unique_edges = []  # 存储去重后的边：[(u, v, type), ...]

for (u, v), edge_type in edge_dict.items():
    # 跳过自环边（u == v）
    if u == v:
        continue

    # 标准化边的表示：确保u < v，方便去重
    if u < v:
        standard_key = (u, v)
    else:
        standard_key = (v, u)

    # 若该边未添加过，则记录
    if standard_key not in added_edges:
        unique_edges.append((u, v, edge_type))
        added_edges.add(standard_key)

# 3. 生成edge.txt（格式：边ID 类型）
with open("edge1.txt", "w") as f:
    for edge_id, (u, v, edge_type) in enumerate(unique_edges):
        f.write(f"{edge_id} {edge_type}\n")

print(f"edge.txt 生成完成！共保留 {len(unique_edges)} 条边（已去除双向重复和自环边）")
