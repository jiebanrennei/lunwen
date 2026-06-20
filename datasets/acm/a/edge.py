import pickle

# 1. 读取pickle文件（假设存储的是每条边的类型，顺序对应边ID 0,1,2...）
with open('../edge2type.pickle', 'rb') as pkl_file:  # 替换为你的pickle文件路径
    edge_dict = pickle.load(pkl_file)  # 加载pickle数据


    # 生成 edge.txt：每行格式为 "边ID 类型"
    with open("edge.txt", "w") as f:
        for edge_id, (edge_key, edge_type) in enumerate(edge_dict.items()):
            f.write(f"{edge_id} {edge_type}\n")

    print("edge.txt 生成完成！")
#     print(edge_types)
#
# # 2. 生成edge.txt，格式：每行是 "边ID 类型"
# with open('edge.txt', 'w') as f:
#     for edge_id in range(len(edge_types)):
#         # 获取当前边的类型（确保是整数类型）
#         edge_type = int(edge_types[edge_id])
#         # 写入一行：边ID + 空格 + 类型
#         f.write(f"{edge_id} {edge_type}\n")
#
# print("edge.txt 生成完成！")




# 解析边数据为列表，每个元素是 (起点, 终点, 类型)
