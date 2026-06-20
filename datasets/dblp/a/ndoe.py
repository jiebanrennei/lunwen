import numpy as np

# # 读取.npy文件
# array = np.load('../node_types.npy')  # 替换为你的.npy文件路径
#
# # 查看数组信息
# print("数组形状：", array.shape)    # 输出数组维度，如 (100, 200)
# print("数组数据类型：", array.dtype)  # 输出数据类型，如 float64
# print("数组前5个元素：", array[11111:])  # 查看部分数据（根据维度调整切片）
#
#

import numpy as np

# 1. 读取node.npy文件（假设存储的是每个顶点的类型，顺序对应顶点ID 0,1,2...）
node_types = np.load('../node_types.npy')  # 替换为你的node.npy文件路径

# 2. 生成vertex.txt，格式：每行是 "顶点ID 类型"
with open('vertex.txt', 'w') as f:
    for vertex_id in range(len(node_types)):
        # 获取当前顶点的类型（确保是整数类型，若不是则转换）
        vertex_type = int(node_types[vertex_id])
        # 写入一行：顶点ID + 空格 + 类型
        f.write(f"{vertex_id} {vertex_type}\n")

print("vertex.txt 生成完成！")
