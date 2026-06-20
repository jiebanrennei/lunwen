import numpy as np
import scipy.sparse as sp
# 1. 读取.npz文件中的邻接矩阵


adjacency_matrix = sp.load_npz('../adj.npz')
   # 替换为文件内邻接矩阵的实际数组名（如'adj'、'matrix'等）

# 生成graph.txt（过滤自环：i != j）
with open('graph.txt', 'w') as f:
    num_vertices = adjacency_matrix.shape[0]
    for i in range(num_vertices):
        # 只保留 j != i 的邻接顶点（排除自环）
        neighbors = [str(j) for j in range(num_vertices) if j != i and adjacency_matrix[i, j] != 0]
        if neighbors:
            f.write(f"{i} {' '.join(neighbors)}\n")
print("graph.txt 生成完成！")
# 存档内的数组名称： ['indices', 'indptr', 'format', 'shape', 'data'] 是这五个