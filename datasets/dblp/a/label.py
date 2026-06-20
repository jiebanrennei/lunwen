import numpy as np

# 读取 .npy 文件
data = np.load('../labels.npy')  # 替换为你的 .npy 文件路径

# 查看读取的数据信息
print("数据类型:", type(data))  # 通常是 numpy.ndarray
print("数组形状:", data)  # 例如 (100, 200) 表示二维数组
print("数据维度:", data.ndim)   # 数组的维度数量
print("数据类型:", data.dtype)  # 数组元素的数据类型，如 int32、float64 等