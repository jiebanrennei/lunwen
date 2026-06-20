import numpy as np

# 2. 读取 labels.npy 文件（需替换为你的文件实际路径）
# 注意：如果 labels.npy 和脚本在同一文件夹，直接写文件名即可；否则需写完整路径（如 "C:/data/labels.npy" 或 "/Users/xxx/data/labels.npy"）
labels = np.load("labels.npy")

# 3. 查看数据（可选，验证是否成功读取）
print("数据类型：", type(labels))  # 通常是 numpy.ndarray（NumPy 数组）
print("数据形状：", labels.shape)  # 查看数组的维度（如 (1000,) 表示1000个元素的1维数组，(500,10) 表示500行10列的2维数组）
print("前10个数据：")
print(labels[:500])  # 打印前10个元素，避免数据过多刷屏