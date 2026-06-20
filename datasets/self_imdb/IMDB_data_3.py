import random


def get_max_movie_id(file_paths):
    """获取所有边文件中的最大电影节点ID"""
    max_id = 0
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 1:
                    current_id = int(parts[0])
                    if current_id > max_id:
                        max_id = current_id
    return max_id


def generate_query_nodes(n, output_path='query.txt', edge_files=['ma.txt', 'md.txt']):
    # 获取电影节点总数
    max_id = get_max_movie_id(edge_files)
    m = max_id + 1  # 节点ID从0开始

    # 输入验证
    if n > m:
        raise ValueError(f"请求的节点数 {n} 超过总电影节点数 {m}")
    if n < 1:
        raise ValueError("请求的节点数必须大于0")

    # 生成不重复随机节点
    query_ids = random.sample(range(m), n)

    # 写入文件
    with open(output_path, 'w') as f:
        for node_id in query_ids:
            f.write(f"{node_id}\n")

    print(f"成功生成 {n} 个查询节点，已保存至 {output_path}")
    print(f"节点数量验证：{len(query_ids)} 个唯一ID")


# 使用示例
if __name__ == "__main__":
    # 参数设置
    n = 40  # 需要生成的查询节点数量

    # 执行生成
    generate_query_nodes(n)