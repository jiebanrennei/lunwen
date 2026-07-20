"""验证 imdb_new 数据集能否被训练管线正常加载。"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from utils import get_cs_dataset

# 单关系模式
data = get_cs_dataset('./datasets', 'IMDB_NEW')[0]
print(f'x: {data.x.shape}')
print(f'y: {data.y.shape}, classes: {data.y.unique().tolist()}')
print(f'edge_index: {data.edge_index.shape}')

# 多关系模式
data = get_cs_dataset('./datasets', 'IMDB_NEW', multi_relation=True)[0]
print(f'num_relations: {data.num_relations}')
for i, (ei, name) in enumerate(zip(data.edge_index_list, data.relation_names)):
    print(f'  relation {i}: {name}, edges={ei.shape[1]}')

print('\n=== IMDB_NEW 加载正常 ===')
