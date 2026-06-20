import numpy as np
import scipy.sparse as sp
import dgl
import torch
import os.path as osp

path = "."
# 加载数据
adj = sp.load_npz(osp.join(path, 'adj.npz'))
g = dgl.DGLGraph(adj + (adj.T))  # 保证无向图
g = dgl.remove_self_loop(g)  # 去除自环
g = g.to('cpu')  # 将图结构移至指定设备

feat_p = sp.load_npz(osp.join(path, "p_feat.npz")).astype("float32")
feat_a = sp.load_npz(osp.join(path, "a_feat.npz")).astype("float32")

# 节点id到节点type的映射
node_types = np.load(osp.join(path, 'node_types.npy'))
id2type = {i: val for i, val in enumerate(node_types)}

# 转换为torch张量
feat_p = torch.tensor(feat_p.toarray(), dtype=torch.float32)
feat_a = torch.tensor(feat_a.toarray(), dtype=torch.float32)

# 获取类型为0和类型为1的节点的id
type_0_nodes = [node for node, node_type in id2type.items() if node_type == 0]
type_1_nodes = [node for node, node_type in id2type.items() if node_type == 1]

# 计算类型为1的节点的特征是否由类型为0的节点的特征的平均值构成
is_avg = True

for node in type_1_nodes:
    print(f"检查类型为1的节点 {node} 的特征")

    # 获取与类型为1的节点直接相连的类型为0的节点
    neighbors = g.successors(node).numpy()  # 获取邻居节点
    neighbors_type_0 = [n for n in neighbors if id2type[n] == 0]

    if len(neighbors_type_0) > 0:
        print(f"与节点 {node} 直接相连的类型为0的邻居节点：{neighbors_type_0}")

        # 获取这些邻居的特征
        neighbor_feats = feat_p[neighbors_type_0]
        print(f"邻居节点的特征：{neighbor_feats}")

        # 打印节点特征和邻居特征的维度
        print(f"类型为0的邻居特征的维度：{neighbor_feats.shape}")
        print(f"类型为1的节点 {node} 的特征的维度：{feat_a[node].shape}")

        # 计算邻居的特征的平均值
        avg_feat = neighbor_feats.mean(dim=0)
        print(f"邻居特征的平均值：{avg_feat}")

        # 打印平均特征的维度
        print(f"邻居特征平均值的维度：{avg_feat.shape}")

        # 检查类型为1的节点的特征是否等于平均特征
        if not torch.allclose(feat_a[node], avg_feat, atol=1e-6):
            print(f"类型为1的节点 {node} 的特征 {feat_a[node]} 与邻居特征的平均值 {avg_feat} 不一致")
            is_avg = False
            break
        else:
            print(f"类型为1的节点 {node} 的特征 {feat_a[node]} 与邻居特征的平均值 {avg_feat} 一致")
    else:
        print(f"节点 {node} 没有与类型为0的节点相连。")
        is_avg = False
        break

if is_avg:
    print("所有类型为1的节点的特征都是由与其直接相连的类型为0的节点的特征的平均值构成。")
else:
    print("至少有一个类型为1的节点的特征不是由与其直接相连的类型为0的节点的特征的平均值构成。")
