import numpy as np
import torch
from actor_critic import ActorCriticCommunityBuilder, train_actor_critic

torch.manual_seed(0)
np.random.seed(0)

# 两个簇, 每簇 30 点; 簇内全连, 簇间少量边
N = 60
H = 16
z = torch.randn(N, H)
z[:30] += torch.tensor([3.0] + [0.0] * (H - 1))   # 簇 A 偏移
z[30:] += torch.tensor([0.0, 3.0] + [0.0] * (H - 2))  # 簇 B 偏移

adj = [set() for _ in range(N)]
def link(a, b):
    adj[a].add(b); adj[b].add(a)
for grp in (range(30), range(30, 60)):
    lst = list(grp)
    for i in lst:
        for j in lst:
            if i < j and np.random.rand() < 0.3:
                link(i, j)
# 簇间少量桥
for _ in range(5):
    link(np.random.randint(0, 30), np.random.randint(30, 60))

intent = torch.randn(H)
node_boost = torch.rand(N)  # 触发 self.lam_boost 分支

builder = ActorCriticCommunityBuilder(H, H, hidden=32, max_size=40)
train_actor_critic(builder, z, adj, intent, node_boost=node_boost,
                   epochs=30, lr=1e-3, num_queries=32, seed=0)

sizes = []
for q in [0, 5, 35, 50]:
    comm = builder.build(z, adj, q, intent, node_boost=node_boost, greedy=True)
    sizes.append(len(comm))
    same_cluster = sum(1 for n in comm if (n < 30) == (q < 30))
    print(f'q={q} size={len(comm)} same_cluster_frac={same_cluster/len(comm):.2f}')

print('MAX_SIZE=', max(sizes))
assert max(sizes) > 1, 'collapse: all communities size 1'
print('SMOKE_OK')
