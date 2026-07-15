"""
创新点 (§7.2 Step4): Actor-Critic 对抗图生成器。

从查询节点出发, 沿图 frontier 逐步加点构建对抗社区。
- Actor: 对 frontier 候选节点 + STOP 动作打分, 采样/贪心。
- Critic: 估计当前状态的价值。
- 奖励: 自监督社区质量 Q(C), 全程不使用节点标签 (忠实于推理设定)。

用训练后冻结的节点表示 z, Actor/Critic 只做轻量 MLP。
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticCommunityBuilder(nn.Module):
    """
    状态 s = [centroid(C) || z_q || intent || mean(z_frontier)]
    动作 = 选 frontier 中某候选加入, 或 STOP。
    """

    def __init__(self, emb_dim, intent_dim, hidden=128, max_size=60, lam_boost=0.3,
                 max_frontier=128):
        super().__init__()
        self.emb_dim = emb_dim
        self.intent_dim = intent_dim
        self.max_size = max_size
        self.lam_boost = lam_boost
        self.max_frontier = max_frontier

        # 候选打分: [z_cand || centroid || z_q || intent] -> logit
        feat_dim = emb_dim * 3 + intent_dim
        self.actor = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        # STOP 分 & Critic 都吃状态: [centroid || z_q || intent || mean_frontier]
        state_dim = emb_dim * 3 + intent_dim
        self.stop_head = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def _frontier_list(self, zc, z_q, frontier):
        """候选过多时按与查询相似度 top-k 截断, 限制内存/计算。"""
        if len(frontier) <= self.max_frontier:
            return sorted(frontier)
        idx = torch.tensor(sorted(frontier), device=zc.device)
        sims = zc[idx] @ z_q
        top = torch.topk(sims, self.max_frontier).indices
        return idx[top].tolist()

    def _state(self, zc, comm, frontier_list, z_q, intent):
        centroid = zc[comm].mean(dim=0) if len(comm) > 0 else torch.zeros_like(z_q)
        if frontier_list:
            mean_fr = zc[frontier_list].mean(dim=0)
        else:
            mean_fr = torch.zeros_like(z_q)
        return centroid, mean_fr, torch.cat([centroid, z_q, intent, mean_fr], dim=-1)

    def step_distribution(self, zc, comm, frontier_list, z_q, intent):
        """返回 (动作分布 Categorical, value, centroid)。动作 0..C-1 为候选, C 为 STOP。"""
        centroid, _mean_fr, state = self._state(zc, comm, frontier_list, z_q, intent)
        value = self.critic(state).squeeze(-1)

        c = len(frontier_list)
        if c > 0:
            z_cand = zc[frontier_list]                          # [c, H]
            cen_exp = centroid.unsqueeze(0).expand(c, -1)
            q_exp = z_q.unsqueeze(0).expand(c, -1)
            int_exp = intent.unsqueeze(0).expand(c, -1)
            cand_feat = torch.cat([z_cand, cen_exp, q_exp, int_exp], dim=-1)
            cand_logits = self.actor(cand_feat).squeeze(-1)     # [c]
        else:
            cand_logits = state.new_zeros(0)

        stop_logit = self.stop_head(state)                      # [1]
        logits = torch.cat([cand_logits, stop_logit], dim=0)    # [c+1]
        dist = torch.distributions.Categorical(logits=logits)
        return dist, value

    @torch.no_grad()
    def build(self, z, adj, q, intent, node_boost=None, greedy=True):
        """推理: 从 q 扩展成社区集合 (含 q)。"""
        self.eval()
        zc = F.normalize(z, dim=-1)
        z_q = zc[q]
        q = int(q)
        visited = {q}
        frontier = set(adj[q]) - visited

        for _ in range(self.max_size - 1):
            if not frontier:
                break
            frontier_list = self._frontier_list(zc, z_q, frontier)
            dist, _v = self.step_distribution(zc, sorted(visited), frontier_list,
                                              z_q, intent)
            action = int(torch.argmax(dist.probs)) if greedy else int(dist.sample())
            if action == len(frontier_list):          # STOP
                break
            node = frontier_list[action]
            visited.add(node)
            frontier.discard(node)
            frontier.update(set(adj[node]) - visited)
        return visited

    @torch.no_grad()
    def build_sequence(self, z, adj, q, intent, node_boost=None, max_size=None):
        """贪心扩展并返回加入顺序 [q, n1, n2, ...]。
        贪心序列与 max_size 无关(每步只依赖已选集合与 frontier),
        故一次展开到最大 cap, 即可用前缀切片还原任意更小 size 的社区,
        供一次评测扫出整条 P-R 曲线。"""
        self.eval()
        cap = max_size if max_size is not None else self.max_size
        zc = F.normalize(z, dim=-1)
        z_q = zc[q]
        q = int(q)
        visited = {q}
        order = [q]
        frontier = set(adj[q]) - visited

        for _ in range(cap - 1):
            if not frontier:
                break
            frontier_list = self._frontier_list(zc, z_q, frontier)
            dist, _v = self.step_distribution(zc, sorted(visited), frontier_list,
                                              z_q, intent)
            action = int(torch.argmax(dist.probs))
            if action == len(frontier_list):          # STOP
                break
            node = frontier_list[action]
            visited.add(node)
            order.append(node)
            frontier.discard(node)
            frontier.update(set(adj[node]) - visited)
        return order

    def rollout(self, zc, adj, q, intent, node_boost, num_nodes, gamma=0.95):
        """训练用采样轨迹, 返回 (log_probs, values, returns)。
        自监督边际奖励: 加入节点与查询的相似度超过全图均值则为正, 否则为负;
        可疑节点额外加权。不使用标签。STOP 奖励 0, 避免"立即停止"塌缩。"""
        z_q = zc[q]
        v = zc @ z_q                                   # [N] 与查询的余弦相似度
        margin = float(v.mean())
        q = int(q)
        visited = {q}
        frontier = set(adj[q]) - visited

        log_probs, values, rewards = [], [], []
        for _ in range(self.max_size - 1):
            frontier_list = self._frontier_list(zc, z_q, frontier)
            dist, value = self.step_distribution(zc, sorted(visited), frontier_list,
                                                 z_q, intent)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            values.append(value)
            a = int(action)

            if a == len(frontier_list):               # STOP
                rewards.append(0.0)
                break
            node = frontier_list[a]
            visited.add(node)
            frontier.discard(node)
            frontier.update(set(adj[node]) - visited)

            r = float(v[node]) - margin               # 高于均值相似度 -> 正奖励
            if node_boost is not None:
                r += self.lam_boost * (float(node_boost[node]) - 0.5)
            rewards.append(r)
            if not frontier:
                break

        # 折扣回报
        returns, R = [], 0.0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return log_probs, values, returns


def train_actor_critic(builder, z, adj, intent, node_boost=None, epochs=100,
                       lr=1e-3, num_queries=64, seed=0, entropy_beta=0.01,
                       verbose=True):
    """
    自监督 A2C 训练。z 为冻结节点表示 [N, H]; intent 为意图向量 [intent_dim]。
    adj 为邻接表 (list[set]); node_boost 为 [N] 可疑分或 None。
    """
    device = z.device
    z = z.detach()
    zc = F.normalize(z, dim=-1)
    N = z.size(0)
    if node_boost is not None:
        node_boost = node_boost.detach().to(device)
    intent = intent.detach().to(device)

    optim = torch.optim.Adam(builder.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    t_start = time.time()
    if verbose:
        print(f'[AC] 开始训练 {time.strftime("%Y-%m-%d %H:%M:%S")}')

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        builder.train()
        qs = rng.choice(N, size=min(num_queries, N), replace=False)
        optim.zero_grad()

        batch_loss = 0.0
        batch_reward = 0.0
        n_used = 0
        n_q = len(qs)
        for q in qs:
            if len(adj[int(q)]) == 0:                 # 孤立点跳过
                continue
            log_probs, values, returns = builder.rollout(
                zc, adj, q, intent, node_boost, N
            )
            if not log_probs:
                continue
            log_probs = torch.stack(log_probs)
            values = torch.stack(values)
            returns_t = torch.tensor(returns, device=device, dtype=values.dtype)

            advantage = returns_t - values.detach()
            policy_loss = -(log_probs * advantage).sum()
            value_loss = F.mse_loss(values, returns_t, reduction='sum')
            entropy = -(log_probs.exp() * log_probs).sum()
            loss = policy_loss + 0.5 * value_loss - entropy_beta * entropy

            # 每条轨迹单独 backward, 梯度累加后立即释放计算图 (省内存)
            (loss / n_q).backward()
            batch_loss += float(loss)
            batch_reward += float(returns_t.sum())
            n_used += 1

        if n_used == 0:
            continue
        optim.step()

        ep_dt = time.time() - ep_t0
        if verbose:
            print(f'[AC] Epoch={epoch:03d}, avg_return={batch_reward / n_used:.4f}, '
                  f'loss={batch_loss / n_used:.4f}, '
                  f'this epoch {ep_dt:.2f}s, total {time.time() - t_start:.2f}s')

    if verbose:
        print(f'[AC] 结束训练 {time.strftime("%Y-%m-%d %H:%M:%S")} '
              f'总耗时 {time.time() - t_start:.2f}s')
    return builder
