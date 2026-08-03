# EDA-GCL Linux 服务器部署 & 实验执行指南

## 前提条件
- Linux 服务器（内存 ≥ 32GB）
- 已安装 Docker（`docker --version` 确认）
- 有 SSH 登录权限

---

## 第一步：把项目传到服务器

在你的 **Windows 本机**打开 PowerShell，执行：

```powershell
# 打包项目（排除不需要的文件）
cd "D:\论文code\Edge Self-Adversarial Augmentation Enhances Graph Contrastive Learning"
tar -czf EDA-GCL-main.tar.gz --exclude='.git' --exclude='__pycache__' --exclude='log' --exclude='checkpoints' --exclude='article' --exclude='.claude' EDA-GCL-main

# 上传到服务器（替换为你的服务器地址和用户名）
scp EDA-GCL-main.tar.gz 用户名@服务器IP:/home/用户名/
```

---

## 第二步：在服务器上解压 + 构建 Docker 镜像

SSH 登录服务器后执行：

```bash
# 1. 解压
cd /home/用户名
tar -xzf EDA-GCL-main.tar.gz
cd EDA-GCL-main

# 2. 创建持久化目录（日志和checkpoint映射到宿主机，容器挂了也不丢）
mkdir -p /home/用户名/eda-gcl-output/log
mkdir -p /home/用户名/eda-gcl-output/checkpoints

# 3. 构建 Docker 镜像（首次约 5-10 分钟，之后有缓存会很快）
docker build -t eda-gcl .
```

如果构建失败（网络问题），加国内镜像源：
```bash
docker build -t eda-gcl \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  .
```

---

## 第三步：验证镜像能跑（冒烟测试，2 分钟）

```bash
# 跑 2 轮 IMDB 确认环境没问题
docker run --rm \
  eda-gcl \
  python train_ig.py --dataset IMDB --encoder hii --intent_source dynamic \
    --num_hidden 256 --num_epochs 2

# 看到 Epoch 001 和 Epoch 002 的输出就说明环境 OK
```

---

## 第四步：跑 3 个数据集的完整实验

### 4.1 ACM 数据集（完整模型 + Actor-Critic）

```bash
docker run -d \
  --name acm-full \
  -v /home/用户名/eda-gcl-output/log:/app/log \
  -v /home/用户名/eda-gcl-output/checkpoints:/app/checkpoints \
  eda-gcl \
  python train_ig.py \
    --dataset ACM \
    --encoder hii \
    --intent_source dynamic \
    --num_hidden 256 \
    --num_epochs 200 \
    --sparsify_topk 50 \
    --lambda_igqc 0.1 \
    --use_actor_critic \
    --ac_epochs 100 \
    --ac_max_size 1400 \
    --ac_size_sweep 200,400,600,800,1000,1200,1400
```

### 4.2 IMDB 数据集

```bash
docker run -d \
  --name imdb-full \
  -v /home/用户名/eda-gcl-output/log:/app/log \
  eda-gcl \
  python train_ig.py \
    --dataset IMDB \
    --encoder hii \
    --intent_source dynamic \
    --num_hidden 256 \
    --num_epochs 200 \
    --lambda_igqc 0.1 \
    --use_actor_critic \
    --ac_epochs 100
```

### 4.3 DBLP 数据集

```bash
docker run -d \
  --name dblp-full \
  -v /home/用户名/eda-gcl-output/log:/app/log \
  eda-gcl \
  python train_ig.py \
    --dataset DBLP \
    --encoder hii \
    --intent_source dynamic \
    --num_hidden 256 \
    --num_epochs 200 \
    --sparsify_topk 50 \
    --lambda_igqc 0.1 \
    --use_actor_critic \
    --ac_epochs 100
```

---

## 第五步：监控运行状态

```bash
# 查看哪些容器在跑
docker ps

# 实时看某个容器的输出（Ctrl+C 退出，不影响容器运行）
docker logs -f acm-full
docker logs -f imdb-full
docker logs -f dblp-full

# 看日志文件（从宿主机直接读）
tail -f /home/用户名/eda-gcl-output/log/run_ACM_*.log
tail -f /home/用户名/eda-gcl-output/log/run_IMDB_*.log
tail -f /home/用户名/eda-gcl-output/log/run_DBLP_*.log

# 查看容器内存占用
docker stats acm-full imdb-full dblp-full
```

---

## 第六步：消融实验（等第四步跑完后）

### 6.1 去掉 HII-GNN（创新点③）→ 用普通 GCN

```bash
# ACM
docker run -d --name acm-no-hii \
  -v /home/用户名/eda-gcl-output/log:/app/log \
  eda-gcl python train_ig.py \
    --dataset ACM --encoder gcn --intent_source static \
    --num_hidden 256 --num_epochs 200 --sparsify_topk 50

# IMDB
docker run -d --name imdb-no-hii \
  -v /home/用户名/eda-gcl-output/log:/app/log \
  eda-gcl python train_ig.py \
    --dataset IMDB --encoder gcn --intent_source static \
    --num_hidden 256 --num_epochs 200

# DBLP
docker run -d --name dblp-no-hii \
  -v /home/用户名/eda-gcl-output/log:/app/log \
  eda-gcl python train_ig.py \
    --dataset DBLP --encoder gcn --intent_source static \
    --num_hidden 256 --num_epochs 200 --sparsify_topk 50
```

### 6.2 去掉可疑节点加权（创新点④）

```bash
docker run -d --name acm-no-susp \
  -v /home/用户名/eda-gcl-output/log:/app/log \
  eda-gcl python train_ig.py \
    --dataset ACM --encoder hii --intent_source dynamic \
    --num_hidden 256 --num_epochs 200 --sparsify_topk 50 \
    --lambda_igqc 0.1 --suspicious_boost 1.0

docker run -d --name imdb-no-susp \
  -v /home/用户名/eda-gcl-output/log:/app/log \
  eda-gcl python train_ig.py \
    --dataset IMDB --encoder hii --intent_source dynamic \
    --num_hidden 256 --num_epochs 200 \
    --lambda_igqc 0.1 --suspicious_boost 1.0

docker run -d --name dblp-no-susp \
  -v /home/用户名/eda-gcl-output/log:/app/log \
  eda-gcl python train_ig.py \
    --dataset DBLP --encoder hii --intent_source dynamic \
    --num_hidden 256 --num_epochs 200 --sparsify_topk 50 \
    --lambda_igqc 0.1 --suspicious_boost 1.0
```

### 6.3 去掉 IGQC 意图对齐（创新点①②相关）

```bash
docker run -d --name acm-no-igqc \
  -v /home/用户名/eda-gcl-output/log:/app/log \
  eda-gcl python train_ig.py \
    --dataset ACM --encoder hii --intent_source dynamic \
    --num_hidden 256 --num_epochs 200 --sparsify_topk 50 \
    --lambda_igqc 0.0

docker run -d --name imdb-no-igqc \
  -v /home/用户名/eda-gcl-output/log:/app/log \
  eda-gcl python train_ig.py \
    --dataset IMDB --encoder hii --intent_source dynamic \
    --num_hidden 256 --num_epochs 200 \
    --lambda_igqc 0.0

docker run -d --name dblp-no-igqc \
  -v /home/用户名/eda-gcl-output/log:/app/log \
  eda-gcl python train_ig.py \
    --dataset DBLP --encoder hii --intent_source dynamic \
    --num_hidden 256 --num_epochs 200 --sparsify_topk 50 \
    --lambda_igqc 0.0
```

### 6.4 去掉 Actor-Critic

就是第四步的命令去掉 `--use_actor_critic` 相关参数，其实 AC 是单独评测的，Full Model 的结果里本身就包含不带 AC 的指标。

---

## 第七步：收集结果

```bash
# 把所有日志下载回本机
# 在 Windows PowerShell 执行：
scp -r 用户名@服务器IP:/home/用户名/eda-gcl-output/log ./results/

# 日志里包含所有指标：
# - micro_f1, macro_f1, acc（节点分类）
# - CS@10/20/50/oracle 的 F1/Precision/Recall（社区搜索）
# - ICRA alpha 分布
# - timing 信息
```

---

## 第八步：清理

```bash
# 停掉所有容器
docker stop acm-full imdb-full dblp-full
docker stop acm-no-hii imdb-no-hii dblp-no-hii
docker stop acm-no-susp imdb-no-susp dblp-no-susp
docker stop acm-no-igqc imdb-no-igqc dblp-no-igqc

# 删除容器
docker rm $(docker ps -aq --filter ancestor=eda-gcl)

# 删除镜像（可选）
docker rmi eda-gcl
```

---

## 常用排错命令

```bash
# 容器挂了，看退出原因
docker logs acm-full --tail 50

# 容器被 OOM 杀了会显示 Exit 137
docker inspect acm-full --format='{{.State.ExitCode}}'
# 137 = OOM killed, 0 = 正常结束, 1 = Python 报错

# 进入运行中的容器调试
docker exec -it acm-full bash

# 查看服务器内存
free -h
```

---

## 预估时间（CPU, num_hidden=256）

| 数据集 | 每轮预估 | 200轮主训练 | AC 100轮 | 评测 | 合计 |
|--------|---------|------------|---------|------|------|
| IMDB   | ~30s    | ~1.5h      | ~30min  | ~10min | ~2.5h |
| ACM    | ~60s    | ~3.5h      | ~50min  | ~15min | ~5h |
| DBLP   | ~90s    | ~5h        | ~1h     | ~20min | ~6.5h |

消融实验不跑 AC，每组少 1 小时左右。全部跑完预计 **2-3 天**。
