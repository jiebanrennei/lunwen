FROM python:3.10-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# 安装 PyTorch CPU 版 (省显存，与你 Windows 上一致)
RUN pip install --no-cache-dir \
    torch==2.0.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# 安装 PyG 及其依赖 (CPU 版)
RUN pip install --no-cache-dir \
    torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.0.0+cpu.html && \
    pip install --no-cache-dir torch-geometric==2.6.1

# 其他依赖
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy \
    scikit-learn==1.6.1 \
    PyYAML==6.0.2 \
    tqdm \
    deeprobust==0.2.11

# 复制项目代码
COPY . /app/

# 创建日志和检查点目录
RUN mkdir -p /app/log /app/checkpoints

# 默认命令
CMD ["python", "train_ig.py", "--dataset", "ACM", "--encoder", "hii", "--intent_source", "dynamic"]
