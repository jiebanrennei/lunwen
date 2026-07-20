"""
从原始 IMDb TSV 数据构建 imdb_new 异构图数据集。

输出文件 (与 self_imdb 格式对齐):
  m_feat.npz       — 电影节点稠密文本特征 (sentence-transformer 编码)
  labels.npy        — 电影类型标签 (0=Action, 1=Comedy, 2=Drama)
  ma.txt            — 电影-演员 边列表
  md.txt            — 电影-导演 边列表
  mam.npz           — Movie-Actor-Movie 元路径邻接矩阵
  mdm.npz           — Movie-Director-Movie 元路径邻接矩阵
  adj.npz           — 全局邻接矩阵 (含自环, 多节点类型)
  node_types.npy    — 节点类型数组 (0=movie, 1=actor, 2=director)
  edge2type.pickle  — (global_src, global_dst) → edge_type 映射
  movie_text.tsv    — movie_id → tconst, title, genres (文本来源)

用法:
  python build_imdb_new.py --raw_dir <解压后的 imbd 目录>
"""

import argparse
import gzip
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np
import scipy.sparse as sp


# ======================================================================
# Step 1: 解析原始 IMDb TSV
# ======================================================================

def parse_basics(path):
    """解析 title.basics.tsv.gz → {tconst: {title, year, genres}}，仅保留 movie。"""
    movies = {}
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        next(f)  # skip header
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if parts[1] != 'movie':
                continue
            tconst = parts[0]
            genres = parts[8]
            if genres == r'\N':
                continue
            movies[tconst] = {
                'title': parts[2],
                'year': parts[5],
                'genres': genres,
            }
    return movies


def parse_ratings(path):
    """解析 title.ratings.tsv.gz → {tconst: (rating, numVotes)}。"""
    ratings = {}
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.rstrip('\n').split('\t')
            ratings[parts[0]] = (float(parts[1]), int(parts[2]))
    return ratings


def parse_principals(path, movie_set):
    """解析 title.principals.tsv.gz，只保留 movie_set 中的电影。
    返回 movie→actors, movie→directors 映射。"""
    m2actors = defaultdict(set)
    m2directors = defaultdict(set)
    actor_cats = {'actor', 'actress'}
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.rstrip('\n').split('\t')
            tc = parts[0]
            if tc not in movie_set:
                continue
            cat = parts[3]
            nc = parts[2]
            if cat in actor_cats:
                m2actors[tc].add(nc)
            elif cat == 'director':
                m2directors[tc].add(nc)
    return m2actors, m2directors


def parse_names(path, name_set):
    """解析 name.basics.tsv.gz，只保留 name_set 中的人名。"""
    names = {}
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.rstrip('\n').split('\t')
            nc = parts[0]
            if nc in name_set:
                names[nc] = parts[1]
    return names


# ======================================================================
# Step 2: 筛选 & 采样
# ======================================================================

TARGET_GENRES = ['Action', 'Comedy', 'Drama']
GENRE2LABEL = {g: i for i, g in enumerate(TARGET_GENRES)}


def assign_label(genres_str):
    """按优先级分配单一标签: 第一个命中 TARGET_GENRES 的类型。"""
    for g in genres_str.split(','):
        if g in GENRE2LABEL:
            return GENRE2LABEL[g]
    return None


def select_movies(movies, ratings, m2actors, m2directors,
                  per_class=1500, min_actors=2, min_directors=1):
    """筛选有演员+导演+评分的电影, 每类取 per_class 部 (按 numVotes 降序)。"""
    candidates = defaultdict(list)
    for tc, info in movies.items():
        label = assign_label(info['genres'])
        if label is None:
            continue
        if len(m2actors.get(tc, set())) < min_actors:
            continue
        if len(m2directors.get(tc, set())) < min_directors:
            continue
        votes = ratings.get(tc, (0, 0))[1]
        candidates[label].append((tc, votes))

    selected = {}
    for label in range(len(TARGET_GENRES)):
        pool = sorted(candidates[label], key=lambda x: -x[1])
        for tc, _ in pool[:per_class]:
            if tc not in selected:
                selected[tc] = label
    return selected


# ======================================================================
# Step 3: 重建图结构 & 编号
# ======================================================================

def build_graph(selected, m2actors, m2directors):
    """构建 movie→actor、movie→director 边, 返回重编号后的结果。"""
    # 收集所有出现的 actor/director
    actor_set, director_set = set(), set()
    for tc in selected:
        actor_set.update(m2actors.get(tc, set()))
        director_set.update(m2directors.get(tc, set()))

    # 重编号: movie 0..M-1, actor 0..A-1, director 0..D-1
    tc_list = sorted(selected.keys())
    tc2mid = {tc: i for i, tc in enumerate(tc_list)}
    actor_list = sorted(actor_set)
    nc2aid = {nc: i for i, nc in enumerate(actor_list)}
    dir_list = sorted(director_set)
    nc2did = {nc: i for i, nc in enumerate(dir_list)}

    ma_edges = []  # (movie_local, actor_local)
    md_edges = []  # (movie_local, director_local)
    for tc in tc_list:
        mid = tc2mid[tc]
        for nc in m2actors.get(tc, set()):
            ma_edges.append((mid, nc2aid[nc]))
        for nc in m2directors.get(tc, set()):
            md_edges.append((mid, nc2did[nc]))

    labels = np.array([selected[tc] for tc in tc_list], dtype=np.int64)

    return (tc_list, tc2mid, actor_list, nc2aid, dir_list, nc2did,
            ma_edges, md_edges, labels)


def build_metapath(edges, num_src, num_mid):
    """从二部图边列表构建 src-mid-src 元路径邻接矩阵。
    edges: [(src_id, mid_id), ...]
    返回 [num_src × num_src] 稀疏矩阵。
    """
    # 先建 mid → [src_ids] 映射
    mid2srcs = defaultdict(list)
    for s, m in edges:
        mid2srcs[m].append(s)

    rows, cols, data = [], [], []
    for m, srcs in mid2srcs.items():
        for i, s1 in enumerate(srcs):
            for s2 in srcs:
                if s1 != s2:
                    rows.append(s1)
                    cols.append(s2)
                    data.append(1)

    mat = sp.coo_matrix((data, (rows, cols)),
                        shape=(num_src, num_src)).tocsr()
    return mat


def build_global_adj(num_m, num_a, num_d, ma_edges, md_edges):
    """构建全局邻接矩阵 (含自环)，节点排列: [M | A | D]。"""
    total = num_m + num_a + num_d
    rows, cols, etypes = [], [], []

    rel2type = {
        'ma': 0, 'am': 2,
        'md': 1, 'dm': 3,
        'mm': 4, 'aa': 5, 'dd': 6,
    }

    for mid, aid in ma_edges:
        gm, ga = mid, num_m + aid
        rows += [gm, ga]
        cols += [ga, gm]
        etypes += [rel2type['ma'], rel2type['am']]

    for mid, did in md_edges:
        gm, gd = mid, num_m + num_a + did
        rows += [gm, gd]
        cols += [gd, gm]
        etypes += [rel2type['md'], rel2type['dm']]

    # 自环
    for i in range(num_m):
        rows.append(i); cols.append(i); etypes.append(rel2type['mm'])
    for i in range(num_a):
        g = num_m + i
        rows.append(g); cols.append(g); etypes.append(rel2type['aa'])
    for i in range(num_d):
        g = num_m + num_a + i
        rows.append(g); cols.append(g); etypes.append(rel2type['dd'])

    adj = sp.coo_matrix((np.ones(len(rows)), (rows, cols)),
                        shape=(total, total)).tocsr()
    edge2type = {(r, c): t for r, c, t in zip(rows, cols, etypes)}
    node_types = ([0] * num_m + [1] * num_a + [2] * num_d)
    return adj, edge2type, np.array(node_types, dtype=np.int64)


# ======================================================================
# Step 4: 文本特征编码
# ======================================================================

def encode_text_features(tc_list, movies, model_name, batch_size=256):
    """用 sentence-transformer 把每部电影的 "Title: X. Genres: Y" 编码成稠密向量。"""
    from sentence_transformers import SentenceTransformer

    texts = []
    for tc in tc_list:
        info = movies[tc]
        text = f"Title: {info['title']}. Genres: {info['genres']}."
        if info['year'] != r'\N':
            text += f" Year: {info['year']}."
        texts.append(text)

    print(f"[text] 编码 {len(texts)} 条电影文本, model={model_name} ...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=batch_size,
                              show_progress_bar=True, normalize_embeddings=True)
    return embeddings, texts


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str,
                        default=r'C:\Users\zhaobin3\Desktop\新建文件夹 (2)\IMDb\raw\imbd',
                        help='解压后的 IMDb TSV 目录')
    parser.add_argument('--out_dir', type=str, default='./',
                        help='输出目录 (默认当前目录)')
    parser.add_argument('--per_class', type=int, default=1500,
                        help='每类采样电影数')
    parser.add_argument('--min_actors', type=int, default=2)
    parser.add_argument('--min_directors', type=int, default=1)
    parser.add_argument('--encoder', type=str,
                        default='paraphrase-multilingual-MiniLM-L12-v2')
    args = parser.parse_args()

    raw = args.raw_dir
    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    # --- 解析原始数据 ---
    print("[1/7] 解析 title.basics ...")
    movies = parse_basics(os.path.join(raw, 'title.basics.tsv.gz'))
    print(f"  总电影数: {len(movies)}")

    print("[2/7] 解析 title.ratings ...")
    ratings = parse_ratings(os.path.join(raw, 'title.ratings.tsv.gz'))
    print(f"  有评分的条目: {len(ratings)}")

    print("[3/7] 解析 title.principals (仅目标电影) ...")
    # 先粗筛有目标 genre 的电影
    target_tcs = {tc for tc, info in movies.items()
                  if assign_label(info['genres']) is not None}
    print(f"  候选电影 (有目标genre): {len(target_tcs)}")
    m2actors, m2directors = parse_principals(
        os.path.join(raw, 'title.principals.tsv.gz'), target_tcs)
    print(f"  有演员信息: {len(m2actors)}, 有导演信息: {len(m2directors)}")

    # --- 采样 ---
    print("[4/7] 筛选 & 采样 ...")
    selected = select_movies(movies, ratings, m2actors, m2directors,
                             per_class=args.per_class,
                             min_actors=args.min_actors,
                             min_directors=args.min_directors)
    print(f"  选中电影: {len(selected)}")
    label_dist = Counter(selected.values())
    for label in range(3):
        print(f"    {TARGET_GENRES[label]}: {label_dist[label]}")

    # --- 构建图 ---
    print("[5/7] 构建图结构 ...")
    (tc_list, tc2mid, actor_list, nc2aid, dir_list, nc2did,
     ma_edges, md_edges, labels) = build_graph(selected, m2actors, m2directors)

    num_m, num_a, num_d = len(tc_list), len(actor_list), len(dir_list)
    print(f"  节点: M={num_m}, A={num_a}, D={num_d}")
    print(f"  边: M-A={len(ma_edges)}, M-D={len(md_edges)}")

    mam = build_metapath(ma_edges, num_m, len(actor_list))
    mdm = build_metapath(md_edges, num_m, len(dir_list))
    print(f"  元路径: MAM nnz={mam.nnz}, MDM nnz={mdm.nnz}")

    adj, edge2type, node_types = build_global_adj(
        num_m, num_a, num_d, ma_edges, md_edges)

    # --- 文本特征 ---
    print("[6/7] 编码文本特征 ...")
    embeddings, texts = encode_text_features(tc_list, movies, args.encoder)
    feat = sp.csr_matrix(embeddings.astype(np.float32))

    # --- 保存 ---
    print("[7/7] 保存文件 ...")
    sp.save_npz(os.path.join(out, 'm_feat.npz'), feat)
    np.save(os.path.join(out, 'labels.npy'), labels)
    sp.save_npz(os.path.join(out, 'mam.npz'), mam)
    sp.save_npz(os.path.join(out, 'mdm.npz'), mdm)
    sp.save_npz(os.path.join(out, 'adj.npz'), adj)
    np.save(os.path.join(out, 'node_types.npy'), node_types)

    with open(os.path.join(out, 'edge2type.pickle'), 'wb') as f:
        pickle.dump(edge2type, f)

    with open(os.path.join(out, 'ma.txt'), 'w') as f:
        for mid, aid in sorted(ma_edges):
            f.write(f"{mid}\t{aid}\n")

    with open(os.path.join(out, 'md.txt'), 'w') as f:
        for mid, did in sorted(md_edges):
            f.write(f"{mid}\t{did}\n")

    # 文本映射: 方便后续查意图来源
    with open(os.path.join(out, 'movie_text.tsv'), 'w', encoding='utf-8') as f:
        f.write("movie_id\ttconst\ttitle\tgenres\tyear\n")
        for i, tc in enumerate(tc_list):
            info = movies[tc]
            f.write(f"{i}\t{tc}\t{info['title']}\t{info['genres']}\t{info['year']}\n")

    print("\n=== 完成 ===")
    print(f"输出目录: {os.path.abspath(out)}")
    print(f"电影: {num_m}, 演员: {num_a}, 导演: {num_d}")
    print(f"特征维度: {embeddings.shape[1]}")
    print(f"标签分布: " + ", ".join(
        f"{TARGET_GENRES[i]}={label_dist[i]}" for i in range(3)))


if __name__ == '__main__':
    main()
