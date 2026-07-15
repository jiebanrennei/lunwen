# -*- coding: utf-8 -*-
"""对比 feedback1.csv 与 feedback2.csv 的复核结果(verdict), 找出同一 id 两边不一致的记录。

输出: 每条不一致的 id、category、file1 的 verdict、file2 的 verdict。
同时把结果写入 verdict_diff.txt (UTF-8), 避免控制台中文乱码。
"""
import csv
import os
import sys

csv.field_size_limit(10 ** 9)

FILE1 = "so_ai_zhixiao_search_feedback1.csv"
FILE2 = "so_ai_zhixiao_search_feedback2.csv"


def load(path):
    """返回 {id: row_dict}。"""
    d = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = (row.get("id") or "").strip()
            if rid:
                d[rid] = row
    return d


def norm(v):
    return (v or "").strip().lower()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    d1 = load(os.path.join(here, FILE1))
    d2 = load(os.path.join(here, FILE2))

    ids1 = set(d1)
    ids2 = set(d2)
    common = ids1 & ids2

    diffs = []          # verdict 不一致
    only1 = ids1 - ids2
    only2 = ids2 - ids1

    for rid in common:
        v1 = norm(d1[rid].get("verdict"))
        v2 = norm(d2[rid].get("verdict"))
        if v1 != v2:
            diffs.append((rid,
                          d1[rid].get("category", "").strip(),
                          d1[rid].get("verdict", "").strip(),
                          d2[rid].get("verdict", "").strip()))

    # id 按数值排序
    def as_int(x):
        try:
            return int(x)
        except ValueError:
            return 10 ** 18

    diffs.sort(key=lambda t: as_int(t[0]))

    lines = []
    lines.append(f"file1={FILE1}  行数={len(d1)}")
    lines.append(f"file2={FILE2}  行数={len(d2)}")
    lines.append(f"共同 id 数={len(common)}  仅 file1={len(only1)}  仅 file2={len(only2)}")
    lines.append(f"复核结果(verdict)不一致条数={len(diffs)}")
    lines.append("")
    lines.append(f"{'id':>8} | {'file1_verdict':<14} | {'file2_verdict':<14} | category")
    lines.append("-" * 70)
    for rid, cat, a, b in diffs:
        lines.append(f"{rid:>8} | {a:<14} | {b:<14} | {cat}")

    if only1:
        lines.append("")
        lines.append(f"仅存在于 file1 的 id ({len(only1)}): "
                     + ", ".join(sorted(only1, key=as_int)))
    if only2:
        lines.append("")
        lines.append(f"仅存在于 file2 的 id ({len(only2)}): "
                     + ", ".join(sorted(only2, key=as_int)))

    text = "\n".join(lines)
    out = os.path.join(here, "verdict_diff.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)

    # 控制台尽量用 utf-8 输出
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(text)
    print(f"\n[已写入] {out}")


if __name__ == "__main__":
    main()

