# -*- coding: utf-8 -*-
import csv, sys
from collections import Counter

path = "so_ai_zhixiao_search_feedback1.csv"
csv.field_size_limit(10**9)

total = 0
cat_counter = Counter()
dead_total = 0
dead_valid = 0
dead_invalid = 0
dead_other = 0
date_counter = Counter()

with open(path, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        total += 1
        subtime = (row.get("subtime") or "")[:10]
        date_counter[subtime] += 1
        cat = (row.get("category") or "").strip()
        cat_counter[cat] += 1
        # 死链类投诉: category 含"死链"或"打不开"
        if ("死链" in cat) or ("打不开" in cat):
            if subtime == "2026-07-06":
                dead_total += 1
                v = (row.get("verdict") or "").strip().lower()
                if v == "valid":
                    dead_valid += 1
                elif v == "invalid":
                    dead_invalid += 1
                else:
                    dead_other += 1

print("总行数:", total)
print("subtime 日期分布:", dict(date_counter))
print()
print("== 7月6号 死链/打不开类投诉 ==")
print("投诉总数:", dead_total)
print("  审核属实 valid  :", dead_valid)
print("  审核不成立 invalid:", dead_invalid)
print("  其他/未审:", dead_other)
print()
print("== 全部投诉类型分布(category) ==")
for c, n in cat_counter.most_common():
    print(f"  {n:5d}  {c}")
