# -*- coding: utf-8 -*-
"""统计 category == '网站死链、打不开、内容已删除' 且 verdict == 'valid'(属实) 的数量。"""
import csv
import glob
import os

csv.field_size_limit(10 ** 9)

TARGET_CATEGORY = "网站死链、打不开、内容已删除"


def count_file(path):
    total = 0           # 该类投诉总数
    valid = 0           # 审核属实
    invalid = 0         # 审核不成立
    other = 0           # 其他/未审
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cat = (row.get("category") or "").strip()
            if cat != TARGET_CATEGORY:
                continue
            total += 1
            v = (row.get("verdict") or "").strip().lower()
            if v == "valid":
                valid += 1
            elif v == "invalid":
                invalid += 1
            else:
                other += 1
    return total, valid, invalid, other


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(os.path.join(here, "*.csv")))
    if not files:
        print("当前文件夹没有 .csv 文件")
        return

    g_total = g_valid = g_invalid = g_other = 0
    for path in files:
        total, valid, invalid, other = count_file(path)
        print(f"[{os.path.basename(path)}]")
        print(f"  category='{TARGET_CATEGORY}' 投诉总数: {total}")
        print(f"    valid   (属实)  : {valid}")
        print(f"    invalid (不成立): {invalid}")
        print(f"    其他/未审       : {other}")
        g_total += total
        g_valid += valid
        g_invalid += invalid
        g_other += other

    if len(files) > 1:
        print("=" * 40)
        print(f"全部文件合计 投诉总数: {g_total}")
        print(f"  valid   (属实)  : {g_valid}")
        print(f"  invalid (不成立): {g_invalid}")
        print(f"  其他/未审       : {g_other}")


if __name__ == "__main__":
    main()
