#!/usr/bin/env python3
"""构建数据集 CLI（本地 / CNB CI 通用）。

用法：python scripts/build_data.py --out data
"""
import argparse
import json

from minillm.data import build_all_splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    stats = build_all_splits(a.out, seed=a.seed)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
