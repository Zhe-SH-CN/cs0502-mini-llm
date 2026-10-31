#!/usr/bin/env python3
"""基线评测 CLI（CPU 可跑，供 CNB CI 冒烟与本地验证）。

用法：python scripts/run_eval.py --data data --results results --model minimind-26m \
        --task courseqa --max-samples 8
"""
import argparse
import json

from minillm.config import ExpConfig
from minillm.data import load_dataset
from minillm.evaluate import evaluate, load_model
from minillm.record import Recorder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--model", default="minimind-26m")
    ap.add_argument("--task", default="courseqa")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-samples", type=int, default=8)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--attn", default="eager")
    a = ap.parse_args()

    cfg = ExpConfig(run_id=f"cpu-baseline-{a.model}-{a.task}", model_key=a.model, task=a.task,
                    method="none", dtype=a.dtype, attn=a.attn, max_new_tokens=48)
    model, tok = load_model(cfg, device="cpu")
    m, ps = evaluate(model, tok, load_dataset(a.task, a.data), a.task, cfg,
                     split=a.split, max_samples=a.max_samples)
    Recorder(a.results).log_run(cfg.run_id, cfg.to_dict(), m, ps)
    print(json.dumps(m, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
