"""实验结果记录器：所有 run 的指标统一落盘，保证报告可追溯、可复现。

落盘结构：
  results/runs.jsonl            每次 run 一行（config + metrics + 环境信息）
  results/<run_id>.json         单次 run 的完整明细（含逐样本预测）
  results/summary.csv           扁平化汇总，便于做表/画图
"""
import csv
import json
import platform
import time
from pathlib import Path


def env_info() -> dict:
    info = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
            info["cuda_version"] = torch.version.cuda
    except Exception:
        info["torch"] = None
        info["cuda_available"] = False
    return info


class Recorder:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def log_run(self, run_id: str, config: dict, metrics: dict, per_sample=None, env=None) -> Path:
        env = env or env_info()
        row = {"run_id": run_id, "config": config, "metrics": metrics, "env": env}
        with (self.root / "runs.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        detail = dict(row)
        detail["per_sample"] = per_sample or []
        p = self.root / f"{run_id}.json"
        p.write_text(json.dumps(detail, ensure_ascii=False, indent=1), encoding="utf-8")
        self._append_csv(row)
        return p

    def _append_csv(self, row: dict) -> None:
        path = self.root / "summary.csv"
        flat = {"run_id": row["run_id"], **{f"cfg_{k}": v for k, v in row["config"].items()
                                            if not isinstance(v, (dict, list))},
                **{f"m_{k}": v for k, v in row["metrics"].items() if not isinstance(v, (dict, list))},
                **{f"env_{k}": v for k, v in row["env"].items()}}
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sorted(flat.keys()))
            if write_header:
                w.writeheader()
            w.writerow(flat)

    def load_runs(self) -> list[dict]:
        p = self.root / "runs.jsonl"
        if not p.exists():
            return []
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
