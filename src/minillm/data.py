"""数据集加载、切分与扰动（数据消融用）。

数据格式（jsonl，每行一个样本）：
  courseqa: {"id","question","answer","topic"}
  jsonout:  {"id","instruction","schema","gold"}
  intent:   {"id","text","label"}
"""
import json
import random
from pathlib import Path
from typing import Optional

SPLITS = ("train", "dev", "test")


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(rows, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class Dataset:
    """一个任务的三分数据 + 便捷访问。"""

    def __init__(self, task: str, root):
        self.task = task
        self.root = Path(root) / task
        self._cache = {}

    def split(self, name: str) -> list[dict]:
        if name not in self._cache:
            self._cache[name] = read_jsonl(self.root / f"{name}.jsonl")
        return self._cache[name]

    def sizes(self) -> dict:
        return {s: len(self.split(s)) for s in SPLITS}


def load_dataset(task: str, root) -> Dataset:
    return Dataset(task, root)


def subsample(rows: list[dict], n: Optional[int], seed: int = 42) -> list[dict]:
    """数据规模消融：确定性前缀采样（保持分布）。"""
    if n is None or n >= len(rows):
        return rows
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(rows)), n))
    return [rows[i] for i in idx]


def inject_label_noise(rows: list[dict], ratio: float, task: str, seed: int = 42) -> list[dict]:
    """数据质量消融：按比例破坏标签/答案，模拟低质量训练数据。

    - courseqa: 用另一条样本的答案替换
    - intent:   随机换标签
    - jsonout:  打乱 gold 的字段值
    返回新列表，不修改原数据；ratio=0 时原样返回。
    """
    if ratio <= 0:
        return rows
    rng = random.Random(seed)
    out = [dict(r) for r in rows]
    k = int(round(len(out) * ratio))
    picks = rng.sample(range(len(out)), k)
    for i in picks:
        j = rng.randrange(len(out))
        if task == "courseqa":
            out[i]["answer"] = out[j]["answer"]
        elif task == "intent":
            labels = sorted({r["label"] for r in rows})
            out[i]["label"] = rng.choice([l for l in labels if l != out[i]["label"]] or labels)
        elif task == "jsonout":
            gold = dict(out[i]["gold"])
            keys = list(gold.keys())
            if keys:
                kk = rng.choice(keys)
                gold[kk] = out[j]["gold"].get(kk, gold[kk])
            out[i]["gold"] = gold
    return out


def build_all_splits(out_root, seed: int = 42) -> dict:
    """构建三任务数据集并做切分，落盘到 out_root/<task>/{train,dev,test}.jsonl。

    courseqa 按 topic 分层切分，避免同 topic 同时出现在 train/test 造成指标虚高；
    其余任务随机切分 60/20/20。返回各任务各 split 的样本数。
    """
    from .tasks import BUILDERS
    rng = random.Random(seed)
    stats = {}
    for task, builder in BUILDERS.items():
        rows = builder(seed=seed)
        if task == "courseqa":
            by = {}
            for r in rows:
                by.setdefault(r["topic"], []).append(r)
            tr, dv, te = [], [], []
            for v in by.values():
                rng.shuffle(v)
                n = len(v)
                te += v[:max(1, n // 5)]
                dv += v[max(1, n // 5):max(2, 2 * n // 5)]
                tr += v[max(2, 2 * n // 5):]
        else:
            rng.shuffle(rows)
            n = len(rows)
            te, dv, tr = rows[:n // 5], rows[n // 5:2 * n // 5], rows[2 * n // 5:]
        for split, rr in (("train", tr), ("dev", dv), ("test", te)):
            write_jsonl(rr, Path(out_root) / task / f"{split}.jsonl")
        stats[task] = {"train": len(tr), "dev": len(dv), "test": len(te)}
    return stats


def to_sft_messages(row: dict, task: str) -> list[dict]:
    """统一转成 chat messages，供 SFT/LoRA 训练与推理共用。"""
    if task == "courseqa":
        return [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]},
        ]
    if task == "jsonout":
        return [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": json.dumps(row["gold"], ensure_ascii=False)},
        ]
    if task == "intent":
        return [
            {"role": "user", "content": f"判断下面这句话的意图，只输出意图标签：\n{row['text']}"},
            {"role": "assistant", "content": row["label"]},
        ]
    raise ValueError(task)
