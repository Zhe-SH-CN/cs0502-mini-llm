"""三任务的评测指标。全部为纯函数、无依赖，便于单测与 CPU/GPU 复用。"""
import json
import re
import unicodedata
from collections import Counter
from typing import Any, Iterable


def normalize_zh(s: str) -> str:
    """中文友好的归一化：去全角空格/标点/大小写差异，保留语义字符。"""
    s = unicodedata.normalize("NFKC", s or "")
    s = s.lower()
    s = re.sub(r"[\s　]+", " ", s)
    s = re.sub(r"[，。、；：？！“”‘’（）《》〈〉,.;:?!'\"()\[\]{}<>/*\-_`~|]", "", s)
    return s.strip()


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_zh(pred) == normalize_zh(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    """字符级 F1（中文无空格分词，按字符计）。"""
    p, g = normalize_zh(pred), normalize_zh(gold)
    if not p or not g:
        return 1.0 if p == g else 0.0
    pc, gc = Counter(p), Counter(g)
    common = sum((pc & gc).values())
    if common == 0:
        return 0.0
    prec = common / sum(pc.values())
    rec = common / sum(gc.values())
    return 2 * prec * rec / (prec + rec)


def contains_answer(pred: str, gold: str) -> float:
    return 1.0 if normalize_zh(gold) and normalize_zh(gold) in normalize_zh(pred) else 0.0


def safe_json_loads(text: str) -> Any:
    """容忍模型输出带 ```json 围栏或前后杂音，抽取首个 JSON 对象/数组。"""
    if text is None:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except Exception:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = t.find(opener), t.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except Exception:
                continue
    return None


def schema_valid(obj: Any, schema: dict) -> tuple[bool, list[str]]:
    """校验对象是否满足 schema：required 键存在 + 类型匹配。返回 (是否通过, 错误列表)。"""
    errors = []
    if not isinstance(obj, dict):
        return False, ["not_an_object"]
    types = {"str": str, "int": int, "float": (int, float), "bool": bool, "list": list, "dict": dict}
    for key, spec in schema.get("fields", {}).items():
        if key not in obj:
            if spec.get("required", True):
                errors.append(f"missing:{key}")
            continue
        want = types.get(spec.get("type", "str"))
        val = obj[key]
        if want is bool:
            ok = isinstance(val, bool)
        elif want is int:
            ok = isinstance(val, int) and not isinstance(val, bool)
        else:
            ok = isinstance(val, want) if want else True
        if not ok:
            errors.append(f"type:{key}")
        if spec.get("enum") and val not in spec["enum"]:
            errors.append(f"enum:{key}")
    return (len(errors) == 0), errors


def field_accuracy(pred: Any, gold: dict, keys: Iterable[str]) -> float:
    """字段级准确率：仅统计 gold 中存在的键。"""
    if not isinstance(pred, dict):
        return 0.0
    keys = list(keys)
    if not keys:
        return 0.0
    hit = 0
    for k in keys:
        if k in gold and pred.get(k) == gold[k]:
            hit += 1
    return hit / len(keys)


def accuracy(preds: list, golds: list) -> float:
    if not golds:
        return 0.0
    return sum(1 for p, g in zip(preds, golds) if p == g) / len(golds)


def macro_f1(preds: list, golds: list) -> float:
    labels = sorted(set(golds) | set(preds))
    fs = []
    for lab in labels:
        tp = sum(1 for p, g in zip(preds, golds) if p == lab and g == lab)
        fp = sum(1 for p, g in zip(preds, golds) if p == lab and g != lab)
        fn = sum(1 for p, g in zip(preds, golds) if p != lab and g == lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(fs) / len(fs) if fs else 0.0
