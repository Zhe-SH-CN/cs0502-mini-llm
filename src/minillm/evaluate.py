"""统一评测：加载模型 -> 生成 -> 按任务算指标。

T4 适配要点：
- dtype=float16（sm7.5 无原生 bf16）
- attn_implementation=sdpa（sm7.5 不支持 FlashAttention-2）
- 4bit 走 bitsandbytes NF4（QLoRA 推理/训练共用）
"""
import json
import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from . import metrics as M
from .config import ExpConfig
from .data import Dataset, to_sft_messages
from .tasks import intent as intent_mod
from .tasks import jsonout as jsonout_mod

_DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def load_model(cfg: ExpConfig, device: str = "auto"):
    kwargs = dict(torch_dtype=_DTYPE[cfg.dtype], attn_implementation=cfg.attn)
    if cfg.use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=_DTYPE[cfg.dtype],
            bnb_4bit_use_double_quant=True,
        )
        kwargs.pop("torch_dtype", None)
    model = AutoModelForCausalLM.from_pretrained(cfg.hf_id, **kwargs)
    tok = AutoTokenizer.from_pretrained(cfg.hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if device != "auto":
        model.to(device)
    model.eval()
    return model, tok


def build_prompt(tok, row: dict, task: str) -> str:
    msgs = to_sft_messages(row, task)[:-1]  # 去掉 assistant 回答，只留 user
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return msgs[0]["content"]


@torch.no_grad()
def generate_one(model, tok, prompt: str, cfg: ExpConfig, max_new: int) -> str:
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new,
        do_sample=cfg.do_sample,
        temperature=cfg.temperature if cfg.do_sample else None,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    new = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True)


def _norm_intent(raw: str) -> Optional[str]:
    r = M.normalize_zh(raw)
    for lab in intent_mod.LABELS:
        if M.normalize_zh(lab) == r or M.normalize_zh(lab) in r:
            return lab
    return None


def evaluate(model, tok, ds: Dataset, task: str, cfg: ExpConfig, split: str = "test",
             max_samples: Optional[int] = None):
    rows = ds.split(split)
    if max_samples:
        rows = rows[:max_samples]
    per_sample = []
    t0 = time.time()
    peak_mem_before = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    if task == "courseqa":
        for row in rows:
            pred = generate_one(model, tok, build_prompt(tok, row, task), cfg, cfg.max_new_tokens)
            em = M.exact_match(pred, row["answer"])
            f1 = M.token_f1(pred, row["answer"])
            cont = M.contains_answer(pred, row["answer"])
            score = f1 if row["scorer"] == "f1" else em
            per_sample.append(dict(id=row["id"], q=row["question"], gold=row["answer"],
                                   pred=pred, em=em, f1=f1, contains=cont, score=score,
                                   qtype=row["qtype"], scorer=row["scorer"]))
        m = dict(
            exact_match=sum(p["em"] for p in per_sample) / len(per_sample),
            token_f1=sum(p["f1"] for p in per_sample) / len(per_sample),
            contains=sum(p["contains"] for p in per_sample) / len(per_sample),
            overall=sum(p["score"] for p in per_sample) / len(per_sample),
        )
    elif task == "jsonout":
        for row in rows:
            pred = generate_one(model, tok, build_prompt(tok, row, task), cfg, cfg.max_new_tokens)
            obj = M.safe_json_loads(pred)
            parsed = obj is not None
            valid, errs = M.schema_valid(obj, row["schema"]) if parsed else (False, ["parse_fail"])
            facc = M.field_accuracy(obj, row["gold"], row["schema"]["fields"].keys()) if parsed else 0.0
            per_sample.append(dict(id=row["id"], instr=row["instruction"],
                                   gold=jsonout_mod.gold_json(row), pred=pred,
                                   parsed=parsed, valid=valid, errors=errs, field_acc=facc))
        m = dict(
            json_parse_rate=sum(p["parsed"] for p in per_sample) / len(per_sample),
            schema_valid_rate=sum(p["valid"] for p in per_sample) / len(per_sample),
            field_accuracy=sum(p["field_acc"] for p in per_sample) / len(per_sample),
            overall=sum(p["valid"] for p in per_sample) / len(per_sample),
        )
    elif task == "intent":
        preds, golds = [], []
        for row in rows:
            raw = generate_one(model, tok, build_prompt(tok, row, task), cfg, 12)
            lab = _norm_intent(raw)
            preds.append(lab)
            golds.append(row["label"])
            per_sample.append(dict(id=row["id"], text=row["text"], gold=row["label"],
                                   raw=raw, pred=lab, correct=(lab == row["label"])))
        m = dict(
            accuracy=M.accuracy(preds, golds),
            macro_f1=M.macro_f1(preds, golds),
            unmapped_rate=sum(1 for p in preds if p is None) / len(preds),
            overall=M.accuracy(preds, golds),
        )
    else:
        raise ValueError(task)

    wall = time.time() - t0
    peak_mem_after = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    m["n"] = len(per_sample)
    m["wall_sec"] = round(wall, 2)
    m["sec_per_sample"] = round(wall / max(1, len(per_sample)), 4)
    if torch.cuda.is_available():
        m["peak_vram_delta_mb"] = round((peak_mem_after - peak_mem_before) / 1024**2, 1)
    return m, per_sample
