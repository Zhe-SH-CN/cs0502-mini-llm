"""训练：SFT / LoRA / QLoRA / DPO，统一入口。

版本策略：trl 0.9–0.15 均可（SFTTrainer / DPOTrainer 签名兼容子集）。
T4 适配：fp16 + sdpa + gradient_checkpointing；QLoRA 用 NF4 双重量化。
"""
import json
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset as HFDataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainingArguments)
from trl import DPOTrainer, SFTTrainer

from .config import ExpConfig, MODELS
from .data import Dataset, inject_label_noise, subsample, to_sft_messages

_TARGETS = {
    "all": None,  # peft 自动选全部线性层
    "qv": ["q_proj", "v_proj"],
    "qkvo": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def _load_base(cfg: ExpConfig):
    kwargs = dict(torch_dtype=torch.float16, attn_implementation=cfg.attn)
    if cfg.use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        kwargs.pop("torch_dtype")
    model = AutoModelForCausalLM.from_pretrained(cfg.hf_id, **kwargs)
    tok = AutoTokenizer.from_pretrained(cfg.hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def _maybe_lora(model, cfg: ExpConfig):
    if cfg.method in ("sft",):
        return model
    if cfg.use_4bit:
        model = prepare_model_for_kbit_training(model)
    lc = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=_TARGETS.get(cfg.lora_target),
    )
    return get_peft_model(model, lc)


def _sft_rows(ds: Dataset, task: str, cfg: ExpConfig, split: str = "train") -> list[dict]:
    rows = subsample(ds.split(split), cfg.max_train_samples, seed=cfg.seed)
    rows = inject_label_noise(rows, cfg.noise_ratio, task, seed=cfg.seed)
    out = []
    for row in rows:
        msgs = to_sft_messages(row, task)
        out.append({"messages": msgs})
    return out


def _to_text(ex, tok):
    try:
        return {"text": tok.apply_chat_template(ex["messages"], tokenize=False)}
    except Exception:
        u = ex["messages"][0]["content"]
        a = ex["messages"][-1]["content"]
        return {"text": f"### 用户\n{u}\n### 回答\n{a}"}


def train_sft(ds: Dataset, task: str, cfg: ExpConfig, out_dir, tok=None, model=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if model is None or tok is None:
        model, tok = _load_base(cfg)
    model = _maybe_lora(model, cfg)
    rows = _sft_rows(ds, task, cfg)
    hf = HFDataset.from_list(rows)
    hf = hf.map(lambda ex: _to_text(ex, tok), remove_columns=["messages"])

    args = TrainingArguments(
        output_dir=str(out_dir / "ckpt"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        fp16=True, bf16=False,
        logging_steps=10,
        save_strategy="epoch",
        seed=cfg.seed,
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=hf,
        max_seq_length=cfg.max_seq_len,
        tokenizer=tok,
        packing=False,
    )
    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    tok.save_pretrained(str(out_dir / "final"))
    return str(out_dir / "final")


def train_dpo(ds: Dataset, task: str, cfg: ExpConfig, out_dir, tok=None, model=None):
    """DPO：chosen=正确标签，rejected=被噪声破坏的标签（与数据质量消融同源）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if model is None or tok is None:
        model, tok = _load_base(cfg)
    model = _maybe_lora(model, cfg)
    clean = subsample(ds.split("train"), cfg.max_train_samples, seed=cfg.seed)
    noisy = inject_label_noise(clean, 1.0, task, seed=cfg.seed + 1)
    pairs = []
    for c, n in zip(clean, noisy):
        pairs.append({
            "prompt": to_sft_messages(c, task)[:-1],
            "chosen": to_sft_messages(c, task),
            "rejected": to_sft_messages(n, task),
        })
    hf = HFDataset.from_list(pairs)

    args = TrainingArguments(
        output_dir=str(out_dir / "ckpt"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=max(1, cfg.batch_size // 2),
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=min(cfg.lr, 5e-5),
        fp16=True, bf16=False,
        logging_steps=10, save_strategy="epoch", seed=cfg.seed,
        gradient_checkpointing=True, report_to=[], remove_unused_columns=False,
    )
    trainer = DPOTrainer(model=model, args=args, train_dataset=hf,
                         beta=0.1, max_length=cfg.max_seq_len,
                         max_prompt_length=cfg.max_seq_len // 2, tokenizer=tok)
    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    tok.save_pretrained(str(out_dir / "final"))
    return str(out_dir / "final")


def train(ds: Dataset, task: str, cfg: ExpConfig, out_dir) -> str:
    if cfg.method == "dpo":
        return train_dpo(ds, task, cfg, out_dir)
    return train_sft(ds, task, cfg, out_dir)
