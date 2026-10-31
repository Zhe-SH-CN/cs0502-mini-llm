"""量化：产出多种精度变体并报告体积/显存，供基准与质量对比。

T4 可用：bnb int8 / bnb NF4 int4 / GGUF(llama.cpp) / GPTQ / AWQ（后两者需校准，可选）。
不可用时返回 None 并记录原因，保证实验矩阵不因单点失败而中断。
"""
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def model_size_mb(path) -> Optional[float]:
    p = Path(path)
    if not p.exists():
        return None
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return round(total / 1024**2, 1)


def load_fp16(hf_id: str, attn: str = "sdpa"):
    m = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float16,
                                             attn_implementation=attn)
    t = AutoTokenizer.from_pretrained(hf_id)
    m.eval()
    return m, t


def load_bnb(hf_id: str, bits: int = 8, attn: str = "sdpa"):
    if bits == 8:
        qcfg = BitsAndBytesConfig(load_in_8bit=True)
    elif bits == 4:
        qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.float16,
                                  bnb_4bit_use_double_quant=True)
    else:
        raise ValueError(bits)
    m = AutoModelForCausalLM.from_pretrained(hf_id, quantization_config=qcfg,
                                             attn_implementation=attn)
    t = AutoTokenizer.from_pretrained(hf_id)
    m.eval()
    return m, t


def export_gguf(hf_id: str, out_dir, quant: str = "q4_k_m") -> Optional[Path]:
    """用 llama.cpp 的 convert 脚本导出 GGUF（需已安装 llama-cpp-python 或 llama.cpp）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"model-{quant}.gguf"
    if target.exists():
        return target
    converter = shutil.which("convert_hf_to_gguf.py")
    if converter is None:
        return None
    subprocess.run(["python", converter, hf_id, "--outtype", "f16",
                    "--outfile", str(out_dir / "model-f16.gguf")], check=True)
    quantize = shutil.which("llama-quantize")
    if quantize is None:
        return out_dir / "model-f16.gguf"
    subprocess.run([quantize, str(out_dir / "model-f16.gguf"), str(target), quant], check=True)
    return target


def try_gptq(hf_id: str, calib_texts, bits: int = 4):
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    except Exception as e:
        return None, f"auto-gptq unavailable: {e}"
    qcfg = BaseQuantizeConfig(bits=bits, group_size=128, desc_act=False)
    model = AutoGPTQForCausalLM.from_pretrained(hf_id, qcfg)
    tok = AutoTokenizer.from_pretrained(hf_id)
    examples = [{"input_ids": tok(t, return_tensors="pt").input_ids[0]} for t in calib_texts]
    model.quantize(examples)
    return model, None


def try_awq(hf_id: str, calib_texts, bits: int = 4):
    try:
        from awq import AutoAWQForCausalLM
    except Exception as e:
        return None, f"autoawq unavailable: {e}"
    model = AutoAWQForCausalLM.from_pretrained(hf_id)
    tok = AutoTokenizer.from_pretrained(hf_id)
    model.quantize(tok, calib_data=calib_texts, quant_config={
        "zero_point": True, "q_group_size": 128, "w_bit": bits, "version": "GEMM"})
    return model, None
