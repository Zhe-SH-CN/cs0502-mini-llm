"""生成 Colab notebooks（自包含、T4 钉死、结果落 Drive）。

用法：python scripts/make_notebooks.py
产出：notebooks/00..04 *.ipynb
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks"
OUT.mkdir(exist_ok=True)

REPO_URL = "https://github.com/Zhe-SH-CN/cs0502-mini-llm.git"

PIP_CELL = """# [1] 安装依赖（T4 / CUDA12 兼容钉死版本）
# 关键：transformers 必须 <5（trl 0.x 与 5.x 不兼容）；不装 flash-attn（T4 sm7.5 不支持）
!pip install -q "torch==2.3.1" "transformers==4.44.2" "datasets==2.20.0" "peft==0.12.0" \\
    "trl==0.9.6" "accelerate==0.33.0" "bitsandbytes==0.43.3" "sentencepiece" "tiktoken" \\
    "numpy<2" "pandas" "matplotlib"
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda)
"""

GPU_CELL = """# [0] GPU 自检：确认拿到 T4，并钉死 T4 兼容参数
import torch
assert torch.cuda.is_available(), "未分配到 GPU！请 运行时->更改运行时类型 选 T4 GPU"
name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU: {name} | VRAM: {vram:.1f} GB | CC: {torch.cuda.get_device_capability(0)}")
# T4 = sm7.5：无原生 bf16、无 FlashAttention-2 -> 统一 fp16 + sdpa
DTYPE, ATTN = "float16", "sdpa"
print("钉死参数: dtype=%s attn=%s (T4 无 bf16/FA2)" % (DTYPE, ATTN))
"""

CLONE_CELL = f"""# [2] 克隆仓库并安装 minillm 包（单一事实源）
import os, subprocess
if not os.path.exists("cs0502-mini-llm"):
    subprocess.run(["git", "clone", "--depth", "1", "{REPO_URL}"], check=True)
os.chdir("cs0502-mini-llm")
!pip install -q -e .
import minillm
from minillm.config import ExpConfig
from minillm import data as D
from minillm import evaluate as E
from minillm import train as T
from minillm import quantize as Q
from minillm import bench as B
from minillm.record import Recorder
print("minillm", minillm.__version__, "| cwd", os.getcwd())
"""

DRIVE_CELL = """# [3] 挂载 Google Drive，统一结果/数据落盘路径（Colab 文件系统易失，必须落 Drive）
from google.colab import drive
drive.mount("/content/drive")
import os
BASE = "/content/drive/MyDrive/cs0502-mini-llm"
DATA = os.path.join(BASE, "data"); RESULTS = os.path.join(BASE, "results")
os.makedirs(DATA, exist_ok=True); os.makedirs(RESULTS, exist_ok=True)
rec = Recorder(RESULTS)
print("DATA:", DATA, "| RESULTS:", RESULTS)
"""

BUILD_DATA_CELL = """# [4] 构建三任务数据集（确定性，seed 固定）并做分层切分（courseqa 按 topic 分层防泄漏）
import json
stats = D.build_all_splits(DATA, seed=42)
print(json.dumps(stats, ensure_ascii=False, indent=1))
"""

TABLE_HELPER = """# 结果打表辅助
import pandas as pd
def show(df, title=""):
    if title: print("\\n==", title, "==")
    from IPython.display import display
    display(df)
    return df
"""


def nb(cells):
    out = []
    for kind, src in cells:
        out.append({"cell_type": kind,
                    "metadata": {},
                    "source": src if isinstance(src, list) else src.splitlines(keepends=True),
                    **({"outputs": [], "execution_count": None} if kind == "code" else {})})
    return {"nbformat": 4, "nbformat_minor": 0,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                         "language_info": {"name": "python"},
                         "colab": {"provenance": [], "gpuType": "T4"},
                         "accelerator": "GPU"},
            "cells": out}


def md(s):
    return ("markdown", s)


def code(s):
    return ("code", s)


# ---------------- 00 环境与数据 ----------------
n00 = [
    md("# 00 · 环境自检与数据构建\n\n**目标**：确认 T4 GPU、钉死兼容参数、构建三任务数据集并落 Drive。\n\n**硬件**：T4 (16GB, sm7.5)。本 notebook 不训练模型。"),
    code(GPU_CELL), code(PIP_CELL), code(CLONE_CELL), code(DRIVE_CELL),
    code(BUILD_DATA_CELL),
    code("""# [5] 抽样检查数据质量（人工 sanity check）
from minillm import data as D
import os
for task in ("courseqa", "jsonout", "intent"):
    ds = D.load_dataset(task, DATA)
    print("\\n###", task, ds.sizes())
    for r in ds.split("test")[:3]:
        print(" ", json.dumps(r, ensure_ascii=False)[:220])
"""),
    md("## 完成\n\n数据已落 `Drive/cs0502-mini-llm/data/`。下一步运行 `01_minimind_sft_eval.ipynb`。"),
]

# ---------------- 01 minimind SFT/LoRA + 评测 ----------------
def exp_cells(model_key, task, methods):
    cells = []
    for method in methods:
        cells.append(code(f"""# 基线（未微调）评测：{model_key} / {task}
cfg = ExpConfig(run_id="{model_key}-{task}-baseline", model_key="{model_key}", task="{task}",
                method="none", dtype=DTYPE, attn=ATTN)
model, tok = E.load_model(cfg)
m_base, ps_base = E.evaluate(model, tok, D.load_dataset("{task}", DATA), "{task}", cfg)
rec.log_run(cfg.run_id, cfg.to_dict(), m_base, ps_base)
del model
import torch as _t; _t.cuda.empty_cache()
show(pd.DataFrame([{{"run": cfg.run_id, **{{k: round(v,4) for k,v in m_base.items() if isinstance(v,(int,float))}}}}]), "BASELINE {model_key}/{task}")
"""))
        cells.append(code(f"""# 微调（{method}）+ 评测：{model_key} / {task}
cfg = ExpConfig(run_id="{model_key}-{task}-{method}", model_key="{model_key}", task="{task}",
                method="{method}", use_4bit={"true" if method == "qlora" else "false"},
                dtype=DTYPE, attn=ATTN, epochs=2, batch_size=4, grad_accum=4, lr=2e-4)
out_dir = os.path.join(RESULTS, "ckpt", cfg.run_id)
T.train(D.load_dataset("{task}", DATA), "{task}", cfg, out_dir)
# 加载微调后模型评测
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base, tok = E.load_model(ExpConfig(run_id="tmp", model_key="{model_key}", dtype=DTYPE, attn=ATTN))
if "{method}" != "sft":
    base = PeftModel.from_pretrained(base, os.path.join(out_dir, "final"))
    base = base.merge_and_unload()
else:
    base = AutoModelForCausalLM.from_pretrained(os.path.join(out_dir, "final"), torch_dtype=_ft(DTYPE), attn_implementation=ATTN)
base.eval()
m_ft, ps_ft = E.evaluate(base, tok, D.load_dataset("{task}", DATA), "{task}", cfg)
rec.log_run(cfg.run_id, cfg.to_dict(), m_ft, ps_ft)
show(pd.DataFrame([{{"run": cfg.run_id, **{{k: round(v,4) for k,v in m_ft.items() if isinstance(v,(int,float))}}}}]), "FINETUNED {method}")
del base; _t.cuda.empty_cache()
"""))
    return cells


n01 = [
    md("# 01 · MiniMind 微调与前后对比（基本要求 1/2/3）\n\n**模型**：MiniMind2-Small(26M) → MiniMind2(104M)\n**任务**：courseqa / jsonout / intent\n**对比**：baseline vs SFT vs LoRA，独立 test 集。"),
    code(GPU_CELL), code(PIP_CELL), code(CLONE_CELL), code(DRIVE_CELL), code(TABLE_HELPER),
    code("import torch as _t\ndef _ft(d): return {'float16':_t.float16,'bfloat16':_t.bfloat16,'float32':_t.float32}[d]\nimport pandas as pd"),
    *exp_cells("minimind-26m", "courseqa", ["sft", "lora"]),
    *exp_cells("minimind-26m", "jsonout", ["sft"]),
    *exp_cells("minimind-26m", "intent", ["sft"]),
    *exp_cells("minimind-104m", "courseqa", ["sft"]),
    code("""# 汇总本 notebook 所有 run
runs = rec.load_runs()
df = pd.DataFrame([{"run": r["run_id"], "model": r["config"]["model_key"], "task": r["config"]["task"],
                    "method": r["config"]["method"],
                    **{k: round(v,4) for k,v in r["metrics"].items() if isinstance(v,(int,float)) and k in ("overall","exact_match","token_f1","json_parse_rate","schema_valid_rate","field_accuracy","accuracy","macro_f1")}}
                   for r in runs])
show(df.sort_values(["task","model","method"]), "ALL RUNS")
"""),
    md("## 截图点\n\n- 本 notebook 最后的 ALL RUNS 表格\n- GPU 自检输出（T4 / 16GB）"),
]

# ---------------- 02 Qwen LoRA/QLoRA ----------------
n02 = [
    md("# 02 · Qwen2.5 升级：LoRA / QLoRA（模型尺寸提升）\n\n**模型**：Qwen2.5-0.5B-Instruct（可全参 SFT / LoRA / QLoRA）、Qwen2.5-1.5B-Instruct（LoRA / QLoRA）\n**显存策略**：0.5B 全参 SFT 可行；1.5B 仅 LoRA/QLoRA。"),
    code(GPU_CELL), code(PIP_CELL), code(CLONE_CELL), code(DRIVE_CELL), code(TABLE_HELPER),
    code("import torch as _t\ndef _ft(d): return {'float16':_t.float16,'bfloat16':_t.bfloat16,'float32':_t.float32}[d]\nimport pandas as pd"),
    *exp_cells("qwen2.5-0.5b", "courseqa", ["lora", "qlora"]),
    *exp_cells("qwen2.5-0.5b", "jsonout", ["lora"]),
    *exp_cells("qwen2.5-0.5b", "intent", ["lora"]),
    *exp_cells("qwen2.5-1.5b", "courseqa", ["qlora"]),
    code("""runs = rec.load_runs()
df = pd.DataFrame([{"run": r["run_id"], "model": r["config"]["model_key"], "task": r["config"]["task"],
                    "method": r["config"]["method"],
                    **{k: round(v,4) for k,v in r["metrics"].items() if isinstance(v,(int,float)) and k in ("overall","exact_match","token_f1","accuracy","macro_f1","schema_valid_rate")}}
                   for r in runs if r["config"]["model_key"].startswith("qwen")])
show(df, "QWEN RUNS")
"""),
]

# ---------------- 03 量化基准 ----------------
n03 = [
    md("# 03 · 量化部署基准（推荐扩展 1：资源优化）\n\n**矩阵**：fp16 / bnb-int8 / bnb-int4(NF4) / GGUF-Q4\n**指标**：模型体积、峰值显存、decode tokens/s、首 token 延迟、质量分（同 test 集）"),
    code(GPU_CELL), code(PIP_CELL), code(CLONE_CELL), code(DRIVE_CELL), code(TABLE_HELPER),
    code("""!pip install -q llama-cpp-python
import pandas as pd, torch as _t
"""),
    code("""# 量化矩阵：体积 + 显存 + 吞吐
rows = []
hf = "Qwen/Qwen2.5-0.5B-Instruct"
prompts = [r["question"] for r in D.load_dataset("courseqa", DATA).split("test")[:16]]
variants = [("fp16", lambda: Q.load_fp16(hf, ATTN)),
            ("int8", lambda: Q.load_bnb(hf, 8, ATTN)),
            ("int4", lambda: Q.load_bnb(hf, 4, ATTN))]
for name, loader in variants:
    try:
        m, t = loader()
        m.to("cuda") if next(m.parameters()).device.type != "cuda" else None
        b = B.bench_throughput(m, t, prompts)
        b["variant"] = name
        b["vram_mb"] = round(_t.cuda.max_memory_allocated()/1024**2,1)
        rows.append(b)
        del m; _t.cuda.empty_cache()
    except Exception as e:
        rows.append({"variant": name, "error": str(e)[:120]})
show(pd.DataFrame(rows), "QUANT BENCH (Qwen2.5-0.5B)")
rec.log_run("quant-bench-0.5b", {"model": hf, "variants": [r["variant"] for r in rows]},
            {"rows": rows}, [])
"""),
    code("""# 量化后质量对比（int4 vs fp16，同 test 集）
res = {}
for name, bits in (("fp16", None), ("int4", 4)):
    if bits is None:
        m, t = Q.load_fp16(hf, ATTN); m.to("cuda")
    else:
        m, t = Q.load_bnb(hf, 4, ATTN)
    cfg = ExpConfig(run_id=f"quant-quality-{name}", model_key="qwen2.5-0.5b", task="courseqa", dtype=DTYPE, attn=ATTN)
    mm, ps = E.evaluate(m, t, D.load_dataset("courseqa", DATA), "courseqa", cfg)
    res[name] = {k: round(v,4) for k,v in mm.items() if k in ("exact_match","token_f1","overall")}
    rec.log_run(cfg.run_id, cfg.to_dict(), mm, ps)
    del m; _t.cuda.empty_cache()
show(pd.DataFrame(res).T, "QUANT QUALITY")
"""),
]

# ---------------- 04 消融 ----------------
n04 = [
    md("# 04 · 数据与训练消融（推荐扩展 2）\n\n**维度**：数据规模(500/2k/全量) × 数据质量(噪声0/0.2/0.5) × LoRA rank(4/8/32)\n**分析**：成功/失败样例归类"),
    code(GPU_CELL), code(PIP_CELL), code(CLONE_CELL), code(DRIVE_CELL), code(TABLE_HELPER),
    code("import torch as _t\ndef _ft(d): return {'float16':_t.float16,'bfloat16':_t.bfloat16,'float32':_t.float32}[d]\nimport pandas as pd"),
    code("""# 消融 A：数据规模
rows = []
for n in (200, 800, None):
    cfg = ExpConfig(run_id=f"abl-size-{n}", model_key="minimind-26m", task="courseqa", method="sft",
                    max_train_samples=n, dtype=DTYPE, attn=ATTN, epochs=2)
    out = os.path.join(RESULTS, "ckpt", cfg.run_id)
    T.train(D.load_dataset("courseqa", DATA), "courseqa", cfg, out)
    base, tok = E.load_model(ExpConfig(run_id="t", model_key="minimind-26m", dtype=DTYPE, attn=ATTN))
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(os.path.join(out,"final"), torch_dtype=_ft(DTYPE), attn_implementation=ATTN).eval()
    m, ps = E.evaluate(base, tok, D.load_dataset("courseqa", DATA), "courseqa", cfg)
    rec.log_run(cfg.run_id, cfg.to_dict(), m, ps)
    rows.append({"n_train": n, **{k: round(v,4) for k,v in m.items() if k in ("exact_match","token_f1","overall")}})
    del base; _t.cuda.empty_cache()
show(pd.DataFrame(rows), "ABLATION: DATA SIZE")
"""),
    code("""# 消融 B：数据质量（标签噪声）
rows = []
for noise in (0.0, 0.2, 0.5):
    cfg = ExpConfig(run_id=f"abl-noise-{noise}", model_key="minimind-26m", task="intent", method="sft",
                    noise_ratio=noise, dtype=DTYPE, attn=ATTN, epochs=2)
    out = os.path.join(RESULTS, "ckpt", cfg.run_id)
    T.train(D.load_dataset("intent", DATA), "intent", cfg, out)
    from transformers import AutoModelForCausalLM
    base, tok = E.load_model(ExpConfig(run_id="t", model_key="minimind-26m", dtype=DTYPE, attn=ATTN))
    base = AutoModelForCausalLM.from_pretrained(os.path.join(out,"final"), torch_dtype=_ft(DTYPE), attn_implementation=ATTN).eval()
    m, ps = E.evaluate(base, tok, D.load_dataset("intent", DATA), "intent", cfg)
    rec.log_run(cfg.run_id, cfg.to_dict(), m, ps)
    rows.append({"noise": noise, **{k: round(v,4) for k,v in m.items() if k in ("accuracy","macro_f1")}})
    del base; _t.cuda.empty_cache()
show(pd.DataFrame(rows), "ABLATION: DATA QUALITY")
"""),
    code("""# 消融 C：LoRA rank
rows = []
for r in (4, 8, 32):
    cfg = ExpConfig(run_id=f"abl-lora-r{r}", model_key="qwen2.5-0.5b", task="courseqa", method="lora",
                    lora_r=r, dtype=DTYPE, attn=ATTN, epochs=2)
    out = os.path.join(RESULTS, "ckpt", cfg.run_id)
    T.train(D.load_dataset("courseqa", DATA), "courseqa", cfg, out)
    from peft import PeftModel
    base, tok = E.load_model(ExpConfig(run_id="t", model_key="qwen2.5-0.5b", dtype=DTYPE, attn=ATTN))
    base = PeftModel.from_pretrained(base, os.path.join(out,"final")).merge_and_unload().eval()
    m, ps = E.evaluate(base, tok, D.load_dataset("courseqa", DATA), "courseqa", cfg)
    rec.log_run(cfg.run_id, cfg.to_dict(), m, ps)
    rows.append({"lora_r": r, **{k: round(v,4) for k,v in m.items() if k in ("exact_match","token_f1","overall")}})
    del base; _t.cuda.empty_cache()
show(pd.DataFrame(rows), "ABLATION: LORA RANK")
"""),
    code("""# 成功/失败样例分析（取最近一次 courseqa run）
runs = [r for r in rec.load_runs() if r["config"]["task"]=="courseqa" and r["config"]["method"]=="sft"]
if runs:
    ps = runs[-1]["per_sample"]
    bad = [p for p in ps if p["score"] < 0.5][:8]
    good = [p for p in ps if p["score"] >= 0.9][:4]
    print("== 失败样例 ==")
    for p in bad: print(f'  Q: {p["q"]}\\n  gold: {p["gold"]}\\n  pred: {p["pred"][:80]}\\n')
    print("== 成功样例 ==")
    for p in good: print(f'  Q: {p["q"]}\\n  pred: {p["pred"][:80]}\\n')
"""),
]

NOTEBOOKS = {
    "00_env_and_data.ipynb": n00,
    "01_minimind_sft_eval.ipynb": n01,
    "02_qwen_lora_qlora.ipynb": n02,
    "03_quant_bench.ipynb": n03,
    "04_ablations.ipynb": n04,
}

for fname, cells in NOTEBOOKS.items():
    (OUT / fname).write_text(json.dumps(nb(cells), ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", fname, len(cells), "cells")
