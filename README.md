# CS0502 Mini-LLM · 小模型部署与微调

上海交通大学《计算机科学导论》小组项目 · 选题五。
部署并微调小型语言模型（MiniMind → Qwen2.5），在三个自选任务上量化微调前后效果，
并做量化部署基准与数据/训练消融。

- 仓库：https://cnb.cool/sizhezhang/cs0502-mini-llm
- 算力：Google Colab **T4 (16GB, sm7.5)**（CNB GPU 额度为 0，见 `docs/pitfalls.md` P-01）
- 代码托管 + 免费 CPU CI：CNB

## 三个自选任务
| 任务 | 说明 | 指标 |
|---|---|---|
| courseqa | 课程问答（闭卷知识注入） | EM / 字符F1 / contains |
| jsonout | 结构化 JSON 输出（schema 约束） | parse率 / schema率 / 字段acc |
| intent | 意图分类（6 类） | acc / macro-F1 |

## 模型阶梯
MiniMind2-Small(26M) → MiniMind2(104M) → Qwen2.5-0.5B → Qwen2.5-1.5B(仅 LoRA/QLoRA)

## 目录
```
src/minillm/        核心包：config/data/metrics/tasks/evaluate/train/quantize/bench/record
notebooks/          Colab notebooks（00 环境数据 / 01 minimind / 02 qwen / 03 量化 / 04 消融）
scripts/            build_data.py / run_eval.py / make_notebooks.py
tests/              纯 Python 单测（16 项，CI 跑）
docs/               plan / DATACARD / pitfalls / screenshots / report
data/, results/     由脚本生成（gitignore）
```

## 运行（Colab）
1. 上传/打开 `notebooks/00_env_and_data.ipynb`，运行时选 **T4 GPU**
2. 依次运行 00 → 01 → 02 → 03 → 04
3. 结果自动落 `Drive/MyDrive/cs0502-mini-llm/results/`

notebook 自包含：安装钉死版本依赖 → 克隆本仓库 → `pip install -e .` → 挂 Drive。

## 运行（CI / 本地，CPU）
```bash
pip install -e .[dev]
pytest tests -q
python scripts/build_data.py --out data
python scripts/run_eval.py --model minimind-26m --task courseqa --max-samples 6
```

## T4 硬约束（全项目钉死）
- `dtype=float16`（sm7.5 无原生 bf16）
- `attn=sdpa`（sm7.5 不支持 FlashAttention-2）
- 1.5B 及以上仅 LoRA/QLoRA（16GB 显存）

## 文档
- 计划与实验矩阵：`docs/plan.md`
- 数据卡：`docs/DATACARD.md`
- 踩坑记录：`docs/pitfalls.md`
- 截图清单：`docs/screenshots.md`
- 技术报告：`docs/report.md`（实验跑完后填充）
