# 项目计划与实验矩阵

## 目标对齐（对照作业 PDF 选题五）
| 作业要求 | 对应交付 | 状态 |
|---|---|---|
| 基本1 部署可运行小模型 | notebook 00/01：MiniMind2-Small/2 加载+推理+评测 | 代码就绪 |
| 基本2 自选任务+数据+微调 | 三任务 + SFT/LoRA/QLoRA/DPO | 代码就绪 |
| 基本3 独立测试集前后对比 | baseline vs finetuned 同 test 集，Recorder 落盘 | 代码就绪 |
| 扩展1 量化部署对比 | notebook 03：fp16/int8/int4/GGUF × 显存/吞吐/延迟/质量 | 代码就绪 |
| 扩展2 数据/训练实验 | notebook 04：规模×质量×LoRA rank + 成功失败样例 | 代码就绪 |

## 模型阶梯（脚踏实地）
1. MiniMind2-Small (26M) — 全流程验证
2. MiniMind2 (104M) — 尺寸提升一档
3. Qwen2.5-0.5B-Instruct — 进入主流小模型，可全参 SFT
4. Qwen2.5-1.5B-Instruct — 仅 LoRA/QLoRA（T4 16GB 约束）

## 实验矩阵（Colab T4）
| # | notebook | 内容 | 预计 GPU 时长 |
|---|---|---|---|
| 00 | env_and_data | 环境自检+数据构建 | ~2min |
| 01 | minimind_sft_eval | 26M/104M × 3任务 × {base,sft,lora} | ~25min |
| 02 | qwen_lora_qlora | 0.5B {lora,qlora} + 1.5B qlora | ~40min |
| 03 | quant_bench | 量化矩阵+吞吐显存+质量 | ~20min |
| 04 | ablations | 规模×质量×rank + 样例分析 | ~35min |

## 执行顺序与依赖
00 → 01 → 02 → 03 → 04（03/04 依赖 00 的数据；02 依赖 01 验证流程无误）

## 风险与回退
- T4 抢不到 → 回退 v5e-1 TPU（需改 dtype/策略，记录为新坑）
- 1.5B QLoRA OOM → 降 batch/grad_accum，或只保留 0.5B
- HF 下载慢 → 换 hf-mirror 或 ModelScope（Qwen 有镜像）
