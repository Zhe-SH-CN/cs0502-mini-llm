# 截图清单（报告占位 · 由 Zhe 手动截取）

> 约定：截图存 `docs/screenshots/S-XX.png`，报告以 `![S-XX](screenshots/S-XX.png)` 引用。
> 每张截图请在文件名后保留编号，便于报告对齐。

| 编号 | 内容 | 在哪里截 | 用途章节 |
|---|---|---|---|
| S-01 | CNB 仓库主页（cs0502-mini-llm） | cnb.cool/sizhezhang/cs0502-mini-llm | 项目结果展示 |
| S-02 | CNB CI 构建列表（含成功/失败） | 仓库→构建 | 过程证据 |
| S-03 | CNB CI 某次构建日志（数据管线跑批） | 构建详情→日志 | 过程证据/AI使用 |
| S-04 | CNB CI 单元测试通过输出 | 构建日志 pytest 段 | 问题与解决 |
| S-05 | Colab 运行时类型对话框（T4 选中） | 运行时→更改运行时类型 | 资源优化 |
| S-06 | Colab GPU 自检输出（T4/16GB/CC7.5） | notebook 00 cell[0] | 资源优化 |
| S-07 | notebook 00 数据集统计输出 | notebook 00 cell[4] | 数据与训练实验 |
| S-08 | notebook 01 baseline 评测表 | notebook 01 | 基本3 前后对比 |
| S-09 | notebook 01 SFT 训练 loss 曲线/日志 | notebook 01 训练 cell | 基本2 |
| S-10 | notebook 01 ALL RUNS 汇总表 | notebook 01 末尾 | 基本3 |
| S-11 | notebook 02 Qwen LoRA/QLoRA 结果表 | notebook 02 | 模型尺寸提升 |
| S-12 | notebook 03 量化基准表（显存/吞吐） | notebook 03 | 扩展1 |
| S-13 | notebook 03 量化质量对比表 | notebook 03 | 扩展1 |
| S-14 | notebook 04 数据规模消融表 | notebook 04 | 扩展2 |
| S-15 | notebook 04 数据质量消融表 | notebook 04 | 扩展2 |
| S-16 | notebook 04 LoRA rank 消融表 | notebook 04 | 扩展2 |
| S-17 | notebook 04 成功/失败样例输出 | notebook 04 末尾 | 扩展2/反思 |
| S-18 | Drive 中 results/ 目录文件列表 | Google Drive | 过程证据 |
| S-19 | CNB GPU 启动失败报错（P-01 证据） | 构建日志 Prepare 段 | 问题与解决 |
| S-20 | 额度查询输出（GPU=0） | 终端 cnb charge | 问题与解决 |

## 已自动留存的机器证据（无需截图）
- `results/runs.jsonl`、`results/summary.csv`：全部 run 的指标
- `docs/pitfalls.md`：踩坑原文（含 CNB 报错原文）
