# 数据卡（DATACARD）

## 总览
| 任务 | train | dev | test | 标签来源 | 指标 |
|---|---|---|---|---|---|
| courseqa 课程问答 | 104 | 34 | 32 | 人工编写 KB | EM / 字符F1 / contains |
| jsonout 结构化输出 | 384 | 128 | 128 | 模板+槽位确定性生成 | parse率 / schema率 / 字段acc |
| intent 意图分类 | 576 | 192 | 192 | 模板+槽位+口语扰动 | acc / macro-F1 |

## 来源与生成方式
- **courseqa**：作者依据《CS0502 八个项目选题》PDF 及课程关联知识点（计算机网络、密码与安全、
  强化学习、数据结构、数据库/数据挖掘、大模型/Transformer、计算理论/信息论/编码）**手工编写**
  35 条概念知识库（`src/minillm/tasks/courseqa.py` 的 `KB`），每条含 `desc`（一句话解释）、
  `short`（极简短答）、`attrs`（附加短问答）。再由 4 类问法模板确定性生成 QA 对。
  **不含任何模型生成内容**，标签 100% 人工可信。
- **jsonout**：4 个 schema（实验结果/作业提交/二手商品/RL 配置），槽位值来自有限池，
  指令由模板把 gold 值渲染成自然语言。标签即 gold JSON，天然正确。
- **intent**：6 类意图 × 6 模板 × 16 概念槽位 + 口语前缀/标点扰动。标签由构造决定。

## 切分协议
- courseqa 按 **topic 分层**切分（60/20/20），避免同 topic 集中泄漏；并保证 id 不跨 split。
- 其余任务随机 60/20/20，seed=42 固定，完全可复现。
- 单测 `test_build_all_splits_no_topic_leak` 校验 id 不跨 split。

## 局限（诚实声明）
1. 数据为**模板/KB 驱动**，句式多样性低于真实用户语料；模型可能学到模板而非泛化语义。
   缓解：intent 加口语扰动；courseqa 用多问法；jsonout 槽位组合空间 >1000。
2. courseqa 规模小（train 104），对 Qwen 级别模型偏少；其定位是**验证微调流程与前后对比**，
   而非追求 SOTA。规模影响由消融实验（notebook 04）显式量化。
3. 未做人工双盲标注一致性检验（数据为构造标签，无标注分歧）。

## 复现
```bash
python scripts/build_data.py --out data --seed 42
```
