# 交接报告（HANDOFF）

> 面向接手 agent / 组员。读完本文件 + `docs/pitfalls.md` 即可无缝继续。
> 更新时间：2026-10-31（本地时间约 17:00）。

## 0. 一句话现状
代码与 5 个 Colab notebook 已全部就绪并推送 GitHub；**notebook 00 已由用户手动跑通**（T4 实测 + 数据已落 Drive）；
notebook 01 刚打开、运行时连接中（未开始跑）；02–04 未跑；报告/PPT 未写。

## 1. 项目与作业对齐
- 课程：上海交大《计算机科学导论》CS0502，小组 5 人，选题五「Mini-LLM 小模型部署与微调」。
- 作业要求 → 交付映射：
  | 要求 | 交付 | 状态 |
  |---|---|---|
  | 基本1 部署可运行小模型 | notebook 00/01（MiniMind2-Small/2 加载+推理+评测） | 代码就绪，00 已跑通 |
  | 基本2 自选任务+数据+微调 | 三任务 + SFT/LoRA/QLoRA/DPO | 代码就绪 |
  | 基本3 独立测试集前后对比 | baseline vs finetuned 同 test 集，Recorder 落盘 | 代码就绪 |
  | 扩展1 量化部署对比 | notebook 03（fp16/int8/int4/GGUF × 显存/吞吐/延迟/质量） | 代码就绪 |
  | 扩展2 数据/训练实验 | notebook 04（规模×质量×LoRA rank + 成功失败样例） | 代码就绪 |
- 评分重点（PDF 13 页）：AI 使用过程、判断与纠错、协作、过程证据与反思 —— **不是作品华丽度**。
  故 `docs/pitfalls.md` 与交互记录是核心得分材料，务必持续积累。

## 2. 仓库与链接
- **主仓库（唯一事实源）**：https://github.com/Zhe-SH-CN/cs0502-mini-llm （branch main）
- CNB 仓库 https://cnb.cool/sizhezhang/cs0502-mini-llm ：**已弃用**（CNB GPU 额度为 0，见 P-01），仅留作历史；不要再向其推送或配置 CI。
- 本地工作区：`/Users/zhe/Documents/1SJTU/计算机科学导论/cs0502-mini-llm/`
- Colab notebook 直链（GitHub 托管，Colab 可直接打开）：
  `https://colab.research.google.com/github/Zhe-SH-CN/cs0502-mini-llm/blob/main/notebooks/0X_*.ipynb`

## 3. 目录结构
```
src/minillm/          核心包
  config.py           ExpConfig + MODELS 注册表 + T4 钉死参数(T4_DTYPE/T4_ATTN)
  data.py             jsonl 读写 / build_all_splits(分层切分) / subsample / inject_label_noise / to_sft_messages
  metrics.py          纯函数指标：normalize_zh/exact_match/token_f1/contains/safe_json_loads/schema_valid/field_accuracy/accuracy/macro_f1
  tasks/              courseqa.py(KB+模板) jsonout.py(schema+槽位) intent.py(6类模板)
  evaluate.py         load_model(T4适配) / build_prompt / generate_one / evaluate(三任务)
  train.py            SFT/LoRA/QLoRA/DPO；trl 0.9–0.15 兼容(SFTConfig/DPOConfig + 签名探测 processing_class)
  quantize.py         fp16/bnb int8/int4/GGUF 导出/GPTQ/AWQ(可选)
  bench.py            吞吐/首token延迟/峰值显存
  record.py           Recorder：runs.jsonl + <run>.json + summary.csv（torch 懒加载）
notebooks/            00 环境数据 / 01 minimind / 02 qwen / 03 量化 / 04 消融（由 scripts/make_notebooks.py 生成）
scripts/              build_data.py / run_eval.py / make_notebooks.py
tests/                16 项纯 Python 单测（本地无 pytest 时用内联 runner 跑过，全绿）
docs/                 plan.md / DATACARD.md / pitfalls.md / screenshots.md / report.md(待写) / HANDOFF.md(本文件)
```

## 4. 环境实测事实（重要，勿改）
- **GPU**：Colab 免费档 = **Tesla T4，14.6 GiB 可用，CC (7,5)**（notebook 00 cell[0] 实测输出）。
  可选还有 v5e-1 TPU；H100/A100/L4/G4 需 Pro+。
- **T4 硬约束（全项目钉死）**：`dtype=float16`（sm7.5 无原生 bf16）；`attn=sdpa`（不支持 FlashAttention-2）。
  见 `config.py` 的 `T4_DTYPE/T4_ATTN`。
- **Colab 预装 torch = 2.11.0+cu130（CUDA 13.0）**。不要钉 torch（P-11）。
- **依赖矩阵（notebook [1] 钉死）**：`transformers==4.57.3` + `trl==0.15.0`，其余 peft/accelerate/bitsandbytes/datasets 不钉。
  该组合实测安装成功（notebook 00 cell[5] 输出 `transformers 4.57.3 | trl 0.15.0`）。
- **包引入方式**：不做 `pip install -e .`（Colab PEP660 editable 坑，P-12）；
  notebook [2] 用 `os.chdir("/content")` + clone + `sys.path.insert(0, <repo>/src)`（P-13 防嵌套）。
- **落盘**：所有 data/results 写 `Drive/MyDrive/cs0502-mini-llm/{data,results}`（P-05）。
  Colab 本地盘易失，断运行时即丢 —— 任何新 cell 都必须经 Recorder 落 Drive。
- **pip 告警**：gradio/diffusers 与 hub 0.36.2 冲突为良性（P-15），忽略。

## 5. 数据与指标（口径）
- 三任务：courseqa（课程问答）/ jsonout（结构化 JSON）/ intent（意图分类 6 类）。
- 规模（seed=42，`build_all_splits`）：courseqa 104/34/23→实为 104/34/32？以 Drive 中 jsonl 行数为准；
  jsonout 384/128/128；intent 576/192/192。（courseqa 按 topic 分层，id 不跨 split，单测校验）
- 指标：
  - courseqa：exact_match / token_f1(字符级) / contains / overall(按 scorer 取 em 或 f1)
  - jsonout：json_parse_rate / schema_valid_rate / field_accuracy / overall=valid
  - intent：accuracy / macro_f1 / unmapped_rate / overall=accuracy
- 评测协议：独立 test 集；baseline 与 finetuned 用同一 test、同一 generate 参数（do_sample=False）；
  每次 run 由 Recorder 记录 config+metrics+per_sample+env。

## 6. 模型阶梯
| key | HF id | 参数量 | T4 训练方式 |
|---|---|---|---|
| minimind-26m | jingyaogong/MiniMind2-Small | 26M | sft/lora |
| minimind-104m | jingyaogong/MiniMind2 | 104M | sft/lora |
| qwen2.5-0.5b | Qwen/Qwen2.5-0.5B-Instruct | 0.5B | sft/lora/qlora |
| qwen2.5-1.5b | Qwen/Qwen2.5-1.5B-Instruct | 1.5B | lora/qlora |
（HF id 均经 hf-mirror API 200 校验；Colab 直连 HF 无墙）

## 7. 踩坑清单索引（详见 docs/pitfalls.md）
P-01 CNB GPU 额度 0 → 弃 CNB 用 Colab｜P-02 特权不含 GPU 且到期｜P-03 T4 无 bf16/FA2｜
P-04 transformers5 与 trl0 不兼容｜P-05 Colab 盘易失必须落 Drive｜P-06 `def` 关键字坑｜
P-07 内置浏览器 vs Browser Connector 两套 surface｜P-08 Colab 对话框在 shadow DOM，用 AX 索引操作｜
P-09 本机 M2/8GB 不跑 GPU｜P-10 CNB CI 无 python3（已作废）｜P-11 不钉 torch｜
P-12 editable 安装 import 不到 → sys.path｜P-13 CWD 漂移嵌套克隆 → chdir /content｜
P-14 Drive 授权范围选错 MessageError → 重启运行时重授（用户手动）｜P-15 hub 告警良性

## 8. Colab 自动化操作要点（给用浏览器的 agent）
- 用户 Chrome 已装 Qoder Browser Connector（v1.7.6），登录账号 zhangsizhe.jincai@gmail.com（Google One 会员）。
- 接管流程：`createBrowserAgent()` → `browsers.list()` → `user.openTabs()` → `claimTab()`；
  或 `browser.tabs.new()` + `goto(...)`。
- **Colab 缓存副本坑**：同一标签页 reload 可能恢复到内存旧副本（含旧 cell 代码）。
  要拿最新 GitHub 代码必须**新开标签页**加载；加载后先校验 cell 文本含预期新代码再运行。
- 对话框（运行时类型/信任警告）不在主 frame light DOM：Playwright role 定位失效；
  用 **AX 索引**：`tab.ax.get('state')` 找 `button <label>` 的行首索引 → `tab.ax.click(idx)`；
  键盘用 `tab.ax.pressKey('Escape')`。信任警告按钮 label=「仍然运行」。
- 运行全部：AX 找「全部运行」按钮索引点击；首个错误会 halt 后续 cell（run-all 默认遇错停）。
- 输出读取：AX state 含 outputframe 文本（可匹配如 `"train": 104`、`Mounted at /content/drive`）；
  失败判定：AX 含「之前的执行操作失败了」。截图用 `tab.screenshot()` + `nodeRepl.emitImage`。
- **Drive OAuth 授权属安全决策：弹窗出现时交用户手动点选，agent 不代点。**
- 运行时复用：跨标签页/ reload 复用同一内核 → CWD 会漂移（P-13）；需要干净环境时
  「代码执行程序 → 断开连接并删除运行时」。

## 9. 下一步精确步骤（按序）
1. **notebook 01**（minimind 26M/104M × courseqa/jsonout/intent × baseline/sft/lora）：
   新开标签页打开 → 连接 T4 → 信任「仍然运行」→ 全部运行 → 等 ALL RUNS 表 → 截图 S-08/S-09/S-10。
   预计 15–25 min。若 OOM：降 batch_size/grad_accum 或 max_seq_len。
2. **notebook 02**（Qwen 0.5B lora/qlora + 1.5B qlora）：同上，截图 S-11。预计 30–45 min。
3. **notebook 03**（量化矩阵 + 质量对比）：截图 S-12/S-13。预计 15–20 min。
4. **notebook 04**（消融 + 成功失败样例）：截图 S-14..S-17。预计 30–40 min。
5. 每个 notebook 跑完：把 Drive `results/summary.csv` 与关键表截图归档；
   将数字填入 `docs/report.md`（模板见下）。
6. 写 `docs/report.md`（技术报告）+ PPT 大纲 + 演示脚本；截图按 `docs/screenshots.md` 编号由用户补。
7. 持续把新坑追加到 `docs/pitfalls.md`（编号续 P-16…）。

## 10. 报告骨架（docs/report.md 待填）
1 任务与起点｜2 环境与方法（T4 约束/依赖矩阵/数据卡）｜3 实验结果（每任务 baseline vs 微调表 + 量化表 + 消融表）｜
4 成功/失败样例分析｜5 AI 使用过程（代表性提示/输出/迭代）｜6 问题与解决（引用 P-01..P-15）｜
7 分工与协作｜8 总结与反思（含未完成项与原因）｜附录 截图索引 S-01..S-20

## 11. 约定与红线
- 不钉 torch；不装 flash-attn；1.5B+ 只走 LoRA/QLoRA。
- 一切结果必须经 Recorder 落 Drive；禁止只 print 不落盘。
- 不向 CNB 推送/配置；不动 Colab 基础环境（不卸载 gradio 等）。
- Drive/Google 授权、任何凭据输入：一律用户手动。
- 提交信息用中文短句 + 坑编号引用；推送前跑 `python3 scripts/make_notebooks.py` 保证 notebook 与生成器一致。
