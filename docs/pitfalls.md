# 踩坑记录（持续更新）

> 原则：每条记录「现象 → 根因 → 解法 → 证据」，供报告「问题与解决」章节直接引用。

## P-01 CNB GPU 额度为 0，L40 开发环境启动即失败
- **现象**：`cnb workspace start-workspace` 受理成功，但 Prepare 阶段报错：
  `根组织的云原生开发-GPU配额已不够预冻结(冻结时间：10.00 min，折合2.67核时)`
- **根因**：CNB 社区版 GPU 免费额度为 **0 核时/月**（构建与开发均为 0），按量 ¥0.5/核时；
  L40 节点 CPU 固定 16 核 → 固定 ¥8/小时，无法调低核数省钱。8 月同配置曾成功，说明是额度耗尽而非配置错误。
- **证据**：`cnb charge get-quota --slug sizhezhang` 返回 `ci_gpu_in_sec.total=0`、`dev_gpu_in_sec.total=0`；
  官方定价文档明确 GPU 免费额度 0。
- **解法**：放弃 CNB GPU，改走 Google Colab T4（免费）。CNB 仅作为代码托管 + 免费 CPU CI。

## P-02 「天才程序节」特权不含 GPU 且当天到期
- **现象**：特权额度里 `compute_build_gpu_corehour=null`、`compute_develop_gpu_corehour=null`。
- **根因**：该特权只给 CPU 核时（构建/开发各 1024），GPU 字段为空；且 expire=2026-10-31T15:59:59Z。
- **解法**：不依赖该特权做 GPU 规划。

## P-03 T4 (sm7.5) 无原生 bf16、不支持 FlashAttention-2
- **现象（预期）**：在 T4 上用 `torch_dtype=bfloat16` 或 `attn_implementation="flash_attention_2"` 会报错/极慢。
- **根因**：Turing 架构 CC=7.5；bf16 原生支持从 Ampere(8.0) 开始；FA2 kernel 要求 sm80+。
- **解法**：全项目钉死 `dtype=float16`、`attn=sdpa`（见 `minillm/config.py` 的 `T4_DTYPE/T4_ATTN`）。

## P-04 transformers 5.x 与 trl 0.x 不兼容（继承自 llm-env-image 的实测结论）
- **现象**：`trl.trainer.dpo_trainer` ImportError。
- **根因**：transformers 5.x 移除 `MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES` 等 API；带 PPOTrainer 的 trl 0.x 是 4.x 时代代码。
- **解法**：钉死 `transformers==4.44.2` + `trl==0.9.6`（notebook PIP_CELL）。

## P-05 Colab 文件系统易失，结果必须落 Drive
- **现象**：运行时断开/回收后 /content 下文件丢失。
- **解法**：所有 notebook 第 [3] 步挂载 Drive，`Recorder` 统一写 `Drive/MyDrive/cs0502-mini-llm/results/`。

## P-06 Python 关键字 `def` 不能作 dict key
- **现象**：`SyntaxError: invalid syntax`（courseqa.py KB 条目）。
- **根因**：知识库字段命名用了 `def=`。
- **解法**：改名 `desc`；单测立即捕获（16 passed）。

## P-07 Qoder Browser Connector 与「内置浏览器」是两套 surface
- **现象**：MCP 浏览器工具打开的 Colab 显示未登录；用户称其 Chrome 已登录。
- **根因**：MCP browser-use 工具默认驱动 Qoder 内置浏览器（独立干净 profile）；用户真实 Chrome 需安装并连接 Qoder Browser Connector 扩展后才可被 Agent 接管。
- **解法**：安装扩展并连接后，用 `agent.browsers.list()` → `user.openTabs()` → `claimTab()` 接管真实标签页；已验证可读到登录态（zhangsizhe.jincai@gmail.com）。

## P-08 Colab 运行时对话框在 shadow DOM / 独立层，Playwright 主 frame 定位失效
- **现象**：`getByRole('radio', {name:'T4 GPU'})` count=0，但 AX 树能看到该 radio（索引 459）。
- **根因**：对话框不在主 frame 的 light DOM 中；AX 树跨层可见。
- **解法**：改用 AX 索引动作（`tab.ax.click(idx)` / `tab.ax.pressKey`）操作；或由用户手动选择运行时。

## P-09 本机（M2 / 8GB）无法承担 GPU 实验
- **现象**：无 CUDA；bitsandbytes CUDA 量化不可用；8GB 内存跑 1.5B 吃力。
- **解法**：按用户决定，全部 GPU 实验在 Colab T4 执行；本机仅做纯 Python 单测与文件生成。
