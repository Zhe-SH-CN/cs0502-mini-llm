# 踩坑记录（持续更新）

> 原则：每条记录「现象 → 根因 → 解法 → 证据」，供报告「问题与解决」章节直接引用。

## P-01 CNB GPU 额度为 0，L40 开发环境启动即失败
- **现象**：`cnb workspace start-workspace` 受理成功，但 Prepare 阶段报错：
  `根组织的云原生开发-GPU配额已不够预冻结(冻结时间：10.00 min，折合2.67核时)`
- **根因**：CNB 社区版 GPU 免费额度为 **0 核时/月**（构建与开发均为 0），按量 ¥0.5/核时；
  L40 节点 CPU 固定 16 核 → 固定 ¥8/小时，无法调低核数省钱。8 月同配置曾成功，说明是额度耗尽而非配置错误。
- **证据**：`cnb charge get-quota --slug sizhezhang` 返回 `ci_gpu_in_sec.total=0`、`dev_gpu_in_sec.total=0`；
  官方定价文档明确 GPU 免费额度 0。
- **解法**：放弃 CNB GPU，改走 Google Colab T4（免费）；代码托管同步迁到 GitHub（Colab 克隆更稳）。

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

## P-10 CNB CI 默认构建镜像无 python3（已随 CNB 弃用而作废，留作过程记录）
- **现象**：setup 阶段 `sh: line 1: python3: command not found`，返回码 127。
- **根因**：未声明 `docker.image` 时默认镜像不含 Python。
- **解法**：`.cnb.yml` 显式 `docker: image: python:3.11-slim`。

## P-11 Colab 预装 torch 已升级到 2.11+cu130，钉旧版 torch 直接安装失败
- **现象**：`ERROR: Could not find a version that satisfies the requirement torch==2.3.1
  (from versions: 2.5.0, 2.5.1, ... 2.15.0)`；环境自检打印 `torch 2.11.0+cu130 | cuda 13.0`。
- **根因**：Colab 运行时基镜像的 torch 随平台升级；默认 PyPI 源对当前 CUDA 组合不再提供 2.3.1 wheel。
- **连带影响**：run-all 在首个错误处 halt，导致后续 Drive 挂载 cell 未执行 → 本轮无落盘（印证 P-05 的必要性）。
- **解法**：notebook 不再钉 torch，直接用预装版本；只钉 `transformers==4.57.3 + trl==0.15.0`
  （已验证兼容矩阵）。train.py 改用 `SFTConfig/DPOConfig` + 签名探测 `processing_class/tokenizer`，
  兼容 trl 0.9–0.15。
- **证据**：notebook 00 cell[0] 成功输出 `GPU: Tesla T4 | VRAM: 14.6 GB | CC: (7, 5)`；
  cell[1] 输出上述 ERROR 与 `torch 2.11.0+cu130`。

## P-12 Colab 上 `pip install -e .`(PEP660 editable) 装完仍 import 不到
- **现象**：`Building editable for minillm (pyproject.toml) ... done` 之后
  `ModuleNotFoundError: No module named 'minillm'`。
- **根因**：Colab 内核与 editable 安装的 path hook 在该环境下未生效（构建隔离 + 内核 path 缓存）。
- **解法**：notebook 不做 editable 安装，改为 `sys.path.insert(0, <repo>/src)` 直接注入；
  第三方依赖仍由 pip 安装。简单、确定、无副作用。

## P-13 内核 CWD 漂移导致仓库嵌套克隆（4 层）
- **现象**：cell 输出 `src: /content/cs0502-mini-llm/cs0502-mini-llm/cs0502-mini-llm/cs0502-mini-llm/src`。
- **根因**：旧版 cell 有 `os.chdir("cs0502-mini-llm")`；Colab 运行时跨 reload 复用同一内核，
  CWD 已深入仓库内部，后续 `os.path.exists("cs0502-mini-llm")` 为 False → 在仓库内再克隆，逐层嵌套。
- **解法**：克隆 cell 开头 `os.chdir("/content")` 强制归位，再判断/克隆/注入 sys.path。

## P-14 drive.mount 授权范围选错 → MessageError
- **现象**：`drive.mount("/content/drive")` 抛 `MessageError`（google.colab/_message.py read_reply_from_input）。
- **根因**：OAuth 授权时选择了错误的账号/范围，Colab 前端返回 error 回包。
- **解法**：重启运行时清空本次运行时的 Drive 授权状态，重新 mount 并在弹窗中选择正确账号与「允许」；
  授权属安全决策，交由用户手动完成（Agent 不代点凭据/范围选择）。

## P-15 pip 依赖解析告警：gradio/diffusers 与 huggingface-hub 0.36.2 冲突（良性）
- **现象**：`gradio-client/diffusers/gradio requires huggingface-hub<3.0,>=1.16.0, but you have 0.36.2`。
- **根因**：transformers 4.57.3 约束 hub<1.0，与 Colab 预装 gradio/diffusers 的 hub>=1.16 互斥；
  pip 只能满足其一，告警但不中断。
- **判定**：良性。本项目不使用 gradio/diffusers；transformers+hub 0.36.2 自洽。
- **动作**：忽略并记录；不卸载/升级 Colab 基础包（避免破坏平台环境）。

## P-16 peft 的 torchao 版本误检：覆写模块属性无效，必须卸载 torchao
- **现象**：LoRA 初始化时 `ImportError: Found an incompatible version of torchao.
  Found version 0.10.0, but only versions above 0.16.0 are supported`。
- **根因**：`peft.import_utils.is_torchao_available()` 在环境存在 <0.16.0 的 torchao 时直接 raise
  （不是返回 False）。Colab 基础镜像预装 torchao==0.10.0。
  覆写 `peft.import_utils` / `peft.tuners.lora.model` 的属性只修住两个绑定；真正调用原函数的
  是其它在 import 期按值持有该函数的 peft 模块（traceback 折叠帧），故补丁无效。
- **正解**：移除触发源——`pip uninstall -y torchao` + `sys.modules.pop("torchao", None)`。
  本项目不用 torchao（量化走 bitsandbytes）；卸载后 find_spec→None，函数对任何调用方返回 False；
  脏内核/新内核均生效，无需重启。已注入 v4 notebook Cell 1 顶部（幂等）。
