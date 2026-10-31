"""实验配置与模型/任务注册表。

硬件适配原则（第一性原理）：
- T4 = Turing sm7.5：无原生 bf16、不支持 FlashAttention-2 → 统一 fp16 + sdpa
- 16GB 显存：0.5B 可全参 SFT；1.5B 及以上只走 QLoRA(4bit)
"""
from dataclasses import dataclass, field, asdict
from typing import Optional

TASKS = ("courseqa", "jsonout", "intent")

# 模型注册表：name -> (HF id, 参数量级, 适配 T4 的训练方式)
MODELS = {
    "minimind-26m": ("jingyaogong/MiniMind2-Small", "26M", ["sft", "lora"]),
    "minimind-104m": ("jingyaogong/MiniMind2", "104M", ["sft", "lora"]),
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B-Instruct", "0.5B", ["sft", "lora", "qlora"]),
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B-Instruct", "1.5B", ["lora", "qlora"]),
}

# T4 兼容的注意力与精度（sm7.5 无 bf16 / 无 FA2）
T4_DTYPE = "float16"
T4_ATTN = "sdpa"


@dataclass
class ExpConfig:
    """一次实验的完整描述，可序列化为 JSON 随结果落盘，保证可复现。"""
    run_id: str
    model_key: str = "minimind-26m"
    task: str = "courseqa"
    method: str = "sft"            # sft | lora | qlora | dpo
    # 训练超参
    epochs: int = 2
    batch_size: int = 4
    grad_accum: int = 4
    lr: float = 2e-4
    max_seq_len: int = 512
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    seed: int = 42
    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target: str = "all"       # all | qv | qkvo
    # 数据规模（消融用）
    max_train_samples: Optional[int] = None
    noise_ratio: float = 0.0       # 标签噪声比例（数据质量消融用）
    # 硬件
    dtype: str = T4_DTYPE
    attn: str = T4_ATTN
    use_4bit: bool = False
    # 生成
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 1.0
    n_samples: int = 1             # 评测采样次数（>1 看方差）
    extra: dict = field(default_factory=dict)

    @property
    def hf_id(self) -> str:
        return MODELS[self.model_key][0]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExpConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})
