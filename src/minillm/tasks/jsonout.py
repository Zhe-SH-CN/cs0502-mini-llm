"""任务二：结构化输出（JSONOut）。

给定自然语言描述 + 目标 schema，模型须输出严格符合 schema 的 JSON。
指标：json_parse_rate / schema_valid_rate / field_accuracy。
该任务对基座模型区分度极高：未微调模型常输出散文而非 JSON。
"""
import json
import random

SCHEMAS = {
    "exp_result": {
        "fields": {
            "model": {"type": "str"},
            "task": {"type": "str", "enum": ["courseqa", "jsonout", "intent"]},
            "em": {"type": "float"},
            "epochs": {"type": "int"},
            "gpu": {"type": "str"},
        }
    },
    "assignment": {
        "fields": {
            "student_id": {"type": "str"},
            "course": {"type": "str"},
            "assignment": {"type": "str"},
            "submitted": {"type": "bool"},
            "score": {"type": "int"},
        }
    },
    "secondhand": {
        "fields": {
            "title": {"type": "str"},
            "price": {"type": "float"},
            "category": {"type": "str", "enum": ["教材", "电子产品", "生活用品", "运动器材"]},
            "condition": {"type": "str", "enum": ["全新", "九成新", "七成新", "有使用痕迹"]},
            "negotiable": {"type": "bool"},
        }
    },
    "rl_config": {
        "fields": {
            "game": {"type": "str"},
            "algorithm": {"type": "str", "enum": ["Q-learning", "DQN", "PPO", "A2C"]},
            "episodes": {"type": "int"},
            "learning_rate": {"type": "float"},
            "use_gpu": {"type": "bool"},
        }
    },
}

_POOL = {
    "exp_result": dict(
        model=["MiniMind2-Small", "MiniMind2", "Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct"],
        task=["courseqa", "jsonout", "intent"],
        em=[0.12, 0.35, 0.61, 0.78, 0.86, 0.93],
        epochs=[1, 2, 3, 4],
        gpu=["T4", "L40", "A100", "CPU"],
    ),
    "assignment": dict(
        student_id=[f"524030910{i:03d}" for i in range(1, 12)],
        course=["计算机科学导论", "数据结构", "计算机网络", "数据库"],
        assignment=["选题五 Mini-LLM", "选题三 Map", "哈夫曼压缩实验", "RSA 签名实验"],
        submitted=[True, False],
        score=[60, 72, 80, 85, 90, 95, 100],
    ),
    "secondhand": dict(
        title=["高等数学教材上下册", "Kindle 阅读器", "自行车", "羽毛球拍", "台灯", "机械键盘"],
        price=[15.0, 39.5, 120.0, 260.0, 45.0, 199.0],
        category=["教材", "电子产品", "生活用品", "运动器材"],
        condition=["全新", "九成新", "七成新", "有使用痕迹"],
        negotiable=[True, False],
    ),
    "rl_config": dict(
        game=["Flappy Bird", "贪吃蛇", "打砖块", "CartPole"],
        algorithm=["Q-learning", "DQN", "PPO", "A2C"],
        episodes=[500, 1000, 2000, 5000],
        learning_rate=[1e-4, 3e-4, 1e-3, 5e-3],
        use_gpu=[True, False],
    ),
}

_TPL = {
    "exp_result": "模型 {model} 在 {task} 任务上微调后精确匹配分数为 {em}，共训练 {epochs} 个 epoch，使用 {gpu}。请把上述实验结果整理成 JSON。",
    "assignment": "学号 {student_id} 的同学在《{course}》课程提交了「{assignment}」，提交状态为{submitted_cn}，得分 {score} 分。请输出对应的 JSON 记录。",
    "secondhand": "有一件二手商品：{title}，售价 {price} 元，分类为{category}，成色{condition}，{neg_cn}议价。请生成 JSON。",
    "rl_config": "要为游戏 {game} 配置强化学习训练：算法用 {algorithm}，训练 {episodes} 个 episode，学习率 {learning_rate}，{gpu_cn}使用 GPU。请输出 JSON 配置。",
}


def _cn_bool(v: bool) -> str:
    return "已" if v else "未"


def _render(schema_key: str, gold: dict) -> str:
    if schema_key == "exp_result":
        return _TPL[schema_key].format(**gold)
    if schema_key == "assignment":
        return _TPL[schema_key].format(submitted_cn=_cn_bool(gold["submitted"]), **gold)
    if schema_key == "secondhand":
        return _TPL[schema_key].format(neg_cn="可" if gold["negotiable"] else "不可", **gold)
    if schema_key == "rl_config":
        return _TPL[schema_key].format(gpu_cn="需要" if gold["use_gpu"] else "不需要", **gold)
    raise ValueError(schema_key)


def build(seed: int = 42, per_schema: int = 160) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    n = 0
    for key, schema in SCHEMAS.items():
        pool = _POOL[key]
        keys = list(schema["fields"].keys())
        for _ in range(per_schema):
            gold = {k: rng.choice(pool[k]) for k in keys}
            instr = _render(key, gold) + f" 只输出 JSON，键为：{', '.join(keys)}。"
            rows.append(dict(id=f"json-{n:04d}", schema_key=key, schema=schema,
                             instruction=instr, gold=gold))
            n += 1
    rng.shuffle(rows)
    return rows


def gold_json(row: dict) -> str:
    return json.dumps(row["gold"], ensure_ascii=False)
