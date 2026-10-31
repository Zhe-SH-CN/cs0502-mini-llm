"""任务三：意图分类（Intent）。

六类意图，覆盖课程场景下的真实用户表达。指标：accuracy / macro-F1。
"""
import random

LABELS = ["概念解释", "作业求助", "代码调试", "资源请求", "项目协作", "闲聊"]

_TPL = {
    "概念解释": [
        "什么是{c}？", "能讲讲{c}的原理吗", "{c}是什么意思", "帮我解释一下{c}",
        "{c}和{c2}有什么区别", "通俗地说{c}是什么",
    ],
    "作业求助": [
        "选题五的微调部分怎么做", "我的{c}实验报告不会写", "作业要求比较微调前后效果，该怎么设计",
        "{c}这道题思路是什么", "实验的消融部分要怎么做", "我的训练 loss 不降，作业怎么交",
    ],
    "代码调试": [
        "我的训练代码报 CUDA out of memory 怎么办", "跑 {c} 的时候报 ImportError",
        "模型输出全是乱码，代码哪里错了", "load 权重报错 state_dict 不匹配",
        "为什么我的 eval 脚本结果和训练时不一致", "bitsandbytes 装不上怎么解决",
    ],
    "资源请求": [
        "有没有{c}的课件", "求一份{c}的入门资料", "哪里有{c}的公开数据集",
        "推荐几篇{c}的论文", "有{c}的中文教程吗", "能给我{c}的代码仓库链接吗",
    ],
    "项目协作": [
        "我们组五个人怎么分工比较合理", "谁负责写技术报告", "答辩 PPT 谁来做",
        "实验结果怎么汇总到一起", "我们把{c}这部分分给谁", "下次组会什么时候开",
    ],
    "闲聊": [
        "今天食堂好多人", "你好呀", "这门课好难啊哈哈", "周末有什么安排",
        "刚打完球回来", "天气真不错",
    ],
}

_CONCEPTS = ["哈夫曼编码", "Dijkstra", "自注意力", "LoRA", "量化", "强化学习", "RSA", "关联规则",
             "图灵机", "信息熵", "BFS", "数字签名", "Transformer", "过拟合", "前缀编码", "支持度"]


def build(seed: int = 42, per_label: int = 160) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    n = 0
    for lab in LABELS:
        tpls = _TPL[lab]
        for i in range(per_label):
            t = tpls[i % len(tpls)]
            c = rng.choice(_CONCEPTS)
            c2 = rng.choice([x for x in _CONCEPTS if x != c])
            text = t.format(c=c, c2=c2)
            # 加少量口语扰动，避免纯模板
            if rng.random() < 0.3:
                text = rng.choice(["请问", "麻烦问下", "hi，", ""]) + text
            if rng.random() < 0.2:
                text += rng.choice(["？", "!", "～", ""])
            rows.append(dict(id=f"int-{n:04d}", text=text.strip(), label=lab))
            n += 1
    rng.shuffle(rows)
    return rows
