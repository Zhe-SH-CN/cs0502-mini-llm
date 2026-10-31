"""任务一：课程问答（CourseQA）。

数据来源：作者依据《CS0502 计算机科学导论》八个选题 PDF 与课程关联知识点
手工编写的结构化知识库 KB（见 docs/DATACARD.md），再由确定性模板生成 QA 对。
不含任何模型生成内容，保证标签可信。

样本字段：{"id","topic","qtype","question","answer","scorer"}
  scorer ∈ {"em","f1"}：em 用精确匹配（短答案），f1 用字符级 F1（一句话解释）
"""
import random

# topic -> 概念条目。short=短答案(EM)，desc=一句话解释(F1)，attrs=附加短问答(EM)
KB = [
    # ---- 计算机网络 / 密码与安全（选题一、四）----
    dict(topic="计算机网络", term="P2P 通信", short="对等通信",
         desc="P2P 通信指网络中各设备地位平等、不依赖中心服务器直接交换数据的通信方式。",
         attrs={"P2P 架构中是否存在中心服务器？": "不存在"}),
    dict(topic="密码与安全", term="对称加密", short="同一把密钥",
         desc="对称加密指加密与解密使用同一把密钥的加密方式。",
         attrs={"对称加密的加解密密钥是否相同？": "相同"}),
    dict(topic="密码与安全", term="非对称加密", short="公钥和私钥",
         desc="非对称加密使用一对密钥：公开的公钥用于加密，保密的私钥用于解密。",
         attrs={"RSA 属于对称加密还是非对称加密？": "非对称加密"}),
    dict(topic="密码与安全", term="数字签名", short="私钥签名公钥验签",
         desc="数字签名用发送方私钥对摘要签名、接收方用公钥验签，以证明身份与完整性。",
         attrs={"数字签名用哪把密钥生成？": "私钥"}),
    dict(topic="密码与安全", term="哈希函数", short="单向不可逆",
         desc="哈希函数把任意长度输入映射为固定长度摘要，且单向不可逆、抗碰撞。",
         attrs={"哈希函数是否可逆？": "不可逆"}),
    dict(topic="密码与安全", term="弱 RSA 密钥", short="模数过小可被分解",
         desc="弱 RSA 密钥指模数位数过小，可被因数分解从而推出私钥的密钥。",
         attrs={"破解小参数 RSA 的核心数学问题是什么？": "大整数因数分解"}),
    # ---- 人工智能 / 机器学习 / 强化学习（选题二）----
    dict(topic="强化学习", term="智能体", short="agent",
         desc="智能体是在环境中感知状态、采取动作并以奖励为目标进行学习的决策主体。",
         attrs={"强化学习中给出反馈信号的是什么？": "奖励"}),
    dict(topic="强化学习", term="状态", short="state",
         desc="状态是环境在某一时刻的观测表示，是智能体决策的依据。",
         attrs={}),
    dict(topic="强化学习", term="奖励函数", short="reward",
         desc="奖励函数把状态与动作映射为标量反馈，定义任务的优化目标。",
         attrs={"强化学习的目标通常是最大化什么？": "累积奖励"}),
    dict(topic="强化学习", term="Q-learning", short="值函数方法",
         desc="Q-learning 是一种通过学习状态-动作价值函数来求解最优策略的无模型强化学习算法。",
         attrs={"Q-learning 学习的是策略还是价值函数？": "价值函数"}),
    dict(topic="机器学习", term="过拟合", short="训练好测试差",
         desc="过拟合指模型在训练集上表现好但在新数据上表现差，即 memorize 了噪声。",
         attrs={"过拟合时训练误差与测试误差的关系？": "训练误差低测试误差高"}),
    dict(topic="机器学习", term="监督学习", short="有标签数据",
         desc="监督学习利用带标签的训练数据学习输入到输出的映射。",
         attrs={"监督学习的训练数据是否带标签？": "带标签"}),
    # ---- 数据结构 / 图 / 搜索（选题三）----
    dict(topic="数据结构", term="BFS", short="广度优先搜索",
         desc="BFS 按层逐层扩展节点，使用队列实现，适合求无权图最短路。",
         attrs={"BFS 使用哪种数据结构？": "队列"}),
    dict(topic="数据结构", term="Dijkstra 算法", short="单源最短路",
         desc="Dijkstra 算法用贪心策略求非负权图的单源最短路径。",
         attrs={"Dijkstra 能否处理负权边？": "不能"}),
    dict(topic="数据结构", term="A* 算法", short="启发式搜索",
         desc="A* 在 Dijkstra 基础上加入启发函数估计剩余代价，以加速搜索。",
         attrs={"A* 的估价函数 f 等于 g 加什么？": "h"}),
    dict(topic="数据结构", term="哈夫曼树", short="最优前缀编码树",
         desc="哈夫曼树是按字符频率构建的最优二叉前缀编码树，频率越高编码越短。",
         attrs={"哈夫曼编码中频率越高的字符编码越怎样？": "越短"}),
    dict(topic="数据结构", term="前缀编码", short="无码字是另一码字前缀",
         desc="前缀编码指任一码字都不是其他码字的前缀，因而可无歧义解码。",
         attrs={}),
    # ---- 数据库 / 数据挖掘（选题七）----
    dict(topic="数据挖掘", term="关联规则", short="共现规律",
         desc="关联规则挖掘事务数据中商品共同出现的规律，如一起被购买的商品组合。",
         attrs={"关联规则挖掘的典型数据是什么？": "交易记录"}),
    dict(topic="数据挖掘", term="支持度", short="support",
         desc="支持度是某商品组合在所有事务中出现的比例。",
         attrs={"支持度衡量的是出现频率还是条件概率？": "出现频率"}),
    dict(topic="数据挖掘", term="置信度", short="confidence",
         desc="置信度是在包含 X 的事务中同时包含 Y 的条件概率。",
         attrs={"置信度属于哪种概率？": "条件概率"}),
    dict(topic="数据库", term="主键", short="唯一标识一行",
         desc="主键是表中唯一标识每一行记录、且非空的字段。",
         attrs={"主键是否允许重复？": "不允许"}),
    # ---- 大模型 / Transformer（选题五、本课核心）----
    dict(topic="大模型", term="Transformer", short="自注意力架构",
         desc="Transformer 是基于自注意力机制的神经网络架构，是当前大语言模型的基础。",
         attrs={"Transformer 的核心机制是什么？": "自注意力"}),
    dict(topic="大模型", term="自注意力", short="self-attention",
         desc="自注意力让每个 token 依据与所有 token 的相关性加权聚合信息。",
         attrs={}),
    dict(topic="大模型", term="token", short="文本最小处理单元",
         desc="token 是模型处理文本的最小单元，由分词器把文本切分得到。",
         attrs={"分词器的作用是把文本切分成什么？": "token"}),
    dict(topic="大模型", term="微调", short="fine-tuning",
         desc="微调是在预训练模型上用任务数据继续训练，使其适配特定任务。",
         attrs={"微调通常在哪个模型基础上进行？": "预训练模型"}),
    dict(topic="大模型", term="LoRA", short="低秩适配",
         desc="LoRA 冻结原权重、只训练注入的低秩旁路矩阵，从而低成本微调大模型。",
         attrs={"LoRA 是否更新原始全部权重？": "不更新"}),
    dict(topic="大模型", term="量化", short="降低权重精度",
         desc="量化用更低比特表示权重与激活，以减少显存与加速推理。",
         attrs={"INT4 量化每个权重占多少比特？": "4"}),
    dict(topic="大模型", term="SFT", short="监督微调",
         desc="SFT 用带标签的指令-回答数据对模型做监督微调。",
         attrs={}),
    dict(topic="大模型", term="幻觉", short="hallucination",
         desc="幻觉指模型生成看似合理但与事实不符的内容。",
         attrs={}),
    # ---- 计算理论 / 信息论 / 编码（选题六、通识）----
    dict(topic="计算理论", term="图灵机", short="计算数学模型",
         desc="图灵机是描述可计算性的抽象数学模型，是现代计算机的理论基础。",
         attrs={"图灵机属于硬件还是数学模型？": "数学模型"}),
    dict(topic="计算理论", term="时间复杂度", short="运行时间随规模增长",
         desc="时间复杂度描述算法运行时间随输入规模增长的阶。",
         attrs={"O(n log n) 与 O(n^2) 哪个增长更慢？": "O(n log n)"}),
    dict(topic="信息论", term="信息熵", short="不确定性的度量",
         desc="信息熵度量信源的平均不确定性，是压缩的理论下界。",
         attrs={"信息熵越高表示不确定性越怎样？": "越高"}),
    dict(topic="信息论", term="无损压缩", short="可完全还原",
         desc="无损压缩保证解压后与原数据完全一致。",
         attrs={"无损压缩解压后是否与原文件一致？": "一致"}),
    dict(topic="编码", term="二进制", short="基数为2",
         desc="二进制是基数为 2 的计数系统，计算机内部以二进制表示数据。",
         attrs={"二进制的基数是多少？": "2"}),
    dict(topic="编码", term="字节", short="8 位",
         desc="字节由 8 个比特组成，是常见的存储计量单位。",
         attrs={"1 字节等于多少比特？": "8"}),
]

_Q_DEF = ["请用一句话解释{term}。", "什么是{term}？请简要说明。", "用一句话说明{term}的含义。"]
_Q_TOPIC = ["「{term}」属于本课程哪个主题？", "{term}对应课程八个选题中的哪个知识主题？"]


def build(seed: int = 42) -> list[dict]:
    """确定性生成 QA 样本。train/dev/test 切分在 build_data.py 中按 topic 分层完成。"""
    rng = random.Random(seed)
    rows = []
    n = 0
    for entry in KB:
        term, topic = entry["term"], entry["topic"]
        # 一句话解释 -> F1（两种问法，扩充规模）
        for q in rng.sample(_Q_DEF, 2):
            rows.append(dict(id=f"cqa-{n:04d}", topic=topic, qtype="def",
                             question=q.format(term=term), answer=entry["desc"], scorer="f1"))
            n += 1
        # 主题归属 -> EM
        q = rng.choice(_Q_TOPIC).format(term=term)
        rows.append(dict(id=f"cqa-{n:04d}", topic=topic, qtype="topic",
                         question=q, answer=topic, scorer="em"))
        n += 1
        # 极简短答 -> EM
        rows.append(dict(id=f"cqa-{n:04d}", topic=topic, qtype="short",
                         question=f"用不超过 12 个字回答：{term} 的核心是什么？",
                         answer=entry["short"], scorer="em"))
        n += 1
        # 附加短问答 -> EM
        for aq, aa in entry["attrs"].items():
            rows.append(dict(id=f"cqa-{n:04d}", topic=topic, qtype="attr",
                             question=aq, answer=aa, scorer="em"))
            n += 1
    rng.shuffle(rows)
    return rows
