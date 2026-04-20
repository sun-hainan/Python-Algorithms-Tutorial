# RNN与LSTM

> _让神经网络拥有"记忆"，处理序列数据的核心技术_

---

## 🎯 先看一个生活中的例子

### 例子：读小说

![rnn lstm](rnn_lstm.png)


假设你读一本书：

```
第一章："小明是一个聪明的小学生，他非常喜欢数学..."
第二章："每天早上，小明都会..."
第三章："...因此，小明在比赛中获得了第一名。大家都为小明鼓掌。"

问题："小明是谁？"
答案：当然是那个"聪明的小学生"，从第一章就知道！
```

**关键：你记住了之前的内容，才能理解后面的话！**

这就是**序列数据**的特点：**前面和后面有关联！**

---

## 🤔 为什么普通神经网络处理不了序列？

### 普通神经网络的问题

```
输入：固定长度的向量
输出：固定长度的向量

但序列数据：
- "小明 喜欢 吃 苹果" 和 "苹果 吃 喜欢 小明" 意思完全不同！
- 输入长度可变
- 需要"记住"之前的内容
```

---

## 🏗️ RNN的基本结构

### RNN的核心思想

**让信息在网络中循环流动，把"记忆"传递下去**

```
标准神经网络：
  输入 → 隐藏层 → 输出

RNN（循环）：
  输入 → 隐藏层 ↔ （循环）
          ↑
          （接收上一步的隐藏状态）
```

### RNN的展开图

```
时间步 t=0:    x₀ ──→ [RNN] ──→ h₀ ──→ y₀
                     ↑
时间步 t=1:    x₁ ──→ [RNN] ──→ h₁ ──→ y₁
                     ↑
时间步 t=2:    x₂ ──→ [RNN] ──→ h₂ ──→ y₂
                     ↑
                ...

每个时间步的隐藏状态 hₜ 包含了之前所有时间步的信息！
```

### RNN的数学公式

```
hₜ = tanh(Wxh·xₜ + Whh·hₜ₋₁ + bh)
yₜ = Why·hₜ + by

其中：
- xₜ: t时刻的输入
- hₜ₋₁: t-1时刻的隐藏状态（"记忆"）
- hₜ: t时刻的隐藏状态
- tanh: 激活函数，把值压缩到 [-1, 1]
```

---

## 💻 RNN代码实现

```python
import numpy as np

class SimpleRNN:
    """简单RNN"""
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size

        # 初始化权重
        scale_xh = np.sqrt(2.0 / (input_size + hidden_size))
        scale_hh = np.sqrt(2.0 / (hidden_size + hidden_size))
        scale_hy = np.sqrt(2.0 / (hidden_size + output_size))

        self.Wxh = np.random.randn(hidden_size, input_size) * scale_xh
        self.Whh = np.random.randn(hidden_size, hidden_size) * scale_hh
        self.Why = np.random.randn(output_size, hidden_size) * scale_hy

        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def tanh(self, z):
        return np.tanh(z)

    def forward_step(self, x, h_prev):
        """单步前向传播"""
        # 计算新的隐藏状态
        h = self.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h_prev) + self.bh)
        # 计算输出
        y = np.dot(self.Why, h) + self.by
        return h, y

    def forward(self, X):
        """处理整个序列"""
        seq_len = len(X)
        h = np.zeros(self.hidden_size)  # 初始隐藏状态
        h_history = []
        y_history = []

        for t in range(seq_len):
            h, y = self.forward_step(X[t], h)
            h_history.append(h)
            y_history.append(y)

        return np.array(h_history), np.array(y_history)
```

---

## ⚠️ RNN的致命问题：梯度消失

### 时间步的反向传播（BPTT）

```
RNN 的反向传播需要沿时间展开：

∂L/∂W = Σₜ ∂Lₜ/∂W

每个时间步的梯度依赖前一个时间步：

∂hₜ/∂hₜ₋₁ = tanh' × Whh

|tanh'| 最大值 = 1
如果 |Whh| < 1：梯度指数级衰减 → 梯度消失！
```

### 长期依赖问题

```
例子：填词

"The cat, which already ate ..., was full."
     ↑
   主语是 cat（单数）

"The cats, which already ate ..., were full."
     ↑
   主语是 cats（复数）

动词形式（was/were）取决于很久之前的主语！

RNN 在反向传播时：
- 梯度需要穿过所有中间层
- 每一步都可能被衰减
- 早期信号几乎传不过去！
```

---

## 🌟 LSTM：解决长期依赖的神器

### LSTM的核心思想

**引入"门控"机制，让网络学会记住什么、忘记什么**

```
LSTM 三个门：
1. 遗忘门：决定丢弃什么信息
2. 输入门：决定存储什么新信息
3. 输出门：决定输出什么信息
```

### LSTM的结构图

```
                 ┌─────────────────────────────────────────┐
                 │                                         │
                 │    ┌───────────────────────────┐  │
      hₜ₋₁ ─────┼───→│                           │←──┤ xₜ
                 │    │   ┌─────┐     ┌─────┐       │  │
                 │    │   │遗忘门│     │输入门│       │  │
                 │    │   └─────┘     └─────┘       │  │
                 │    │       ↓           ↓         │  │
                 │    │   ┌─────────────────┐       │  │
                 │    │   │                 │       │  │
                 │    │   │    Cell State   │←────+──┤
                 │    │   │   (长期记忆)    │       │  │
                 │    │   │                 │       │  │
                 │    │   └─────────────────┘       │  │
                 │    │            ↓               │  │
                 │    │       ┌─────┐             │  │
                 │    │       │输出门│             │  │
                 │    │       └─────┘             │  │
                 │    └───────────────────────────────┘  │
                 │                    ↓                 │
                 └───────────────────→ hₜ               │

LSTM 核心：Cell State（细胞状态）= 传送带
```

### 三个门的工作原理

#### 1. 遗忘门（Forget Gate）

```
fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)

作用：决定从细胞状态中丢弃什么信息

fₜ = 0：完全丢弃
fₜ = 1：完全保留
```

#### 2. 输入门（Input Gate）

```
iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)        ← 决定更新多少
C̃ₜ = tanh(Wc·[hₜ₋₁, xₜ] + bc)    ← 候选值
```

#### 3. 更新细胞状态

```
Cₜ = fₜ × Cₜ₋₁ + iₜ × C̃ₜ

解释：
- fₜ × Cₜ₋₁：遗忘门乘以旧状态，丢弃不需要的
- iₜ × C̃ₜ：输入门乘以新候选状态，添加新信息
```

#### 4. 输出门（Output Gate）

```
oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)
hₜ = oₜ × tanh(Cₜ)
```

---

## 💻 LSTM代码实现

```python
class LSTMCell:
    """LSTM 单元"""
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        input_with_hidden = input_size + hidden_size

        # 初始化权重（随机小数值）
        self.Wf = np.random.randn(hidden_size, input_with_hidden) * 0.1
        self.Wi = np.random.randn(hidden_size, input_with_hidden) * 0.1
        self.Wc = np.random.randn(hidden_size, input_with_hidden) * 0.1
        self.Wo = np.random.randn(hidden_size, input_with_hidden) * 0.1

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, x, h_prev, C_prev):
        """单步前向传播"""
        concat = np.vstack([h_prev, x])

        # 遗忘门：决定丢弃什么
        f = self.sigmoid(np.dot(self.Wf, concat) + self.bf)

        # 输入门：决定更新什么
        i = self.sigmoid(np.dot(self.Wi, concat) + self.bi)
        C_tilde = np.tanh(np.dot(self.Wc, concat) + self.bc)

        # 更新细胞状态
        C = f * C_prev + i * C_tilde

        # 输出门：决定输出什么
        o = self.sigmoid(np.dot(self.Wo, concat) + self.bo)
        h = o * np.tanh(C)

        return h, C
```

---

## 🌟 GRU：LSTM的简化版

### GRU的结构

```
GRU 只有两个门：
- 更新门（类似 LSTM 的遗忘+输入门）
- 重置门（控制忽略上一隐藏状态的程度）

比 LSTM 少一个门，参数更少，训练更快！
```

```python
class GRUCell:
    """GRU 单元（简化版 LSTM）"""
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size

        self.Wr = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.Wz = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.Wh = np.random.randn(hidden_size, input_size + hidden_size) * 0.1

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, x, h_prev):
        concat = np.vstack([h_prev, x])

        # 重置门
        r = self.sigmoid(np.dot(self.Wr, concat))

        # 更新门
        z = self.sigmoid(np.dot(self.Wz, concat))

        # 候选隐藏状态
        concat_r = np.vstack([r * h_prev, x])
        h_tilde = np.tanh(np.dot(self.Wh, concat_r))

        # 最终隐藏状态
        h = (1 - z) * h_prev + z * h_tilde

        return h
```

---

## 🧪 例子：情感分析

```python
# 情感分析：判断句子是正面还是负面
# "这家店的东西很好吃" -> 正面 (1)
# "服务态度太差了" -> 负面 (0)

# 简化版：把每个词编码成一个向量
# 实际应用中会用 Word2Vec, BERT 等预训练词向量

# 假设词向量维度是 4
vocab = {
    '这': 0, '家': 1, '店': 2, '的': 3, '东': 4, '西': 5,
    '很': 6, '好': 7, '吃': 8, '服': 9, '务': 10, '态': 11,
    '度': 12, '太': 13, '差': 14, '了': 15
}

# 简化：用一个随机向量代表每个词
np.random.seed(42)
word_vectors = {k: np.random.randn(4) for k in vocab}

# 句子
sentence1 = ['这', '家', '店', '的', '东', '西', '很', '好', '吃']  # 正面
sentence2 = ['服', '务', '态', '度', '太', '差', '了']  # 负面
```

---

## ✅ 本章小结

| 概念 | 解释 |
|------|------|
| RNN | 循环神经网络，能处理序列数据 |
| 隐藏状态 | 传递"记忆"的载体 |
| 梯度消失 | RNN 难以学习长期依赖的根本原因 |
| LSTM | 长短期记忆网络，用门控解决长期依赖 |
| 遗忘门 | 决定丢弃什么信息 |
| 输入门 | 决定存储什么新信息 |
| 输出门 | 决定输出什么信息 |
| GRU | LSTM 的简化版，效果相当 |

---

## 🔗 继续学习

RNN 和 LSTM 奠定了序列建模的基础，但它们有一个根本限制：必须逐个处理序列。下一章我们将学习 **Attention 注意力机制**，它可以并行处理整个序列！

👉 [Attention注意力机制](./Attention注意力机制.md)
