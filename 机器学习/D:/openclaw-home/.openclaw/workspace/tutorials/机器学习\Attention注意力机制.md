# Attention注意力机制

_让模型"关注"最重要的信息_

---

## 📖 学习目标

1. 理理解Attention的核心思想
2. 掌握Self-Attention的计算过程
3. 为学习Transformer打下基础

---

## 第一部分：什么是Attention？

### 🎯 核心思想

**不再把所有信息一次性处理，而是让模型"看"哪些部分更重要。**

```
RNN: 所有信息压缩成一个固定向量
Attention: 对所有位置加权求和，权重由相关性决定
```

### 图解

```
翻译任务：The cat sat on the mat
            ↓
Attention 权重可视化：
"猫" 位置会更多关注 "cat" 的信息
"坐" 位置会更多关注 "sat" 的信息
```

---

## 第二部分：Attention 计算过程

### 三步走

```
1. 计算 Query (Q), Key (K), Value (V)

2. 计算注意力分数：score(Q, K)

3. 加权求和：attention = softmax(score) × V
```

### 数学公式

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

- Q, K, V: 查询、键、值矩阵
- d_k: 键的维度
- √d_k: 缩放因子，防止点积过大
```

---

## 第三部分：代码实现

```python
import numpy as np

def attention(Q, K, V, mask=None):
    """
    缩放点积注意力

    Q: (seq_len, d_k)
    K: (seq_len, d_k)
    V: (seq_len, d_v)
    """
    d_k = K.shape[-1]

    # 1. 计算点积
    scores = np.dot(Q, K.T) / np.sqrt(d_k)  # (seq_len, seq_len)

    # 2. 应用 mask（可选）
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # 3. softmax
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # 4. 加权求和
    output = np.dot(attention_weights, V)

    return output, attention_weights


# 测试
seq_len, d_k, d_v = 4, 8, 8
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_v)

output, weights = attention(Q, K, V)
print(f"输出形状: {output.shape}")
print(f"注意力权重形状: {weights.shape}")
print(f"注意力权重（每行和为1）: {weights.sum(axis=1)}")
```

---

## 第四部分：Self-Attention

### 什么是 Self-Attention？

**Query, Key, Value 都来自同一个输入。**

```python
class SelfAttention:
    """Self-Attention 层"""
    def __init__(self, d_model):
        self.d_model = d_model

        # Q, K, V 变换矩阵
        self.Wq = np.random.randn(d_model, d_model) * 0.1
        self.Wk = np.random.randn(d_model, d_model) * 0.1
        self.Wv = np.random.randn(d_model, d_model) * 0.1

    def forward(self, x):
        # x: (seq_len, d_model)
        Q = np.dot(x, self.Wq)
        K = np.dot(x, self.Wk)
        V = np.dot(x, self.Wv)

        # 计算注意力
        output, weights = attention(Q, K, V)

        return output, weights
```

---

## 第五部分：多头注意力 (Multi-Head Attention)

```python
class MultiHeadAttention:
    """多头注意力"""
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # 多个注意力头
        self.Wq = [np.random.randn(d_model, self.d_k) * 0.1 for _ in range(num_heads)]
        self.Wk = [np.random.randn(d_model, self.d_k) * 0.1 for _ in range(num_heads)]
        self.Wv = [np.random.randn(d_model, self.d_v) * 0.1 for _ in range(num_heads)]
        self.Wo = np.random.randn(d_model, d_model) * 0.1

    def forward(self, x):
        outputs = []
        weights_all = []

        for i in range(self.num_heads):
            Q = np.dot(x, self.Wq[i])
            K = np.dot(x, self.Wk[i])
            V = np.dot(x, self.Wv[i])

            out, weights = attention(Q, K, V)
            outputs.append(out)
            weights_all.append(weights)

        # 拼接所有头的输出
        concat = np.concatenate(outputs, axis=-1)
        output = np.dot(concat, self.Wo)

        return output, weights_all
```

---

## 第六部分：PyTorch 实现

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 线性变换
        Q = self.Wq(query)
        K = self.Wk(key)
        V = self.Wv(value)

        # 分成多个头
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        # 合并多头
        concat = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        output = self.Wo(concat)
        return output, attention_weights
```

---

## 第七部分：实际应用场景

### 📝 场景1：机器翻译

```python
# 源语言的每个词，关注目标语言的相关词
# "I love you" → "我 爱 你"
```

### 🤖 场景2：ChatGPT 的核心

```python
# Transformer 中完全使用 Self-Attention
# GPT: Decoder-only, 单向注意力
# BERT: Encoder-only, 双向注意力
```

### 📚 场景3：文档摘要

```python
# 每个句子关注文档中最重要的其他句子
```

---

## 第八部分：名词解释

### Query, Key, Value

```
Query (Q): 我要找什么
Key (K): 我有什么
Value (V): 实际的内容

类比图书馆：
- Query: 你想找的书（问题）
- Key: 书的索引（标签）
- Value: 书的实际内容

注意力 = 在索引中找到最相关的书，然后获取内容
```

### 缩放因子 √d_k

**原因：** 当 d_k 很大时，点积的值会很大，导致 softmax 梯度很小。

---

## ✅ 小结

1. **Attention**让模型关注最重要的信息
2. 核心：**Query-Key-Value** 三元组
3. 计算：softmax(QK^T/√d_k) × V
4. **Multi-Head Attention**从多个角度关注信息
5. 是 Transformer 的核心组件

---

_继续学习：下一章「Transformer」_
