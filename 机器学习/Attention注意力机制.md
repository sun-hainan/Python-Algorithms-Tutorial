# Attention注意力机制

_让神经网络学会"关注"重点_

---

## 📖 学习目标

1. 理解 Seq2Seq 的瓶颈问题
2. 掌握 Query-Key-Value 的核心思想
3. 推导 Attention 的数学公式
4. 理解多头注意力（Multi-Head Attention）

---

## 第一部分：Seq2Seq 的问题

### 编码器-解码器架构

```
Encoder:                Decoder:
"今天天气很好" ──→ [上下文向量] ──→ "Today weather is good"

问题：上下文向量是固定长度的！
      不管输入多长，都压缩成一个向量！
      信息瓶颈！
```

### 信息瓶颈的可视化

```
短句子："天气好" → [向量c] → "Good"
        信息量少，压缩容易 ✓

长句子："今天早上我和朋友去公园...（省略100字）...所以我们最后..." → [向量c]
        信息量巨大，压缩困难 ✗
        后面的信息可能丢失！
```

---

## 第二部分：Attention 的核心思想

### 解决之道

```
不要只用一个上下文向量！
每个解码步骤，使用不同的上下文向量！

 Decoder step 1: 关注输入的第 1,2 个词 → 输出第 1 个词
 Decoder step 2: 关注输入的第 3,4 个词 → 输出第 2 个词
 Decoder step 3: 关注输入的第 5,6 个词 → 输出第 3 个词
```

### Attention 图解

```
翻译: "今天天气很好" → "Today weather is good"

Step 1 (输出 "Today"):
  "今天天气很好"
   ↓↓↓
  [0.9, 0.1, 0.0, 0.0]  ← 关注"今"
   ↓↓↓                  "今"+"天" → "Today"

Step 2 (输出 "weather"):
  "今天天气很好"
     ↓↓↓
  [0.0, 0.8, 0.2, 0.0]  ← 关注"天气"
   ↓↓↓                  "天气" → "weather"

Step 3 (输出 "is"):
  "今天天气很好"
        ↓↓↓
  [0.0, 0.2, 0.7, 0.1]  ← 关注"很"
   ↓↓↓                  "很好" → "is good"
```

---

## 第三部分：Query-Key-Value 理解

### 图书馆的比喻

```
Query (查询): "我想找一本关于机器学习的书"
              ↓
  你在找什么？

Key (键): 每个书架上的标签
          "机器学习" "深度学习" "Python编程"
          ↓
  标签匹配度？

Value (值): 书架上书的实际内容
            ↓
  匹配后要获取的信息

Attention = 根据 Query 找到最相关的 Key，取出对应的 Value
```

### 数学表达

```
每个词有三个向量：
- Query (Q): 我要找什么
- Key (K): 我有什么标签
- Value (V): 实际内容

Attention(Q, K, V) = 加权平均(V)

权重 = softmax(相似度(Q, K))

其中相似度 = Q · K^T / √d_k
```

---

## 第四部分：Attention 数学推导

### 第一步：计算 Query、Key、Value

```python
def compute_qkv(X, W_q, W_k, W_v):
    Q = np.dot(X, W_q)  # (seq_len, d_k)
    K = np.dot(X, W_k)  # (seq_len, d_k)
    V = np.dot(X, W_v)  # (seq_len, d_v)
    return Q, K, V
```

### 第二步：计算注意力分数

```python
def attention_scores(Q, K):
    d_k = K.shape[-1]
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    return scores
```

### 第三步：Softmax 归一化

```python
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True))
```

### 第四步：加权求和

```python
def attention(Q, K, V):
    d_k = K.shape[-1]
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    weights = softmax(scores)
    context = np.dot(weights, V)
    return context, weights
```

### 完整 Attention 函数

```python
def self_attention(X, d_k, d_v, W_q, W_k, W_v):
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)

    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    weights = softmax(scores)
    context = np.dot(weights, V)

    return context, weights
```

---

## 第五部分：多头注意力（Multi-Head Attention）

### 为什么需要多头？

```
一个注意力头只能学习一种"相关性"

多个注意力头可以同时学习不同类型的相关性：
- 头1: 关注语法关系（主语-动词）
- 头2: 关注语义关系（同义词）
- 头3: 关注位置关系（相邻词）
```

### PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        concat = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(concat)

        return output, attention_weights
```

---

## 第六部分：Self-Attention vs Cross-Attention

### Self-Attention（自注意力）

```
Query, Key, Value 都来自同一个输入

例子：分析"今天天气很好"
- "今天"的 query 查找和所有词的关系
- 每个词都在"关注"其他词

应用：Transformer 的 Encoder
```

### Cross-Attention（交叉注意力）

```
Query 来自 Decoder，Key/Value 来自 Encoder

例子：翻译
- Decoder 的 query 查找 Encoder 的 key/value
- 问："现在应该生成什么词？"（Query）
- 查找："编码器看到了什么？"（Key/Value）

应用：Transformer 的 Decoder
```

---

## ✅ 小结

1. **Attention**解决了 Seq2Seq 的信息瓶颈问题
2. 核心思想：**动态选择**需要关注的信息
3. Query-Key-Value 三元组：查询、键、值
4. Attention(Q,K,V) = softmax(QK^T/√d_k) × V
5. **多头注意力**从多个角度学习不同的相关性
6. Self-Attention：Q,K,V 同源
7. Cross-Attention：Q 来自 Decoder，K,V 来自 Encoder

---

_继续学习：下一章「Transformer」_
