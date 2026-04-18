# Attention注意力机制

_让模型"关注"最重要的信息_

---

## 第一部分：什么是Attention？

### 核心思想

**不再把所有信息一次性处理，而是让模型"看"哪些部分更重要。**

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

---

## 第二部分：计算过程

```
1. 计算 Query (Q), Key (K), Value (V)

2. 计算注意力分数：score(Q, K)

3. 加权求和：attention = softmax(score) × V
```

---

## 第三部分：代码实现

```python
def attention(Q, K, V):
    d_k = K.shape[-1]
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)
    output = np.dot(attention_weights, V)
    return output, attention_weights
```

---

## 第四部分：PyTorch 实现

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
        Q = self.Wq(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.Wk(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.Wv(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        concat = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.Wo(concat), attention_weights
```

---

## 小结

1. **Attention**让模型关注最重要的信息
2. 核心：**Query-Key-Value** 三元组
3. 是 Transformer 的核心组件

---

_继续学习：下一章「Transformer」_
