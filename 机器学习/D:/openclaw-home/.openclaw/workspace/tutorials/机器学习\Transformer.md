# Transformer

_深度学习的革命性架构_

---

## 📖 学习目标

1. 理解 Transformer 的整体架构
2. 掌握 Positional Encoding
3. 能理解 Encoder-Decoder 结构

---

## 第一部分：Transformer 概述

### 🎯 核心思想

**完全基于 Attention，抛弃 RNN 和 CNN。**

```
Transformer 架构：

输入 → [Encoder] × N → [Decoder] × N → 输出

Encoder: Multi-Head Self-Attention + Feed Forward
Decoder: Masked Self-Attention + Encoder-Decoder Attention + Feed Forward
```

### 图解

```
Transformer：

     输入 embeddings
          │
          ↓
    ┌─────────────┐
    │  Positional │
    │   Encoding  │
    └─────────────┘
          │
          ↓
    ┌─────────────┐
    │   Encoder   │ × N
    │  Layer 1    │
    └─────────────┘
          │
          ↓
    ┌─────────────┐
    │   Encoder   │ × N
    │  Layer N    │
    └─────────────┘
          │
    ┌─────┴─────┐
    │           │
    ↓           ↓
┌────────┐ ┌────────────┐
│Output  │ │  Decoder   │
│Shifted │ │   × N      │
│Left    │ └────────────┘
└────────┘
```

---

## 第二部分：Positional Encoding

### 为什么需要位置编码？

```
RNN: 天然有序，循环处理
Attention: 同时计算所有位置，丢失顺序信息

解决：给每个位置加一个独特的编码
```

### 公式

```python
import numpy as np

def positional_encoding(seq_len, d_model):
    """
    位置编码

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    PE = np.zeros((seq_len, d_model))

    positions = np.arange(seq_len)[:, np.newaxis]
    indices = np.arange(d_model)[np.newaxis, :]

    # 奇偶位置用不同的公式
    PE[:, 0::2] = np.sin(positions / np.power(10000, 2 * indices[:, 0::2] / d_model))
    PE[:, 1::2] = np.cos(positions / np.power(10000, 2 * indices[:, 1::2] / d_model))

    return PE


# 可视化
# 每个位置有一个独特的编码，不同位置的值不同
```

---

## 第三部分：Encoder 层

```python
class EncoderLayer:
    """单层 Encoder"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = dropout

    def forward(self, x, mask=None):
        # Self-Attention + Residual
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed Forward + Residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class FeedForward:
    """前馈网络"""
    def __init__(self, d_model, d_ff):
        self.linear1 = np.random.randn(d_model, d_ff) * 0.1
        self.linear2 = np.random.randn(d_ff, d_model) * 0.1

    def forward(self, x):
        return np.maximum(0, np.dot(x, self.linear1)) @ self.linear2  # ReLU


class LayerNorm:
    """层归一化"""
    def __init__(self, d_model, eps=1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
```

---

## 第四部分：Decoder 层

```python
class DecoderLayer:
    """单层 Decoder"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 1. Masked Self-Attention（不能看到未来的词）
        attn1, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + attn1)

        # 2. Cross-Attention（看 Encoder 的输出）
        attn2, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + attn2)

        # 3. Feed Forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + ff_output)

        return x
```

---

## 第五部分：完整 Transformer

```python
class Transformer:
    """完整 Transformer 模型"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512,
                 num_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
        self.encoder_layers = [
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ]
        self.decoder_layers = [
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ]
        self.output_linear = np.random.randn(d_model, tgt_vocab_size) * 0.1
        self.softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

    def forward(self, src, tgt):
        # Encoder
        encoder_output = src
        for layer in self.encoder_layers:
            encoder_output = layer.forward(encoder_output)

        # Decoder
        decoder_output = tgt
        for layer in self.decoder_layers:
            decoder_output = layer.forward(decoder_output, encoder_output)

        # 输出层
        logits = np.dot(decoder_output, self.output_linear)
        return self.softmax(logits)
```

---

## 第六部分：PyTorch 实现

```python
import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512,
                 num_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Embedding
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        # Encoder & Decoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, d_ff, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # 输出
        self.output = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        # 编码
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoding(src)
        memory = self.encoder(src)

        # 解码
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoding(tgt)
        output = self.decoder(tgt, memory)

        return self.output(output)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)
```

---

## 第七部分：名词解释

### Encoder-Decoder

```
Encoder: 理解输入序列，提取特征
Decoder: 根据 Encoder 的输出，生成目标序列
```

### Mask

```
Padding Mask: 忽略填充的无效位置
Look-ahead Mask: Decoder 不能看到未来的词
```

### Layer Normalization

**定义：** 对每一层的输出进行归一化，加速训练。

---

## ✅ 小结

1. **Transformer**完全基于Attention，抛弃RNN
2. 核心组件：Positional Encoding、Multi-Head Attention、Feed Forward
3. Encoder提取特征，Decoder生成序列
4. LayerNorm 和残差连接保证稳定训练
5. 奠定了现代 NLP 的基础（BERT、GPT 等）

---

_深度学习入门部分结束！_
_继续学习：下一章「VAE自编码器」_
