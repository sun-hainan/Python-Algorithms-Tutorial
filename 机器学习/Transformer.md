# Transformer

_深度学习的革命性架构_

---

## 第一部分：Transformer 概述

### 核心思想

**完全基于 Attention，抛弃 RNN 和 CNN。**

```
Transformer 架构：

输入 → [Encoder] × N → [Decoder] × N → 输出

Encoder: Multi-Head Self-Attention + Feed Forward
Decoder: Masked Self-Attention + Encoder-Decoder Attention + Feed Forward
```

---

## 第二部分：Positional Encoding

### 为什么需要位置编码？

```
RNN: 天然有序，循环处理
Attention: 同时计算所有位置，丢失顺序信息

解决：给每个位置加一个独特的编码
```

```python
import numpy as np

def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    positions = np.arange(seq_len)[:, np.newaxis]
    indices = np.arange(d_model)[np.newaxis, :]
    PE[:, 0::2] = np.sin(positions / np.power(10000, 2 * indices[:, 0::2] / d_model))
    PE[:, 1::2] = np.cos(positions / np.power(10000, 2 * indices[:, 1::2] / d_model))
    return PE
```

---

## 第三部分：PyTorch 实现

```python
import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, num_layers=6):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, batch_first=True), num_layers)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, num_heads, batch_first=True), num_layers)
        self.output = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        src = self.pos_encoding(self.src_embedding(src) * math.sqrt(512))
        tgt = self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(512))
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        return self.output(output)
```

---

## 小结

1. **Transformer**完全基于Attention，抛弃RNN
2. 核心组件：Positional Encoding、Multi-Head Attention、Feed Forward
3. 奠定了现代 NLP 的基础（BERT、GPT 等）

---

_继续学习：下一章「VAE自编码器」_
