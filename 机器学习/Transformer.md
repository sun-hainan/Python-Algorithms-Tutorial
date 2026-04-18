# Transformer

_完全基于Attention的革命性架构_

---

## 📖 学习目标

1. 理解Transformer的整体架构
2. 掌握Positional Encoding
3. 理解Encoder和Decoder的区别
4. 理解LayerNorm和残差连接

---

## 第一部分：Transformer 概述

### 核心创新

```
RNN的问题：
- 顺序处理，无法并行
- 长期依赖难以学习

Transformer的解决方案：
- 完全基于Attention
- 并行计算
- 直接建模任意距离的依赖关系
```

### 整体架构

```
                Transformer
                        │
    ┌───────────────────┴───────────────────┐
    │                                       │
 Encoder                                 Decoder
    │                                       │
    ├──→ Multi-Head Attention ────────────→  │
    │         ↓                             │
    │      Add & LayerNorm                  │
    │         ↓                             │
    │      Feed Forward                    │
    │         ↓                             │
    │      Add & LayerNorm                 │
    │         ↓                             │
    └──→ (重复N次)                            │
                                           │
    ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←│
                    ↑
                    │ ← Cross Attention
```

---

## 第二部分：Positional Encoding

### 为什么需要位置编码？

```
Attention 本身不包含位置信息！
"狗咬人" 和 "人咬狗" 用 Attention 看是一样的！

需要在输入中加入位置信息。
```

### 三角函数位置编码

```python
import numpy as np

def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    positions = np.arange(seq_len)[:, np.newaxis]
    indices = np.arange(d_model)[np.newaxis, :]
    angles = positions / np.power(10000, 2 * indices / d_model)
    PE[:, 0::2] = np.sin(angles[:, 0::2])
    PE[:, 1::2] = np.cos(angles[:, 1::2])
    return PE
```

### 为什么用三角函数？

```
优点1：可以表示相对位置
  sin(pos+k) 和 cos(pos+k) 可以写成 sin(pos) 和 cos(pos) 的线性组合

优点2：模型可以看到任意位置的编码
```

---

## 第三部分：Encoder

### 单个 Encoder 层

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x
```

---

## 第四部分：Decoder

### 单个 Decoder 层

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.masked_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 掩码自注意力（不能看到未来）
        mask_attn, _ = self.masked_attention(x, x, x, tgt_mask)
        x = self.norm1(x + mask_attn)

        # 交叉注意力
        cross_attn, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + cross_attn)

        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.norm3(x + ffn_output)

        return x
```

---

## 第五部分：完整 Transformer

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512,
                 num_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.src_embedding(src)
        encoder_output = self.pos_encoding(encoder_output)

        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, src_mask)

        decoder_output = self.tgt_embedding(tgt)
        decoder_output = self.pos_encoding(decoder_output)

        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, src_mask, tgt_mask)

        output = self.fc_out(decoder_output)
        return output
```

---

## 第六部分：训练技巧

### Label Smoothing

```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        smooth_value = self.smoothing / (self.num_classes - 1)

        one_hot = torch.zeros_like(pred).fill_(smooth_value)
        one_hot.scatter_(1, target.unsqueeze(1), confidence)

        log_prob = F.log_softmax(pred, dim=-1)
        loss = -torch.sum(one_hot * log_prob) / len(target)
        return loss
```

---

## ✅ 小结

1. **Transformer**完全基于Attention，抛弃了RNN
2. **Positional Encoding**给模型加入位置信息
3. Encoder用**Self-Attention**，Decoder用**Masked Self-Attention + Cross-Attention**
4. **LayerNorm**和**残差连接**保证训练的稳定性
5. 奠定了BERT、GPT等预训练模型的基础

---

_继续学习：下一章「VAE自编码器」_
