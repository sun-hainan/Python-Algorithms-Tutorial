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
    │      Feed Forward                     │
    │         ↓                             │
    │      Add & LayerNorm                  │
    │         ↓                             │
    └──→ (重复N次)                            │
                                           │
    ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←│
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
    """
    位置编码

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    pos: 位置（0, 1, 2, ...）
    i: 维度索引（0, 1, 2, ..., d_model-1）
    """
    PE = np.zeros((seq_len, d_model))

    # 位置
    positions = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)

    # 维度
    indices = np.arange(d_model)[np.newaxis, :]  # (1, d_model)

    # 计算编码
    angles = positions / np.power(10000, 2 * indices / d_model)

    # 偶数维度用 sin，奇数维度用 cos
    PE[:, 0::2] = np.sin(angles[:, 0::2])
    PE[:, 1::2] = np.cos(angles[:, 1::2])

    return PE


# 测试
pe = positional_encoding(10, 4)
print("位置编码（前5个位置）:")
print(pe[:5])
```

### 为什么用三角函数？

```
优点1：可以表示相对位置
  sin(pos+k) 和 cos(pos+k) 可以写成 sin(pos) 和 cos(pos) 的线性组合

优点2：模型可以看到任意位置的编码
  即使训练时没有见过某个位置，模型也能泛化

优点3：不同位置的编码不同
  PE(0) ≠ PE(1) ≠ PE(2) ≠ ...
```

---

## 第三部分：Encoder

### 单个 Encoder 层

```python
class EncoderLayer(nn.Module):
    """单个 Encoder 层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # 多头自注意力
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-Attention + 残差 + LayerNorm
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed Forward + 残差 + LayerNorm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x
```

### LayerNorm vs BatchNorm

```python
# BatchNorm：对每个特征在一个batch上归一化
# LayerNorm：对一个样本的所有特征归一化

# BatchNorm（NLP中很少用）：
# [batch, seq_len, d_model]
# 在 d_model 维度上归一化，但对 batch 和 seq 方向计算均值/方差

# LayerNorm（NLP中常用）：
# 对每个样本 [1, seq_len, d_model]
# 在 d_model 维度上归一化
# 更稳定，不依赖 batch size
```

---

## 第四部分：Decoder

### 单个 Decoder 层

```python
class DecoderLayer(nn.Module):
    """单个 Decoder 层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # 1. 掩码自注意力（不能看到未来）
        self.masked_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)

        # 2. 交叉注意力（看 Encoder 输出）
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)

        # 3. 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 1. 掩码自注意力
        # 每个词只能看到它之前的所有词，不能"偷看"未来的词
        mask_attn_output, _ = self.masked_attention(x, x, x, tgt_mask)
        x = self.norm1(x + mask_attn_output)

        # 2. 交叉注意力
        # Query 来自 Decoder，Key/Value 来自 Encoder
        cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + cross_attn_output)

        # 3. 前馈网络
        ffn_output = self.ffn(x)
        x = self.norm3(x + ffn_output)

        return x
```

### 掩码机制（Masking）

```python
def create_causal_mask(seq_len):
    """
    创建下三角掩码（防止看到未来）

    对于位置 i，只能看到 0, 1, ..., i 的内容
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype('uint8')
    # 上三角为 1（要mask掉），下三角为 0（可见）
    return mask


def create_padding_mask(seq):
    """
    创建填充掩码（忽略填充位置）
    """
    return (seq == 0).astype('uint8')
```

---

## 第五部分：完整 Transformer

```python
class Transformer(nn.Module):
    """完整的 Transformer 模型"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512,
                 num_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()

        # 词嵌入
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 输出层
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encoder
        encoder_output = self.src_embedding(src)
        encoder_output = self.pos_encoding(encoder_output)

        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, src_mask)

        # Decoder
        decoder_output = self.tgt_embedding(tgt)
        decoder_output = self.pos_encoding(decoder_output)

        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output,
                                  src_mask, tgt_mask)

        # 输出
        output = self.fc_out(decoder_output)

        return output
```

---

## 第六部分：训练技巧

### Label Smoothing

```python
class LabelSmoothingLoss(nn.Module):
    """标签平滑损失"""
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        smooth_value = self.smoothing / (self.num_classes - 1)

        # 将真实标签的概率设为 confidence，其他标签的概率设为 smooth_value
        one_hot = torch.zeros_like(pred).fill_(smooth_value)
        one_hot.scatter_(1, target.unsqueeze(1), confidence)

        # 计算交叉熵
        log_prob = F.log_softmax(pred, dim=-1)
        loss = -torch.sum(one_hot * log_prob) / len(target)

        return loss
```

### Warmup 学习率

```python
def get_lrscheduler(optimizer, warmup_steps, total_steps):
    """学习率预热+衰减"""
    def lr_lambda(step):
        if step < warmup_steps:
            # 线性预热
            return float(step) / float(max(1, warmup_steps))
        else:
            # 余弦衰减
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

---

## ✅ 小结

1. **Transformer**完全基于Attention，抛弃了RNN
2. **Positional Encoding**给模型加入位置信息
3. Encoder用**Self-Attention**，Decoder用**Masked Self-Attention + Cross-Attention**
4. **LayerNorm**和**残差连接**保证训练的稳定性
5. Label Smoothing和Learning Rate Warmup是常用训练技巧
6. 奠定了BERT、GPT等预训练模型的基础

---

_继续学习：下一章「VAE自编码器」_
