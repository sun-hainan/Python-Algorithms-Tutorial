# RNN与LSTM

_让神经网络拥有"记忆"_

---

## 📖 学习目标

1. 理解为什么需要处理序列数据
2. 掌握RNN的基本结构和前向传播
3. 理解决策树 vs 神经网络的区别
4. 理解RNN的训练困难（梯度消失/爆炸）
5. 掌握LSTM的门控机制

---

## 第一部分：为什么需要RNN？

### 序列数据无处不在

```
文本：    "今天天气很好，我们去...（后面是什么？）"
          ↑记住前面的内容，才能预测

语音：    音频信号是连续的，前面影响后面

视频：    每一帧都依赖前后的帧

股票：    今天的价格和历史价格相关
```

### 全连接网络的问题

```
输入：固定长度向量
输出：固定长度向量

但序列数据：
- 长度可变
- 每个位置的含义取决于上下文
```

---

## 第二部分：RNN 基本结构

### 循环结构

```
标准神经网络：
  输入 → 隐藏层 → 输出

RNN（循环）：
  输入 → 隐藏层 ↔
          ↑
          （接收上一步的隐藏状态）
```

### 展开图

```
时间步 t=0:    x₀ ──→ [RNN] ──→ h₀ ──→ y₀
                     ↑
时间步 t=1:    x₁ ──→ [RNN] ──→ h₁ ──→ y₁
                     ↑
时间步 t=2:    x₂ ──→ [RNN] ──→ h₂ ──→ y₂

每个时间步的隐藏状态 hₜ 包含了之前所有时间步的信息！
```

### 数学公式

```
hₜ = tanh(Wxh·xₜ + Whh·hₜ₋₁ + bh)

yₜ = Why·hₜ + by

其中：
- xₜ: t时刻的输入
- hₜ₋₁: t-1时刻的隐藏状态
- hₜ: t时刻的隐藏状态
- Wxh, Whh, Why: 权重矩阵
- tanh: 激活函数
```

---

## 第三部分：RNN 前向传播

### 代码实现

```python
import numpy as np

class SimpleRNN:
    """简单RNN"""
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size

        # 初始化权重（使用 Xavier 初始化）
        scale = np.sqrt(2.0 / (input_size + hidden_size))

        self.Wxh = np.random.randn(hidden_size, input_size) * scale
        self.Whh = np.random.randn(hidden_size, hidden_size) * scale
        self.Why = np.random.randn(output_size, hidden_size) * scale

        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def tanh(self, z):
        return np.tanh(z)

    def forward_step(self, x, h_prev):
        """
        单步前向传播

        x: 当前输入 (input_size,)
        h_prev: 上一步的隐藏状态 (hidden_size,)
        """
        # 计算新的隐藏状态
        h = self.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h_prev) + self.bh)

        # 计算输出
        y = np.dot(self.Why, h) + self.by

        return h, y

    def forward(self, X):
        """
        处理整个序列

        X: 输入序列 (seq_len, input_size)
        返回: 所有隐藏状态, 所有输出
        """
        seq_len = len(X)

        # 初始化隐藏状态
        h = np.zeros(self.hidden_size)
        h_history = []
        y_history = []

        for t in range(seq_len):
            h, y = self.forward_step(X[t], h)
            h_history.append(h)
            y_history.append(y)

        return np.array(h_history), np.array(y_history)
```

---

## 第四部分：RNN 的梯度问题

### 时间步的反向传播（BPTT）

```
RNN 的反向传播需要沿时间展开：

∂L/∂W = Σₜ ∂Lₜ/∂W

每个时间步的梯度依赖前一个时间步：

∂hₜ/∂hₜ₋₁ = tanh' × Whh

如果 |tanh' × Whh| > 1：梯度爆炸
如果 |tanh' × Whh| < 1：梯度消失

tanh' 最大值 = 1（当z=0时）
所以当 Whh 较大时，梯度爆炸；较小时，梯度消失
```

### 梯度爆炸的表现

```python
# 梯度爆炸
gradients = []
for step in range(100):
    grad = compute_gradient()
    gradients.append(grad)
    if grad > 100:
        print(f"Step {step}: 梯度爆炸！")
        break
```

### 梯度消失的表现

```python
# 梯度消失
gradients = []
for step in range(100):
    grad = compute_gradient()
    gradients.append(grad)

print(f"Step 0 梯度: {gradients[0]}")
print(f"Step 99 梯度: {gradients[99]}")  # 可能接近 0！
```

### 长期依赖问题

```
问题：RNN 很难学习相隔很远的时间步之间的关系

例子：
"The cat, which already ate ..., was full."
"The cats, which already ate ..., were full."

单数和复数的主语决定谓语形式
但它们可能相隔几十个词！

RNN 在反向传播时：
- 梯度需要穿过所有中间层
- 每一步都可能被衰减
- 早期信号几乎传不过去！
```

---

## 第五部分：LSTM 详解

### LSTM 的核心思想

```
引入"门控"机制：
- 决定什么信息保留
- 决定什么信息丢弃
- 决定什么信息新增

就像一个智能的记忆系统！
```

### LSTM 结构图

```
                 ┌─────────────────────────────────────────┐
                 │                                         │
                 │    ┌───────────────────────────────┐  │
      hₜ₋₁ ─────┼───→│                               │←──┤ xₜ
                 │    │   ┌─────┐     ┌─────┐       │  │
                 │    │   │遗忘门│     │输入门│       │  │
                 │    │   └─────┘     └─────┘       │  │
                 │    │       ↓           ↓         │  │
                 │    │   ┌─────────────────┐       │  │
                 │    │   │                 │       │  │
                 │    │   │    Cell State   │←─────+──┤
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

```
1. 遗忘门（Forget Gate）：
   决定从细胞状态中丢弃什么信息

   fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)

   输出 0~1 之间，0 表示"完全丢弃"，1 表示"完全保留"

2. 输入门（Input Gate）：
   决定在细胞状态中存储什么新信息

   iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)
   C̃ₜ = tanh(Wc·[hₜ₋₁, xₜ] + bc)

3. 更新细胞状态：
   Cₜ = fₜ × Cₜ₋₁ + iₜ × C̃ₜ

   遗忘门乘以旧状态，丢弃不需要的
   加输入门乘以新候选状态，添加新信息

4. 输出门（Output Gate）：
   决定输出什么

   oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)
   hₜ = oₜ × tanh(Cₜ)
```

### 代码实现

```python
class LSTMCell:
    """LSTM 单元"""
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size

        # 三个门的权重和偏置
        self.Wf = np.random.randn(hidden_size, hidden_size + input_size) * 0.1
        self.Wi = np.random.randn(hidden_size, hidden_size + input_size) * 0.1
        self.Wc = np.random.randn(hidden_size, hidden_size + input_size) * 0.1
        self.Wo = np.random.randn(hidden_size, hidden_size + input_size) * 0.1

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, x, h_prev, C_prev):
        """
        单步前向传播

        x: 当前输入
        h_prev: 上一步的隐藏状态
        C_prev: 上一步的细胞状态
        """
        concat = np.vstack([h_prev, x])

        # 遗忘门：决定丢弃什么
        f = self.sigmoid(np.dot(self.Wf, concat) + self.bf)

        # 输入门：决定添加什么
        i = self.sigmoid(np.dot(self.Wi, concat) + self.bi)
        C_tilde = np.tanh(np.dot(self.Wc, concat) + self.bc)

        # 更新细胞状态
        C = f * C_prev + i * C_tilde

        # 输出门：决定输出什么
        o = self.sigmoid(np.dot(self.Wo, concat) + self.bo)
        h = o * np.tanh(C)

        return h, C


class LSTM:
    """LSTM 网络"""
    def __init__(self, input_size, hidden_size, output_size):
        self.cell = LSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.Why = np.random.randn(output_size, hidden_size) * 0.1
        self.by = np.zeros((output_size, 1))

    def forward(self, X):
        """
        处理整个序列

        X: (seq_len, input_size)
        """
        seq_len = len(X)

        # 初始化
        h = np.zeros((self.hidden_size, 1))
        C = np.zeros((self.hidden_size, 1))
        h_history = []
        y_history = []

        for t in range(seq_len):
            h, C = self.cell.forward(X[t], h, C)
            h_history.append(h)

            # 输出（如果需要）
            y = np.dot(self.Why, h) + self.by
            y_history.append(y)

        return h_history, y_history
```

---

## 第六部分：GRU（门控循环单元）

### LSTM 的简化版本

```
GRU 只用两个门：
- 更新门（类似 LSTM 的遗忘+输入门）
- 重置门（控制忽略上一隐藏状态的程度）

参数更少，训练更快，效果相当！
```

```python
class GRUCell:
    """GRU 单元"""
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size

        # 重置门和更新门
        self.Wr = np.random.randn(hidden_size, hidden_size + input_size) * 0.1
        self.Wz = np.random.randn(hidden_size, hidden_size + input_size) * 0.1
        self.Wh = np.random.randn(hidden_size, hidden_size + input_size) * 0.1

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, x, h_prev):
        concat = np.vstack([h_prev, x])

        # 重置门：决定忽略多少过去
        r = self.sigmoid(np.dot(self.Wr, concat))

        # 更新门：决定更新多少
        z = self.sigmoid(np.dot(self.Wz, concat))

        # 候选隐藏状态（使用重置门）
        concat_r = np.vstack([r * h_prev, x])
        h_tilde = np.tanh(np.dot(self.Wh, concat_r))

        # 新隐藏状态
        h = (1 - z) * h_prev + z * h_tilde

        return h
```

---

## 第七部分：PyTorch 实现

```python
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    """LSTM 文本分类器"""
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_classes):
        super().__init__()

        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # LSTM
        self.lstm = nn.LSTM(
            embed_size,                    # 输入维度
            hidden_size,                   # 隐藏层维度
            num_layers,                    # 层数
            batch_first=True,             # batch 在第一维
            dropout=0.5,                  # dropout 防止过拟合
            bidirectional=True             # 双向 LSTM
        )

        # 全连接分类层
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 因为双向

    def forward(self, x):
        # x: (batch_size, seq_len)

        # 词嵌入
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_size)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        # lstm_out: (batch_size, seq_len, hidden_size * 2)

        # 取最后时刻的输出（双向所以拼接）
        hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)  # (batch_size, hidden_size * 2)

        # 分类
        logits = self.fc(hidden)  # (batch_size, num_classes)

        return logits
```

---

## 第八部分：序列到序列（Seq2Seq）

### Encoder-Decoder 结构

```
Encoder (处理输入序列):
  "你好" → [RNN/LSTM] → 上下文向量 c

Decoder (生成输出序列):
  c → [RNN/LSTM] → "hello"
```

```python
class Encoder(nn.Module):
    """编码器"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        outputs, (h, c) = self.rnn(x)
        return h, c  # 传递最后的隐藏状态


class Decoder(nn.Module):
    """解码器"""
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h, c):
        output, (h, c) = self.rnn(x, (h, c))
        logits = self.fc(output)
        return logits, h, c


class Seq2Seq(nn.Module):
    """序列到序列模型"""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt):
        # 编码
        h, c = self.encoder(src)

        # 解码（teacher forcing）
        outputs = []
        decoder_input = tgt[:, 0:1]  # <start> token
        for t in range(tgt.size(1)):
            out, h, c = self.decoder(decoder_input, h, c)
            outputs.append(out)
            decoder_input = tgt[:, t:t+1]  # 下一步输入

        return torch.cat(outputs, dim=1)
```

---

## ✅ 小结

1. **RNN**能处理序列数据，通过隐藏状态传递信息
2. 但存在**梯度消失/爆炸**问题，难以学习长期依赖
3. **LSTM**通过门控机制解决长期依赖问题
4. **遗忘门、输入门、输出门**控制信息流动
5. **GRU**是LSTM的简化版，效果相当
6. **Seq2Seq**模型用于机器翻译等任务
7. 现在 Transformer 在很多任务上取代了 RNN

---

_继续学习：下一章「Attention注意力机制」_
