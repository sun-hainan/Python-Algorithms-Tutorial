# RNN与LSTM

_处理序列数据的神经网络_

---

## 📖 学习目标

1. 理理解RNN的工作原理和梯度问题
2. 掌握LSTM和GRU的结构
3. 能处理序列预测任务

---

## 第一部分：什么是RNN？

### 🎯 核心思想

**RNN 具有"记忆"能力，能处理序列数据。**

```
RNN 结构：

     h_t
      ↑
x_t → [RNN] → h_t ──→ output y_t
         ↑
x_{t-1} → [RNN] → h_{t-1}
         ↑
x_0 → [RNN] → h_0

循环：每个时间步的隐藏状态传递给下一个时间步
```

### 数学表达

```python
# RNN 前向传播
def rnn_step(x, h_prev, Wh,Wx, b):
    """一个时间步的计算"""
    h = np.tanh(np.dot(Wx, x) + np.dot(Wh, h_prev) + b)
    return h
```

---

## 第二部分：RNN 的问题

### 梯度消失 / 梯度爆炸

```python
# RNN 反向传播时，梯度需要沿时间反向传播
# 如果序列很长，梯度会指数级衰减（梯度消失）或爆炸（梯度爆炸）

# 梯度消失例子：
# dl/dh_0 = dl/dh_T * Π(dh_{t+1}/dh_t)
# 如果 dh_{t+1}/dh_t < 1，连乘后趋近0
```

### 长期依赖问题

```
"今天天气很好，我们去公园...
（中间有1000个词）...
所以我们最后在___野餐"

RNN 很难记住"今天"这个信息
```

---

## 第三部分：LSTM 结构

### 🎯 核心思想

**LSTM 通过"门控"机制解决长期依赖问题。**

```
LSTM 单元：

         ┌─────────────────────────────────┐
         │                                 │
    h_{t-1} ──┬──→ forget gate ───→ 乘 ──→ h_t
              │        │                    │
              │        ↓                    │
              │   [sigmoid]                 │
              │        │                    │
              │    input gate ───→ + ──────┤
              │        │    (cell state)    │
    x_t ──────┴──→ output gate              │
                         │                  │
                         ↓                  │
                    [sigmoid] → × → tanh ──┘
```

### 三个门

| 门 | 作用 | 公式 |
|------|------|------|
| 遗忘门 | 决定丢弃什么信息 | f = σ(W_f · [h_{t-1}, x_t] + b_f) |
| 输入门 | 决定存储什么信息 | i = σ(W_i · [h_{t-1}, x_t] + b_i) |
| 输出门 | 决定输出什么 | o = σ(W_o · [h_{t-1}, x_t] + b_o) |

---

## 第四部分：LSTM 代码实现

```python
import numpy as np

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size

        # 初始化权重
        self.Wf = np.random.randn(hidden_size, hidden_size + input_size) * 0.1
        self.Wi = np.random.randn(hidden_size, hidden_size + input_size) * 0.1
        self.Wc = np.random.randn(hidden_size, hidden_size + input_size) * 0.1
        self.Wo = np.random.randn(hidden_size, hidden_size + input_size) * 0.1

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x, h_prev, c_prev):
        """LSTM 单步前向传播"""
        concat = np.vstack((h_prev, x))

        # 遗忘门：决定丢弃哪些信息
        f = self.sigmoid(np.dot(self.Wf, concat) + self.bf)

        # 输入门：决定添加哪些新信息
        i = self.sigmoid(np.dot(self.Wi, concat) + self.bi)

        # 候选记忆
        c_tilde = np.tanh(np.dot(self.Wc, concat) + self.bc)

        # 更新 cell state
        c = f * c_prev + i * c_tilde

        # 输出门
        o = self.sigmoid(np.dot(self.Wo, concat) + self.bo)
        h = o * np.tanh(c)

        return h, c


class LSTM:
    def __init__(self, input_size, hidden_size):
        self.cell = LSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward_sequence(self, X):
        """处理整个序列"""
        n_steps = X.shape[0]
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        outputs = []

        for t in range(n_steps):
            h, c = self.cell.forward(X[t], h, c)
            outputs.append(h)

        return outputs, h
```

---

## 第五部分：GRU（门控循环单元）

### 更简洁的结构

```python
class GRUCell:
    """GRU：比LSTM更简单，但效果相当"""
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size

        # 重置门
        self.Wr = np.random.randn(hidden_size, hidden_size + input_size) * 0.1
        # 更新门
        self.Wz = np.random.randn(hidden_size, hidden_size + input_size) * 0.1
        # 候选隐藏状态
        self.Wh = np.random.randn(hidden_size, hidden_size + input_size) * 0.1

    def forward(self, x, h_prev):
        concat = np.vstack((h_prev, x))

        # 重置门
        r = sigmoid(np.dot(self.Wr, concat))

        # 更新门
        z = sigmoid(np.dot(self.Wz, concat))

        # 候选隐藏状态（使用重置门）
        concat_r = np.vstack((r * h_prev, x))
        h_tilde = tanh(np.dot(self.Wh, concat_r))

        # 新隐藏状态
        h = (1 - z) * h_prev + z * h_tilde

        return h
```

---

## 第六部分：PyTorch 实现

```python
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_size)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(embedded)

        # 取最后一个隐藏状态
        last_hidden = h_n[-1]  # (batch_size, hidden_size)

        # 分类
        logits = self.fc(last_hidden)
        return logits


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hidden = self.gru(embedded)
        return self.fc(hidden[-1])
```

---

## 第七部分：实际应用场景

### 📝 场景1：文本分类

```python
# 输入：文本序列
# 输出：情感标签（正面/负面）
```

### 🗣️ 场景2：机器翻译

```python
# Encoder (RNN/LSTM) 编码源语言
# Decoder (RNN/LSTM) 解码目标语言
```

### 📊 场景3：时间序列预测

```python
# 股票价格预测、天气预测
```

---

## 第八部分：名词解释

### Cell State

**定义：** LSTM 中的"记忆单元"，类似传送带，信息可以在其中流动。

### Hidden State

**定义：** RNN/LSTM 的输出，包含当前时间步的信息。

### 门控机制

**定义：** 通过 sigmoid 函数输出 0-1 的值，控制信息流动。

---

## ✅ 小结

1. **RNN**能处理序列数据，但有梯度消失问题
2. **LSTM**通过门控机制解决长期依赖
3. **GRU**是LSTM的简化版，效果相当
4. 适用于：文本、语音、时间序列等
5. 现在更多被 Transformer 架构取代（除短序列外）

---

_继续学习：下一章「Attention注意力机制」_
