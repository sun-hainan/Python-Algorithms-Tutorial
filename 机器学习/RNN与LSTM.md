# RNN与LSTM

_处理序列数据的神经网络_

---

## 第一部分：RNN 的问题

### 梯度消失 / 梯度爆炸

```
RNN 反向传播时，梯度需要沿时间反向传播
如果序列很长，梯度会指数级衰减（梯度消失）或爆炸（梯度爆炸）
```

### 长期依赖问题

```
"今天天气很好，我们去公园...
（中间有1000个词）...
所以我们最后在___野餐"

RNN 很难记住"今天"这个信息
```

---

## 第二部分：LSTM 结构

### 三个门

| 门 | 作用 |
|------|------|
| 遗忘门 | 决定丢弃什么信息 |
| 输入门 | 决定存储什么信息 |
| 输出门 | 决定输出什么 |

### PyTorch 实现

```python
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        return self.fc(h_n[-1])
```

---

## 第三部分：GRU（门控循环单元）

```python
class GRUCell:
    """GRU：比LSTM更简单，但效果相当"""
    def forward(self, x, h_prev):
        r = sigmoid(np.dot(self.Wr, np.vstack([h_prev, x])))
        z = sigmoid(np.dot(self.Wz, np.vstack([h_prev, x])))
        h_tilde = tanh(np.dot(self.Wh, np.vstack([r * h_prev, x])))
        h = (1 - z) * h_prev + z * h_tilde
        return h
```

---

## 小结

1. **RNN**能处理序列数据，但有梯度消失问题
2. **LSTM**通过门控机制解决长期依赖
3. **GRU**是LSTM的简化版，效果相当

---

_继续学习：下一章「Attention注意力机制」_
