# BP反向传播

_训练神经网络的核心算法_

---

## 第一部分：什么是反向传播？

### 核心思想

**正向传播算输出，反向传播算梯度。**

```
前向传播：输入 → 隐藏层1 → 隐藏层2 → 输出
反向传播：输出误差 ← 反向传递 ← 计算每层梯度 ← 更新参数
```

---

## 第二部分：代码实现

```python
class NeuralNetwork:
    def __init__(self, layers):
        self.weights = [np.random.randn(layers[i], layers[i+1]) * 0.1 
                        for i in range(len(layers)-1)]
        self.biases = [np.zeros(layers[i+1]) 
                       for i in range(len(layers)-1)]

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, X):
        self.activations = [X]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(self.activations[-1], w) + b
            a = self.sigmoid(z)
            self.activations.append(a)
        return self.activations[-1]

    def backward(self, y, lr=0.1):
        m = y.shape[0]
        delta = self.activations[-1] - y
        for i in range(len(self.weights)-1, -1, -1):
            dw = np.dot(self.activations[i].T, delta) / m
            db = np.sum(delta, axis=0) / m
            delta = np.dot(delta, self.weights[i].T) * self.activations[i] * (1 - self.activations[i])
            self.weights[i] -= lr * dw
            self.biases[i] -= lr * db
```

---

## 第三部分：PyTorch 实现

```python
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
```

---

## 小结

1. **反向传播**是训练神经网络的核心算法
2. 基于链式法则计算梯度
3. 常见问题：梯度消失/爆炸、过拟合

---

_继续学习：下一章「卷积神经网络CNN」_
