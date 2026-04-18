# BP反向传播

_训练神经网络的核心算法_

---

## 📖 学习目标

1. 理理解反向传播的原理
2. 掌握梯度计算和链式法则
3. 能用代码实现完整的神经网络训练

---

## 第一部分：什么是反向传播？

### 🎯 核心思想

**正向传播算输出，反向传播算梯度。**

```
前向传播（计算输出）：
输入 → 隐藏层1 → 隐藏层2 → 输出

反向传播（计算梯度）：
输出误差 ← 反向传递 ← 计算每层梯度 ← 更新参数
```

---

## 第二部分：链式法则

```python
# 链式法则示例
# y = f(g(x))
# dy/dx = dy/dg * dg/dx

# 神经网络中：
# loss = L(y_pred, y_true)
# y_pred = sigmoid(w*x + b)
# dl/dw = dl/dy_pred * dy_pred/dz * dz/dw
```

---

## 第三部分：代码实现

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.weights = []
        self.biases = []
        self.activations = []

        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros(layers[i+1])
            self.weights.append(w)
            self.biases.append(b)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def forward(self, X):
        """前向传播"""
        self.activations = [X]
        self.z_values = []

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(self.activations[-1], w) + b
            self.z_values.append(z)

            if i < len(self.weights) - 1:
                a = self.relu(z)  # 隐藏层用 ReLU
            else:
                a = self.sigmoid(z)  # 输出层用 Sigmoid
            self.activations.append(a)

        return self.activations[-1]

    def backward(self, y, lr=0.1):
        """反向传播"""
        m = y.shape[0]
        n_layers = len(self.weights)

        # 输出层误差
        delta = self.activations[-1] - y

        # 反向传播
        for i in range(n_layers - 1, -1, -1):
            w = self.weights[i]
            a = self.activations[i]

            # 梯度
            dw = np.dot(a.T, delta) / m
            db = np.sum(delta, axis=0) / m

            # 传播到上一层
            if i > 0:
                delta = np.dot(delta, w.T) * self.relu_derivative(self.z_values[i-1])

            # 更新参数
            self.weights[i] -= lr * dw
            self.biases[i] -= lr * db

    def train(self, X, y, epochs=1000, lr=0.1):
        """训练"""
        for epoch in range(epochs):
            # 前向传播
            output = self.forward(X)

            # 反向传播
            self.backward(y, lr)

            if epoch % 100 == 0:
                loss = -np.mean(y * np.log(output + 1e-8) + (1 - y) * np.log(1 - output + 1e-8))
                print(f"Epoch {epoch}: loss = {loss:.4f}")

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)


# 测试：XOR 问题（多层感知机解决）
X_xor = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y_xor = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork([2, 4, 1])  # 隐藏层有4个神经元
nn.train(X_xor, y_xor, epochs=5000)

print("\n测试 XOR：")
for x in X_xor:
    print(f"{x} -> {nn.predict([x])[0]}")
```

---

## 第四部分：PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 定义网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# 训练
model = Net()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
y = torch.FloatTensor([[0], [1], [1], [0]])

for epoch in range(1000):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

# 预测
model.eval()
with torch.no_grad():
    pred = model(X)
    print(pred.round())
```

---

## 第五部分：常见问题

### 梯度消失

```python
# 原因：激活函数导数小于1，连乘后梯度趋近0
# 解决：使用 ReLU、Batch Normalization、残差连接

# ReLU 导数：max(0, z) -> 1 if z > 0 else 0
```

### 梯度爆炸

```python
# 原因：梯度连乘后变得很大
# 解决：梯度裁剪、Batch Normalization

# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 过拟合

```python
# 解决：正则化、Dropout、Early Stopping

# Dropout 示例
class NetWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4, 1)
```

---

## 第六部分：名词解释

### 前向传播

**定义：** 数据从输入层流经隐藏层到输出层的过程。

### 反向传播

**定义：** 将输出误差反向传播，计算每个参数的梯度。

### 损失函数

**定义：** 衡量预测值与真实值差距的函数。

```python
# 回归常用：MSE
# 分类常用：Cross Entropy
```

### 学习率

**定义：** 每次参数更新的步长。

---

## ✅ 小结

1. **反向传播**是训练神经网络的核心算法
2. 基于链式法则计算梯度
3. 过程：前向传播 → 计算误差 → 反向传播 → 更新参数
4. 常见问题：梯度消失/爆炸、过拟合
5. 解决方案：ReLU、BatchNorm、Dropout、梯度裁剪

---

_继续学习：下一章「卷积神经网络CNN」_
