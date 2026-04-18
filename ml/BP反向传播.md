# BP反向传播

_让神经网络学习的核心算法——链式法则的巧妙应用_

---

## 📖 学习目标

1. 理解前向传播和反向传播的完整流程
2. 掌握链式法则在神经网络中的应用
3. 推导并理解每一层的梯度计算
4. 理解梯度消失和梯度爆炸问题

---

## 第一部分：整体框架

### 前向传播 vs 反向传播

```
前向传播（Forward Pass）：
输入 → 隐藏层1 → 隐藏层2 → ... → 输出层
     计算每个神经元的值，逐层传递

反向传播（Backward Pass）：
输出误差 ← 反向传递 ← 计算梯度 ← 更新参数
     从输出层开始，计算每个参数对误差的"贡献"
```

### 形象比喻

```
想象你要调整一把狙击枪的参数来击中目标：

前向传播：扣扳机，看子弹打到哪里
反向传播：分析弹着点，计算每个参数应该怎么调

比如：
- 子弹偏右上方 → 往左下方调整瞄准镜
- 子弹偏左下方 → 往右上方调整瞄准镜
- 风太大 → 需要更大调整
```

---

## 第二部分：网络结构

### 两层神经网络示例

```
输入层(2)    隐藏层(3)    输出层(1)
                    
x₁ ─────────┬──→ h₁ ──┬──→ y ──→ ŷ
            │         │
x₂ ─────────┼──→ h₂ ──┘
            │
            └──→ h₃ ──┘

参数：
- W₁: 2×3 权重矩阵（输入→隐藏）
- b₁: 3×1 偏置向量（隐藏层）
- W₂: 3×1 权重矩阵（隐藏→输出）
- b₂: 标量（输出层）
```

### 激活函数

```python
def sigmoid(z):
    """Sigmoid: 0~1之间"""
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    """Sigmoid导数: a(1-a)"""
    return a * (1 - a)

def relu(z):
    """ReLU: max(0,z)"""
    return np.maximum(0, z)

def relu_derivative(z):
    """ReLU导数: z>0时为1，否则为0"""
    return (z > 0).astype(float)
```

---

## 第三部分：前向传播

### 数学推导

```
输入: x = (x₁, x₂)

隐藏层:
h = σ(W₁x + b₁)

具体计算:
h₁ = σ(w₁₁x₁ + w₂₁x₂ + b₁₁)
h₂ = σ(w₁₂x₁ + w₂₂x₂ + b₁₂)
h₃ = σ(w₁₃x₁ + w₂₃x₂ + b₁₃)

输出层:
y = σ(W₂h + b₂)

最终输出:
ŷ = σ(w₁₂₁h₁ + w₂₂₁h₂ + w₃₂₁h₃ + b₂)
```

### 代码实现

```python
def forward(self, X):
    """
    前向传播

    X: 输入 (n_samples, n_features)
    返回: 输出 (n_samples,)
    """
    n_samples = len(X)

    # 隐藏层: z₁ = W₁X + b₁
    self.z1 = np.dot(X, self.W1) + self.b1
    # 激活: a₁ = σ(z₁)
    self.a1 = sigmoid(self.z1)

    # 输出层: z₂ = W₂a₁ + b₂
    self.z2 = np.dot(self.a1, self.W2) + self.b2
    # 激活: a₂ = σ(z₂)
    self.a2 = sigmoid(self.z2)

    return self.a2
```

---

## 第四部分：损失函数

### 常用损失函数

```python
def mse_loss(y_true, y_pred):
    """均方误差: 用于回归"""
    return np.mean((y_true - y_pred) ** 2)

def cross_entropy_loss(y_true, y_pred):
    """交叉熵: 用于分类"""
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1-eps)
    return -np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))
```

### 本章使用 MSE

```
L = (1/n) × Σ(y_i - ŷ_i)²

对于单个样本：
L = (y - ŷ)²
```

---

## 第五部分：反向传播数学推导

### 核心思想：链式法则

```
链式法则：
如果 y = f(g(x))，则 dy/dx = f'(g(x)) × g'(x)

神经网络中的链式法则：
∂L/∂W₂ = ∂L/∂ŷ × ∂ŷ/∂z₂ × ∂z₂/∂W₂
```

### 第一步：输出层梯度

```
已知: L = (y - ŷ)² = (y - σ(z₂))²

∂L/∂ŷ = 2(ŷ - y)

∂ŷ/∂z₂ = σ'(z₂) = ŷ(1-ŷ)  [Sigmoid导数]

∂z₂/∂W₂ = h  [z₂ = W₂h + b₂，所以∂z₂/∂W₂ = h]

所以:
∂L/∂W₂ = ∂L/∂ŷ × ∂ŷ/∂z₂ × ∂z₂/∂W₂
       = 2(ŷ - y) × ŷ(1-ŷ) × h
```

```python
# 输出层误差
delta_out = (y_pred - y_true) * sigmoid_derivative(y_pred)
# 或简写
delta_out = y_pred - y_true  # 对于 MSE + Sigmoid

# W₂ 梯度
dW2 = np.dot(self.a1.T, delta_out) / n_samples
db2 = np.sum(delta_out) / n_samples
```

### 第二步：传播到隐藏层

```
我们需要计算 ∂L/∂h（误差对隐藏层输出的"敏感度"）

∂L/∂h = ∂L/∂z₂ × ∂z₂/∂h
       = δ₂ × W₂

然后:
∂L/∂z₁ = ∂L/∂h × ∂h/∂z₁
       = (∂L/∂h) × σ'(z₁)

∂z₁/∂W₁ = X
```

```python
# 反向传播到隐藏层
delta_h = np.dot(delta_out, self.W2.T) * sigmoid_derivative(self.a1)

# W₁ 梯度
dW1 = np.dot(X.T, delta_h) / n_samples
db1 = np.sum(delta_h, axis=0) / n_samples
```

---

## 第六部分：完整代码实现

```python
import numpy as np

class NeuralNetwork:
    """两层神经网络"""
    def __init__(self, n_input, n_hidden, n_output):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        # 初始化权重（使用 Xavier 初始化）
        # 权重不能太大也不能太小
        self.W1 = np.random.randn(n_input, n_hidden) * np.sqrt(2.0 / n_input)
        self.b1 = np.zeros(n_hidden)

        self.W2 = np.random.randn(n_hidden, n_output) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros(n_output)

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)  # 防止溢出
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        """a 是 sigmoid 的输出"""
        return a * (1 - a)

    def forward(self, X):
        """前向传播"""
        # 隐藏层
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        # 输出层
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    def backward(self, X, y_true, learning_rate):
        """
        反向传播

        X: 输入
        y_true: 真实标签
        learning_rate: 学习率
        """
        n_samples = len(y_true)

        # ===== 前向传播 =====
        y_pred = self.forward(X)

        # ===== Step 1: 输出层梯度 =====
        # δ₂ = (ŷ - y) × σ'(ŷ)
        delta_out = (y_pred - y_true) * self.sigmoid_derivative(y_pred)

        # ∂L/∂W₂ = a₁ᵀ × δ₂
        dW2 = np.dot(self.a1.T, delta_out) / n_samples
        db2 = np.sum(delta_out, axis=0) / n_samples

        # ===== Step 2: 隐藏层梯度 =====
        # δ₁ = δ₂ × W₂ᵀ × σ'(h)
        delta_h = np.dot(delta_out, self.W2.T) * self.sigmoid_derivative(self.a1)

        # ∂L/∂W₁ = Xᵀ × δ₁
        dW1 = np.dot(X.T, delta_h) / n_samples
        db1 = np.sum(delta_h, axis=0) / n_samples

        # ===== Step 3: 更新参数 =====
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs=1000, learning_rate=0.1, verbose=True):
        """训练神经网络"""
        for epoch in range(epochs):
            # 前向 + 反向
            self.backward(X, y, learning_rate)

            if verbose and (epoch + 1) % 100 == 0:
                y_pred = self.forward(X)
                loss = np.mean((y - y_pred) ** 2)
                accuracy = np.mean((y_pred >= 0.5).astype(int) == y)
                print(f"Epoch {epoch+1:4d}: loss={loss:.4f}, accuracy={accuracy:.2%}")

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)
```

---

## 第七部分：梯度消失与梯度爆炸

### 问题描述

```
在反向传播中，梯度需要从输出层传播回输入层：

∂L/∂W₁ = ∂L/∂ŷ × ∂ŷ/∂h × ... × ∂h₁/∂W₁

如果每层的激活函数导数都小于1（如Sigmoid导数最大0.25），
梯度会指数衰减 → 梯度消失！

如果每层的激活函数导数都大于1，
梯度会指数增长 → 梯度爆炸！
```

### Sigmoid 的问题

```python
# Sigmoid 导数最大值
max_derivative = 0.25  # 在 z=0 时

# 如果网络有 5 层
# 梯度会被衰减 0.25^5 = 0.00098 ≈ 0.1%
# 前面的层几乎学不到东西！
```

### 解决方案

```
1. 使用 ReLU 激活函数
   - 导数是 0 或 1
   - 避免了梯度衰减

2. 残差连接（ResNet）
   - 提供"高速公路"让梯度直接传回去

3. 批量归一化（BatchNorm）
   - 让每层的输入分布稳定

4. 梯度裁剪
   - 限制梯度的最大/最小值
```

---

## 第八部分：PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 定义网络
class Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# 创建网络
model = Net(2, 4, 1)

# 损失函数
criterion = nn.MSELoss()

# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练数据
X = torch.FloatTensor([[0,0], [0,1], [1,0], [1,1]])
y = torch.FloatTensor([[0], [1], [1], [0]])  # XOR

# 训练
for epoch in range(1000):
    optimizer.zero_grad()    # 清零梯度
    outputs = model(X)       # 前向传播
    loss = criterion(outputs, y)  # 计算损失
    loss.backward()          # 反向传播（自动计算梯度！）
    optimizer.step()        # 更新参数

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

# 预测
with torch.no_grad():
    predictions = model(X)
    print("\n预测结果:")
    for x, pred in zip(X, predictions):
        print(f"  {x.tolist()} -> {pred.item():.4f}")
```

---

## 第九部分：训练过程的形象理解

```
初始状态：随机权重
↓
前向传播：输入通过随机网络，产生随机输出
↓
计算误差：输出和真实标签的差距
↓
反向传播：误差信号传回网络
↓
更新参数：每个权重根据"自己的贡献"调整
↓
重复：直到网络学会正确映射

类比学习：
- 神经网络 = 学生
- 前向传播 = 做题
- 计算误差 = 批改试卷
- 反向传播 = 分析错题原因
- 更新参数 = 理解错在哪里，下次做对
```

---

## ✅ 小结

1. **反向传播**是训练神经网络的核心算法
2. 基于**链式法则**计算每个参数的梯度
3. 前向传播算输出，反向传播算梯度
4. 梯度 = 误差信号 × 激活导数 × 权重
5. 常见问题：**梯度消失**和**梯度爆炸**
6. 解决方案：ReLU、残差连接、BatchNorm、梯度裁剪
7. PyTorch 可以自动求导，简化实现

---

_继续学习：下一章「卷积神经网络CNN」_
