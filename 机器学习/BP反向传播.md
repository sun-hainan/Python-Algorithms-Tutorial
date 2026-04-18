# BP反向传播

> _让多层神经网络学习的核心算法——链式法则的精妙应用_

---

## 🎯 先看一个生活中的例子

### 例子：调整投篮力度

假设你在练习投篮：

```
1. 你瞄准篮筐，投出一球
2. 球偏右了，撞到框弹出
3. 你分析：偏右 → 说明力气太大了 or 角度不对
4. 你调整：下次少用点力，往左偏一点
5. 再投，又偏左了
6. 再调整：稍微增加力气
7. 重复... 直到投进！
```

**关键：你不知道具体应该调整多少力气，但通过观察结果，慢慢调整，最终学会了投篮！**

---

## 🤔 BP反向传播是做什么的？

### 前向传播 vs 反向传播

```
前向传播（Forward Pass）：
输入 ──→ 网络 ──→ 输出
    就像投篮，看结果

反向传播（Backward Pass）：
误差 ←─── 反向传递 ←─── 计算梯度
    就像分析为什么偏了，计算应该怎么调整
```

### 为什么要反向传播？

```
单层感知机：直接算梯度，手动更新

多层神经网络：
- 输入层 → 隐藏层1 → 隐藏层2 → ... → 输出层
- 每一层都可能有很多神经元
- 最后一层的误差，怎么传递到第一层？
- 这就是 BP 反向传播解决的问题！
```

---

## 📐 链式法则：BP的数学基础

### 什么是链式法则？

```
如果 y = f(g(x))，那么：
    dy/dx = df/dg × dg/dx

换句话说：外层函数的导数 × 内层函数的导数
```

### 链式法则的例子

```
假设：
    y = (x² + 1)³

令 u = x² + 1，则 y = u³

dy/du = 3u²
du/dx = 2x

所以：
dy/dx = dy/du × du/dx = 3u² × 2x = 3(x²+1)² × 2x = 6x(x²+1)²
```

### 链式法则在神经网络中的应用

```
神经网络有很多层：
    输入 x → L1 → L2 → L3 → 输出 y

计算 ∂L/∂w₁：
    ∂L/∂w₁ = ∂L/∂y × ∂y/∂L3 × ∂L3/∂L2 × ∂L2/∂L1 × ∂L1/∂w₁
           = 链式法则把所有层的导数连乘起来！
```

---

## 🏗️ 两层神经网络的结构

### 网络图示

```
输入层(2)     隐藏层(3)     输出层(1)

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

### 数学公式

```
隐藏层：
    z₁ = W₁x + b₁        (线性组合)
    h = σ(z₁)            (激活函数，如 sigmoid 或 ReLU)

输出层：
    z₂ = W₂h + b₂        (线性组合)
    ŷ = σ(z₂)            (激活函数)

损失函数：
    L = (y - ŷ)²         (均方误差)
```

---

## 🧮 反向传播数学推导

### 以单个样本为例

```
给定：
- 输入：x = (x₁, x₂)
- 真实标签：y
- 隐藏层激活：h = (h₁, h₂, h₃)
- 输出：ŷ

目标：计算 ∂L/∂W₁ 和 ∂L/∂W₂ 等参数的梯度
```

### 第一步：前向传播（从前往后算）

```
1. 隐藏层线性组合：
   z₁ = W₁x + b₁

2. 隐藏层激活：
   h = σ(z₁)

3. 输出层线性组合：
   z₂ = W₂h + b₂

4. 输出激活：
   ŷ = σ(z₂)

5. 计算损失：
   L = (y - ŷ)²
```

### 第二步：反向传播（从后往前算）

#### 计算 ∂L/∂W₂

```
L = (y - ŷ)²
ŷ = σ(z₂)
z₂ = W₂h + b₂

链式法则：
∂L/∂W₂ = ∂L/∂ŷ × ∂ŷ/∂z₂ × ∂z₂/∂W₂

逐个计算：
1. ∂L/∂ŷ = 2(ŷ - y)

2. ∂ŷ/∂z₂ = σ'(z₂) = ŷ(1 - ŷ)      [Sigmoid 导数]

3. ∂z₂/∂W₂ = h                        [z₂ = W₂h + b₂]

所以：
∂L/∂W₂ = 2(ŷ - y) × ŷ(1 - ŷ) × h
```

#### 计算 ∂L/∂b₂

```
∂L/∂b₂ = ∂L/∂ŷ × ∂ŷ/∂z₂ × ∂z₂/∂b₂
       = 2(ŷ - y) × ŷ(1 - ŷ) × 1
```

#### 计算 ∂L/∂W₁

```
∂L/∂W₂ 是输出层的梯度。
现在要把误差传递给隐藏层。

∂L/∂h = ∂L/∂z₂ × ∂z₂/∂h = δ₂ × W₂

其中 δ₂ = 2(ŷ - y) × ŷ(1 - ŷ)（输出层误差）

然后传回上一层：
∂L/∂z₁ = ∂L/∂h × ∂h/∂z₁ = δ₂W₂ × σ'(z₁)

∂L/∂W₁ = ∂L/∂z₁ × ∂z₁/∂W₁ = δ₂W₂σ'(z₁) × x
```

---

## 💻 代码实现

### 两层神经网络类

```python
import numpy as np

class NeuralNetwork:
    """两层神经网络"""

    def __init__(self, n_input, n_hidden, n_output, learning_rate=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.lr = learning_rate

        # 初始化权重（Xavier 初始化）
        self.W1 = np.random.randn(n_input, n_hidden) * np.sqrt(2.0 / n_input)
        self.b1 = np.zeros(n_hidden)

        self.W2 = np.random.randn(n_hidden, n_output) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros(n_output)

    def sigmoid(self, z):
        """Sigmoid 激活函数"""
        z = np.clip(z, -500, 500)  # 防止溢出
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        """Sigmoid 导数：已知激活值 a，求导数值"""
        return a * (1 - a)

    def forward(self, X):
        """
        前向传播

        X: (n_samples, n_input)
        返回: 输出 (n_samples, n_output)
        """
        # 隐藏层
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        # 输出层
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    def backward(self, X, y):
        """
        反向传播

        X: (n_samples, n_input)
        y: (n_samples,) 真实标签
        """
        n = len(y)

        # ===== 第1步：计算输出层误差 =====
        # δ₂ = (ŷ - y) × σ'(z₂)
        delta2 = (self.a2 - y.reshape(-1, 1)) * self.sigmoid_derivative(self.a2)

        # ===== 第2步：计算 W2 和 b2 的梯度 =====
        dW2 = np.dot(self.a1.T, delta2) / n
        db2 = np.sum(delta2, axis=0) / n

        # ===== 第3步：误差传递到隐藏层 =====
        # δ₁ = δ₂ × W₂ × σ'(z₁)
        delta1 = np.dot(delta2, self.W2.T) * self.sigmoid_derivative(self.a1)

        # ===== 第4步：计算 W1 和 b1 的梯度 =====
        dW1 = np.dot(X.T, delta1) / n
        db1 = np.sum(delta1, axis=0) / n

        # ===== 第5步：更新参数 =====
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def fit(self, X, y, epochs=1000, verbose=True):
        """训练神经网络"""
        for epoch in range(epochs):
            # 前向传播
            y_pred = self.forward(X)

            # 反向传播
            self.backward(X, y)

            # 打印进度
            if verbose and (epoch + 1) % 100 == 0:
                loss = np.mean((y_pred.flatten() - y) ** 2)
                accuracy = np.mean((y_pred.flatten() >= 0.5).astype(int) == y)
                print(f"Epoch {epoch+1:4d}: loss={loss:.4f}, accuracy={accuracy:.2%}")

        return self

    def predict(self, X):
        """预测"""
        return (self.forward(X).flatten() >= 0.5).astype(int)
```

---

## 🧪 完整例子：XOR 问题

### XOR 为什么单层感知机解决不了？

```python
# XOR 数据
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 1, 1, 0])

print("=== XOR 问题 ===")
print("真值表：")
print("x1 x2 | y")
print("---------")
for i in range(4):
    print(f"{X[i][0]}  {X[i][1]}  | {y[i]}")
```

### 用两层神经网络解决 XOR

```python
# 创建神经网络：2个输入，4个隐藏神经元，1个输出
nn = NeuralNetwork(n_input=2, n_hidden=4, n_output=1, learning_rate=1.0)

# 训练
print("\n=== 训练过程 ===")
nn.fit(X, y, epochs=5000, verbose=True)

# 测试
print("\n=== 预测结果 ===")
predictions = nn.predict(X)
for i in range(4):
    print(f"输入: {X[i]} -> 预测: {predictions[i]} (真实: {y[i]})")

print(f"\n准确率: {np.mean(predictions == y):.2%}")
```

### XOR 的决策边界可视化

```python
import matplotlib.pyplot as plt

def plot_xor_decision_boundary(nn, X, y):
    """画 XOR 的决策边界"""
    h = 0.01  # 网格步长
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = nn.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=50)
    plt.title('XOR 决策边界（两层神经网络）')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

plot_xor_decision_boundary(nn, X, y)
```

---

## ⚠️ 梯度消失和梯度爆炸

### 问题描述

```
反向传播时，梯度从后往前传：

∂L/∂W₁ = ∂L/∂ŷ × ∂ŷ/∂h₃ × ∂h₃/∂h₂ × ... × ∂h₁/∂W₁

如果每层的激活函数导数都小于 1（如 Sigmoid 最大 0.25）：
梯度会指数级衰减 → 梯度消失！

如果激活函数导数都大于 1：
梯度会指数级增长 → 梯度爆炸！
```

### Sigmoid 的问题

```python
# Sigmoid 导数的最大值
print(f"Sigmoid 导数最大值: {max(0.25)}")  # 0.25（在 z=0 时）

# 5 层之后
print(f"5层之后梯度: {0.25**5:.6f}")  # ≈ 0.0001，前几层几乎学不到
```

### 解决方案：ReLU

```python
def relu(z):
    """ReLU: max(0, z)"""
    return np.maximum(0, z)

def relu_derivative(z):
    """ReLU 导数"""
    return (z > 0).astype(float)

# ReLU 导数：z > 0 时为 1，z < 0 时为 0
# 不会梯度衰减！
```

---

## 🔑 关键公式总结

### 常用激活函数的导数

| 激活函数 | 表达式 | 导数 |
|---------|--------|------|
| Sigmoid | σ(z) = 1/(1+e^(-z)) | σ'(z) = σ(z)(1-σ(z)) |
| Tanh | tanh(z) | tanh'(z) = 1 - tanh²(z) |
| ReLU | max(0, z) | 0 (z<0), 1 (z>0) |
| Leaky ReLU | max(0.01z, z) | 0.01 (z<0), 1 (z>0) |

### BP 反向传播核心公式

```
输出层误差：
    δₗ = (ŷ - y) ⊙ σ'(zₗ)

隐藏层误差：
    δₗ = (Wₗ₊₁ᵀδₗ₊₁) ⊙ σ'(zₗ)

梯度：
    ∂L/∂Wₗ = aₗ₋₁ᵀδₗ
    ∂L/∂bₗ = Σδₗ
```

---

## ✅ 本章小结

| 概念 | 解释 |
|------|------|
| 前向传播 | 输入→隐藏层→输出，计算预测值 |
| 反向传播 | 从输出层开始，计算每个参数的梯度 |
| 链式法则 | BP 的数学基础 |
| 梯度 | 损失函数对参数的偏导数 |
| 梯度消失 | 链式相乘导致梯度太小，前层学不到 |
| 梯度爆炸 | 链式相乘导致梯度太大，参数剧烈振荡 |
| 初始化 | Xavier/He 初始化，避免梯度问题 |

---

## 🔗 继续学习

现在你已经理解了神经网络的核心训练算法。接下来让我们学习专门处理图像的卷积神经网络 CNN！

👉 [卷积神经网络CNN](./卷积神经网络CNN.md)
