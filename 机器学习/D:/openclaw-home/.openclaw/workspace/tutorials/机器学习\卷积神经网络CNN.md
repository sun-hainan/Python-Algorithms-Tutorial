# 卷积神经网络CNN

_图像处理的核心深度学习模型_

---

## 📖 学习目标

1. 理理解卷积神经网络的核心概念
2. 掌握卷积层、池化层的工作原理
3. 能用 CNN 处理图像分类问题

---

## 第一部分：CNN 的核心思想

### 🎯 为什么需要 CNN？

```
全连接网络处理图像的问题：
- 100x100 的图像 = 10000 个像素
- 如果第一层有 1000 个神经元，需要 1000万 个参数！
- 无法利用图像的空间结构

CNN 的解决方案：
- 局部连接：只看一小块区域
- 权重共享：同一个卷积核在整张图上共享
```

---

## 第二部分：核心组件

### 1. 卷积层 (Convolutional Layer)

```python
import numpy as np

def conv2d(image, kernel, stride=1, padding=0):
    """
    二维卷积

    image: 输入图像 (H, W)
    kernel: 卷积核 (K, K)
    stride: 步长
    padding: 填充
    """
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')

    h, w = image.shape
    k = kernel.shape[0]

    out_h = (h - k) // stride + 1
    out_w = (w - k) // stride + 1
    output = np.zeros((out_h, out_w))

    for i in range(0, h - k + 1, stride):
        for j in range(0, w - k + 1, stride):
            output[i // stride, j // stride] = np.sum(
                image[i:i+k, j:j+k] * kernel
            )

    return output


# 示例：边缘检测卷积核
image = np.array([
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0]
])

# Sobel 边缘检测核
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
result = conv2d(image, sobel_x)
print(result)
```

### 2. 池化层 (Pooling Layer)

```python
def max_pooling(image, pool_size=2, stride=2):
    """
    最大池化：取区域内的最大值
    """
    h, w = image.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    output = np.zeros((out_h, out_w))

    for i in range(0, h - pool_size + 1, stride):
        for j in range(0, w - pool_size + 1, stride):
            output[i // stride, j // stride] = np.max(
                image[i:i+pool_size, j:j+pool_size]
            )

    return output


def avg_pooling(image, pool_size=2, stride=2):
    """平均池化：取区域内的平均值"""
    h, w = image.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    output = np.zeros((out_h, out_w))

    for i in range(0, h - pool_size + 1, stride):
        for j in range(0, w - pool_size + 1, stride):
            output[i // stride, j // stride] = np.mean(
                image[i:i+pool_size, j:j+pool_size]
            )

    return output
```

### 3. 全连接层 (Fully Connected Layer)

```python
class FullyConnected:
    """全连接层"""
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros(output_size)

    def forward(self, x):
        return np.dot(x, self.weights) + self.bias
```

---

## 第三部分：CNN 架构示例

```
LeNet-5 结构：

输入 (32x32) → Conv(6, 5x5) → AvgPool(2x2) → Conv(16, 5x5) → AvgPool(2x2)
→ FC(120) → FC(84) → FC(10) → 输出
```

---

## 第四部分：PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 第一个卷积块
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1通道输入，32通道输出
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二个卷积块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Conv -> ReLU -> Pool
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))

        # Flatten
        x = x.view(-1, 64 * 7 * 7)

        # FC -> ReLU -> Dropout -> FC
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 训练
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

---

## 第五部分：经典 CNN 架构

### LeNet (1998)

```
输入 → Conv(6) → Pool → Conv(16) → Pool → FC(120) → FC(84) → FC(10)
用于手写数字识别
```

### AlexNet (2012)

```
输入 → Conv(96, 11x11) → Pool → Conv(256, 5x5) → Pool
→ Conv(384, 3x3) × 2 → Conv(256, 3x3) → Pool
→ FC(4096) → FC(4096) → FC(1000)
ImageNet 竞赛冠军，深度学习复兴标志
```

### ResNet (2015)

```python
class ResidualBlock(nn.Module):
    """残差块：引入跳跃连接"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, stride)

    def forward(self, x):
        residual = self.shortcut(x)
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        return torch.relu(residual + x)
```

---

## 第六部分：名词解释

### 卷积核 (Kernel/Filter)

**定义：** 一个小的权重矩阵，在图像上滑动进行特征提取。

### 步长 (Stride)

**定义：** 卷积核每次移动的像素数。

### 填充 (Padding)

**定义：** 在图像边缘填充像素，常用 0 填充。

### 感受野 (Receptive Field)

**定义：** 输出特征图上的一个像素对应输入图像的区域大小。

---

## ✅ 小结

1. **CNN**通过局部连接和权重共享大幅减少参数量
2. 核心组件：**卷积层**（特征提取）、**池化层**（降维）、**全连接层**（分类）
3. 经典架构：LeNet → AlexNet → VGG → ResNet
4. ResNet 通过残差连接解决了深层网络训练难题
5. 广泛应用于：图像分类、目标检测、语义分割

---

_继续学习：下一章「RNN与LSTM」_
