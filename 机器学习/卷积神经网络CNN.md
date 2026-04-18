# 卷积神经网络CNN

> _让计算机拥有"眼睛"的革命性技术_

---

## 🎯 先看一个生活中的例子

### 例子：你是怎么认出猫的？

当你在街上看到一只猫，你的眼睛和大脑是怎么工作的？

```
1. 你的眼睛看到的是一堆像素（图像）
2. 你的大脑开始分析：
   - 边缘（比如猫的轮廓）
   - 纹理（猫毛的质感）
   - 形状（尖尖的耳朵）
   - 颜色
3. 这些特征组合起来 → "哦，是一只猫！"
```

**CNN 的工作方式几乎和人类视觉系统一样！**

---

## 🤔 为什么全连接网络处理图像效果不好？

### 全连接网络处理图像的问题

假设我们有一张图片：

```
图片大小：100 × 100 像素（很小的图片！）
颜色：灰度图（每个像素1个值）

总像素数：100 × 100 = 10,000 个像素
```

如果第一个隐藏层有 1000 个神经元：

```
参数量 = 10,000 × 1,000 = 10,000,000（1000万！）
```

### CNN 的解决方案

```
两种核心思想：

1. 局部连接（Local Connectivity）
   - 每个神经元只连接一小块区域
   - 就像你一次只关注一小块图像，而不是整张图

2. 权重共享（Weight Sharing）
   - 同一个卷积核在整张图上滑动
   - 大幅减少参数量！
```

---

## 🔍 什么是卷积？

### 卷积的直观理解

**卷积 = 用一个"小窗口"在图像上滑动，每一步做加权求和**

```
卷积核（3×3）在图像上滑动：
    ┌─────────────────┐
    │ 1 2 3 0 1 2 3 0 │
    │ 0 1 2 3 0 1 2 3 │    ┌───┐
    │ 1 0 1 2 3 0 1 2 │    │1 0│ ← 卷积核
    │ 2 1 0 1 2 3 0 1 │    │0 1│
    │ 1 2 3 0 1 2 3 0 │    └───┘
    └─────────────────┘

每一步滑动，计算一小块区域的"加权和"
```

### 卷积操作的图解

```
输入图像 (5×5)：
┌─────────┐
│ 1 2 3 0 1 │
│ 0 1 2 3 0 │
│ 1 0 1 2 3 │
│ 2 1 0 1 2 │
│ 1 2 3 0 1 │
└─────────┘
       ×

卷积核 (3×3)：
┌─────────┐
│ 1 0 1 │
│ 0 1 0 │
│ 1 0 1 │
└─────────┘
       ↓

输出特征图 (3×3)：
┌─────────┐
│ ? ? ? │
│ ? ? ? │
│ ? ? ? │
└─────────┘

计算输出[0,0]：
= 1×1 + 2×0 + 3×1 + 0×0 + 1×1 + 2×0 + 1×1 + 0×0 + 1×1
= 1 + 0 + 3 + 0 + 1 + 0 + 1 + 0 + 1
= 7
```

---

## 💻 卷积操作代码

### NumPy 实现

```python
import numpy as np

def conv2d(image, kernel, stride=1, padding=0):
    """
    二维卷积

    参数:
        image: 输入图像 (H, W)
        kernel: 卷积核 (K, K)
        stride: 步长（每次滑动多少像素）
        padding: 边缘填充（周围补0的圈数）
    返回:
        输出特征图
    """
    # 如果有填充
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')

    # 计算输出尺寸
    h, w = image.shape
    k = kernel.shape[0]
    out_h = (h - k) // stride + 1
    out_w = (w - k) // stride + 1
    output = np.zeros((out_h, out_w))

    # 卷积操作：滑动窗口
    for i in range(0, h - k + 1, stride):
        for j in range(0, w - k + 1, stride):
            # 提取图像块
            image_patch = image[i:i+k, j:j+k]
            # 逐元素相乘后求和
            output[i // stride, j // stride] = np.sum(image_patch * kernel)

    return output


# 测试：边缘检测
image = np.array([
    [1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0],
])

# 垂直边缘检测核
vertical_kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

result = conv2d(image, vertical_kernel)
print("垂直边缘检测结果:")
print(result)
```

---

## 📐 卷积的三个关键概念

### 1. 填充（Padding）

**问题**：每次卷积后，图像会变小一圈

```
输入 5×5，卷积核 3×3，步长1：
- 输出 = (5-3+0)/1+1 = 3×3（变小了！）
- 边缘信息丢失
```

**解决方案：填充**

```
Same 卷积（填充后保持大小）：
- 输入 5×5，卷积核 3×3，填充1，步长1
- 输出 = (5-3+2×1)/1+1 = 5×5（保持大小）
```

### 2. 步长（Stride）

**步长 = 卷积核每次移动的像素数**

```
步长1：正常滑动，每像素移动一次
步长2：跳过一格，输出尺寸缩小一半
```

### 3. 多通道和多个卷积核

**RGB 图像 = 3 个通道**

```
卷积核也有 3 个通道！
输出 = R卷积 + G卷积 + B卷积 + 偏置

一个卷积核 → 一个特征图
多个卷积核 → 多个特征图
```

---

## 📐 池化层（Pooling）

### 为什么要池化？

```
1. 降维：减少计算量
2. 提取主要特征：减少冗余
3. 提供一定程度的平移不变性
```

### 最大池化（Max Pooling）

```
2×2 池化，步长2：

输入:              输出:
┌─────────┐        ┌───────┐
│ 1 3 2 1 │        │ 6  3 │
│ 6 2 1 5 │  →    │ 8  5 │
│ 8 1 3 2 │        └───────┘
│ 1 4 5 6 │
└─────────┘

每个 2×2 区域取最大值
```

### 平均池化（Average Pooling）

```
同样区域，取平均值
```

### 代码实现

```python
def max_pooling(image, pool_size=2, stride=2):
    h, w = image.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    output = np.zeros((out_h, out_w))

    for i in range(0, h - pool_size + 1, stride):
        for j in range(0, w - pool_size + 1, stride):
            patch = image[i:i+pool_size, j:j+pool_size]
            output[i // stride, j // stride] = np.max(patch)

    return output
```

---

## 🏗️ LeNet-5：第一个卷积神经网络

### LeNet-5 结构（1998年，用于手写数字识别）

```
输入 (32×32) ──Conv1(6,5×5)──→ (28×28) ──Pool1(2×2)──→ (14×14)
                              │
                              ↓
                    Conv2(16,5×5) → (10×10) ──Pool2(2×2)──→ (5×5)
                              │
                              ↓
                    Conv3(120,5×5) → (1×1)
                              │
                              ↓
                    FC(84) → FC(10) → 输出(10个数字)
```

### PyTorch 实现

```python
import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()

        # Conv1: 1通道输入，6个5×5卷积核
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv2: 6通道输入，16个5×5卷积核
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv3: 16通道输入，120个5×5卷积核
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)

        # 全连接层
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # 展平
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

---

## 🏗️ AlexNet：深度学习复兴

### ImageNet 比赛 2012

```
AlexNet 赢了 ImageNet 比赛！
错误率从 26% 降到 15%，震惊了整个计算机视觉界！
```

### AlexNet 的关键创新

```
1. 深度：8层（比 LeNet 深很多）
2. ReLU：比 Sigmoid 快 6 倍
3. Dropout：防止过拟合
4. GPU 训练：用两块 GTX 580 训练
5. 数据增强：翻转、裁剪
```

---

## 🏗️ ResNet：残差网络

### 问题：越深效果越好吗？

```
实验发现：56层网络比20层网络的训练误差更高！

原因：深层网络难以训练（梯度消失）

解决方案：引入"捷径"（Shortcut Connection）！
```

### 残差块

```
普通网络：
    x → [Conv] → [Conv] → ReLU → y

残差网络：
    x → [Conv] → [Conv] → ReLU → y
      ↳_______Shortcuts_________↑

输出 = F(x) + x

如果 F(x) = 0，那么输出 = x（恒等映射）
这让网络更容易学习！
```

### PyTorch 实现

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut：如果通道数不同，需要调整
        self.shortcut = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += residual  # 关键：加回原始输入
        out = torch.relu(out)

        return out
```

---

## 📊 CNN 学到了什么？

### 第一层学到的特征

```
第一层卷积核学到的特征：

    边缘检测核          纹理检测核
┌───────────┐        ┌───────────┐
│-1 0 1│    │        │ 0 1 0│    │
│-1 0 1│    │        │ 1 1 1│    │
│-1 0 1│    │        │ 0 1 0│    │
└───────────┘        └───────────┘

这些核能检测：边缘、纹理、颜色等基本特征
```

### 深层学到的特征

```
层次化学习：

第1层：边缘、纹理
第2层：纹理组合 → 形状
第3层：形状组合 → 部件（眼睛、耳朵）
第4层：部件组合 → 物体（猫、狗）
第5层：物体组合 → 场景
```

---

## ✅ 本章小结

| 概念 | 解释 |
|------|------|
| 卷积 | 用卷积核在图像上滑动，提取特征 |
| 卷积核 | 小的权重矩阵，学习特征 |
| 局部连接 | 每个神经元只连接一小块区域 |
| 权重共享 | 同一个卷积核在整张图上共享参数 |
| 池化 | 降维，提取主要特征 |
| Padding | 边缘填充，保持尺寸 |
| Stride | 步长，控制输出尺寸 |
| ResNet | 残差连接，解决深层网络训练问题 |

---

## 🔗 继续学习

👉 [RNN与LSTM](./RNN与LSTM.md)
