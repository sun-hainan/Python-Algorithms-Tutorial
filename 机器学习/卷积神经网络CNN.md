# 卷积神经网络CNN

_图像处理的核心深度学习模型_

---

## 第一部分：CNN 的核心思想

### 为什么需要 CNN？

```
全连接网络处理图像的问题：
- 100x100 的图像 = 10000 个像素
- 如果第一层有 1000 个神经元，需要 1000万 个参数！

CNN 的解决方案：
- 局部连接：只看一小块区域
- 权重共享：同一个卷积核在整张图上共享
```

---

## 第二部分：核心组件

### 卷积层

```python
def conv2d(image, kernel, stride=1, padding=0):
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
    h, w = image.shape
    k = kernel.shape[0]
    out_h = (h - k) // stride + 1
    out_w = (w - k) // stride + 1
    output = np.zeros((out_h, out_w))
    for i in range(0, h - k + 1, stride):
        for j in range(0, w - k + 1, stride):
            output[i // stride, j // stride] = np.sum(image[i:i+k, j:j+k] * kernel)
    return output
```

### 池化层

```python
def max_pooling(image, pool_size=2, stride=2):
    h, w = image.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    output = np.zeros((out_h, out_w))
    for i in range(0, h - pool_size + 1, stride):
        for j in range(0, w - pool_size + 1, stride):
            output[i // stride, j // stride] = np.max(image[i:i+pool_size, j:j+pool_size])
    return output
```

---

## 第三部分：PyTorch 实现

```python
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

---

## 小结

1. **CNN**通过局部连接和权重共享大幅减少参数量
2. 核心组件：**卷积层**、**池化层**、**全连接层**
3. 经典架构：LeNet → AlexNet → VGG → ResNet

---

_继续学习：下一章「RNN与LSTM」_
