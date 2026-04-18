# VAE自编码器

> _学习数据分布的生成模型，能生成新的样本_

---

## 🎯 先看一个生活中的例子

### 例子：画人脸

假设你是一个画家，你见过很多张人脸：
- 高鼻子的人
- 圆脸的人
- 大眼睛的人
- 等等

现在让你画一张"新"的人脸（不是见过的任何一张，但看起来像真人）。

你会怎么做？

```
1. 你会提取"人脸"的共同特征：
   - 两只眼睛
   - 一个鼻子
   - 一张嘴
   - 位置大概对称

2. 你会理解这些特征的"分布"：
   - 眼睛一般在上半部分
   - 鼻子在中间
   - 嘴巴在下半部分
   - 等等

3. 然后你从这个分布中采样，生成新的人脸！
```

**VAE 就是让机器学习这种能力！**

---

## 🤔 AE（自编码器）vs VAE

### 自编码器（AE）

```
AE 结构：
- Encoder：x → z（压缩到潜在空间）
- Decoder：z → x（从潜在空间重建）

训练目标：让重建的 x̂ 尽可能接近 x
```

### AE 的问题

```
问题：潜在空间可能不连续！

训练后：
- 输入猫的图片 → 潜在空间中某个点 A
- 输入狗的图片 → 潜在空间中另一个点 B

但 A 和 B 之间的区域可能什么都没有！
如果从 A 和 B 之间随便选一个点，解码出来可能是噪声！
```

### VAE 的解决方案

```
VAE 的核心思想：
z 不再是一个确定向量，而是一个概率分布！

Encoder 输出：均值 μ 和 方差 σ（或 logσ²）
z 从 N(μ, σ²) 中采样

这样潜在空间是连续的！
```

---

## 📐 VAE的数学原理

### 目标：学习数据分布

```
我们有一堆数据 {x⁽¹⁾, x⁽²⁾, ..., x⁽ⁿ⁾}
假设这些数据是从某个分布 P(x) 中生成的

VAE 的目标：学习这个分布 P(x)

如果能学到 P(x)，就可以从 P(x) 中采样，生成新的数据！
```

### 隐变量模型

```
隐变量 z：潜在的、不可见的变量

例子：生成人脸时
z = [眼睛大小, 鼻子高度, 脸型, ...]  ← 这些是隐变量

我们假设：
- z 服从标准正态分布 N(0, I)
- 给定 z，x 服从某个分布 P(x|z)

所以 P(x) = ∫ P(x|z) P(z) dz
```

### KL散度：衡量两个分布的差异

```
KL散度：KL(P || Q) = Σ P(x) log(P(x)/Q(x))

VAE 中的 KL：
- P(z|x)：真实后验（给定 x，z 应该是什么）
- Q(z|x)：近似后验（Encoder 输出的分布）

目标：让 Q(z|x) 尽可能接近 P(z|x)
```

---

## 🔑 重参数化技巧（Reparameterization Trick）

### 问题

```
VAE 需要从 N(μ, σ²) 中采样 z

但采样操作是随机的，无法求导！

这意味着梯度无法流过采样过程！
```

### 解决方案

```
z = μ + σ × ε, 其中 ε ~ N(0, 1)

这样：
- 采样的是 ε（可以求导）
- μ 和 σ 是确定的值（可以求导）

梯度可以流过 μ 和 σ 了！
```

---

## 💻 代码实现

### PyTorch VAE

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """变分自编码器"""
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # 输出均值和方差的对数
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # 如果输入是 [0,1] 范围
        )

    def encode(self, x):
        """编码，返回均值和方差"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """解码"""
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def vae_loss(x_recon, x, mu, logvar):
    """
    VAE 损失函数 = 重建损失 + KL散度

    重建损失：让解码器输出的图像尽可能接近原始图像
    KL散度：让潜在空间接近标准正态分布
    """
    # 重建损失：二元交叉熵
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')

    # KL 散度：KL(N(μ,σ²) || N(0,I))
    # = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss
```

---

## 🧪 完整例子：MNIST手写数字生成

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 超参数
INPUT_DIM = 784  # 28x28 展平
LATENT_DIM = 2  # 潜在空间维度（可以画出来）
BATCH_SIZE = 128
EPOCHS = 50
LR = 1e-3

# 数据
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# 模型
model = VAE(INPUT_DIM, LATENT_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 训练
for epoch in range(EPOCHS):
    total_loss = 0
    for batch, (images, _) in enumerate(train_loader):
        images = images.view(-1, INPUT_DIM)

        # 前向传播
        recon_images, mu, logvar = model(images)

        # 计算损失
        loss = vae_loss(recon_images, images, mu, logvar)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_data):.4f}")

# 生成新样本
with torch.no_grad():
    # 从标准正态分布采样
    z = torch.randn(16, LATENT_DIM)
    generated = model.decode(z).view(-1, 1, 28, 28)

    # 画出来
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated[i].squeeze(), cmap='gray')
        ax.axis('off')
    plt.suptitle('VAE 生成的 MNIST 数字')
    plt.show()
```

---

## 📊 VAE的潜在空间可视化

### 在2D潜在空间中观察

```python
# 显示潜在空间的分布
with torch.no_grad():
    # 编码所有训练数据
    mu_all = []
    for images, _ in train_loader:
        images = images.view(-1, INPUT_DIM)
        mu, _ = model.encode(images)
        mu_all.append(mu)
    mu_all = torch.cat(mu_all, dim=0).numpy()

# 画潜在空间
plt.figure(figsize=(10, 8))
for digit in range(10):
    mask = train_data.targets.numpy() == digit
    plt.scatter(mu_all[mask, 0], mu_all[mask, 1], label=str(digit), alpha=0.5)
plt.legend()
plt.xlabel('z₁')
plt.ylabel('z₂')
plt.title('VAE 潜在空间分布')
plt.show()

# 插值实验：在两个数字之间插值
with torch.no_grad():
    z1 = mu_all[train_data.targets.numpy() == 1][0]  # 一个"1"
    z7 = mu_all[train_data.targets.numpy() == 7][0]  # 一个"7"

    # 在 1 和 7 之间插值
    fig, axes = plt.subplots(1, 9, figsize=(18, 2))
    for i, alpha in enumerate(np.linspace(0, 1, 9)):
        z = z1 * (1-alpha) + z7 * alpha
        z = torch.tensor(z, dtype=torch.float32)
        digit = model.decode(z).view(28, 28)
        axes[i].imshow(digit, cmap='gray')
        axes[i].axis('off')
    plt.suptitle('1 到 7 的连续插值（潜在空间是连续的！）')
    plt.show()
```

---

## 📊 VAE vs AE 的区别

| 特性 | AE（自编码器）| VAE（变分自编码器）|
|------|--------------|-------------------|
| 潜在空间 | 可能不连续 | 连续、可采样 |
| 输出 | 确定向量 | 概率分布 |
| 生成能力 | 差（随机采样可能无效）| 强（从连续空间采样）|
| 损失函数 | 重建损失 | 重建损失 + KL散度 |
| 应用 | 降维、特征提取 | 生成任务 |

---

## ✅ 本章小结

| 概念 | 解释 |
|------|------|
| 自编码器 | Encoder + Decoder，无监督降维 |
| VAE | 让 z 变成概率分布，实现可生成的模型 |
| 重参数化技巧 | z = μ + σ×ε，让采样可导 |
| KL散度 | 衡量两个分布的差异 |
| 重建损失 | 让解码器能重建输入 |
| 潜在空间 | 连续的、可采样的隐空间 |

---

## 🔗 继续学习

VAE 是生成模型的基础。接下来让我们学习另一种生成模型——GAN！

👉 [GAN生成对抗网络](./GAN生成对抗网络.md)
