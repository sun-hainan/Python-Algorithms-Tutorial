# VAE自编码器

_变分自编码器：学习数据分布的生成模型_

---

## 📖 学习目标

1. 理解决AE和VAE的区别
2. 掌握重参数化技巧
3. 理解 VAE 的损失函数

---

## 第一部分：从 AE 到 VAE

### 自编码器 (AE)

```
Encoder: x → z（压缩）
Decoder: z → x（重建）

训练目标：重建 x ≈ Decoder(Encoder(x))
```

### 变分自编码器 (VAE)

```
区别：z 不是确定向量，而是概率分布

Encoder: x → μ, σ（均值、方差）
       z ~ N(μ, σ)（采样）

Decoder: z → x（从分布采样重建）
```

---

## 第二部分：VAE 的核心思想

### 🎯 为什么需要 VAE？

```
普通 AE 的问题：
- 潜在空间不连续，无法随机生成新样本
- encoder(x1) 和 encoder(x2) 之间的 z 可能没有对应的好图像

VAE 的解决方案：
- 每个样本变成一个分布
- 潜在空间是连续的
- 可以从 N(0,1) 随机采样生成新样本
```

---

## 第三部分：代码实现

```python
import numpy as np
import torch
import torch.nn as nn

class VAE(nn.Module):
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
        self.fc_mu = nn.Linear(128, latent_dim)      # 均值
        self.fc_logvar = nn.Linear(128, latent_dim)  # 对数方差

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # 输出 [0, 1] 范围
        )

    def reparameterize(self, mu, logvar):
        """重参数化技巧：让采样可导"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 编码
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # 重参数化采样
        z = self.reparameterize(mu, logvar)

        # 解码
        x_recon = self.decoder(z)

        return x_recon, mu, logvar

    def generate(self, z):
        """从潜在向量生成"""
        return self.decoder(z)
```

---

## 第四部分：损失函数

```python
def vae_loss(x_recon, x, mu, logvar):
    """
    VAE 损失 = 重建损失 + KL 散度

    1. 重建损失：让解码器能重建输入
    2. KL 散度：让潜在分布接近标准正态分布
    """
    # 1. 重建损失（BCELoss）
    recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')

    # 2. KL 散度
    # KL(N(μ,σ) || N(0,1)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss


# 训练循环
def train_vae(model, data_loader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            x = batch.view(-1, 784)  # 假设是 28x28 图像

            optimizer.zero_grad()
            x_recon, mu, logvar = model(x)

            loss = vae_loss(x_recon, x, mu, logvar)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: Loss = {total_loss / len(data_loader.dataset):.4f}")
```

---

## 第五部分：生成新样本

```python
# 训练后，从 N(0,1) 采样生成新图像
model.eval()
with torch.no_grad():
    # 随机采样
    z = torch.randn(16, latent_dim)
    generated = model.generate(z)

    # 或者在两个潜在向量之间插值
    z1 = model.fc_mu(model.encoder(x1))
    z2 = model.fc_mu(model.encoder(x2))
    z_interpolate = 0.5 * z1 + 0.5 * z2
    interpolated = model.generate(z_interpolate)
```

---

## 第六部分：名词解释

### 重参数化技巧

```
问题：z ~ N(μ, σ) 中的采样操作不可导

解决：z = μ + σ * ε, 其中 ε ~ N(0, 1)

优点：μ and σ 可导，z 也可导
```

### KL 散度

```
定义：两个概率分布的差异

KL(P||Q) = Σ P(x) * log(P(x)/Q(x))

VAE 中：让学到的分布接近标准正态分布
```

---

## ✅ 小结

1. **VAE**用概率分布代替确定向量
2. 核心：**重参数化技巧**解决采样不可导问题
3. 损失 = 重建损失 + KL 散度
4. 优点：潜在空间连续，可生成新样本
5. 缺点：生成图像较模糊

---

_继续学习：下一章「GAN生成对抗网络」_
