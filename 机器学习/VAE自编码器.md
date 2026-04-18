# VAE自编码器

_学习数据分布的生成模型_

---

## 📖 学习目标

1. 理解决AE和VAE的区别
2. 掌握重参数化技巧
3. 理解KL散度在VAE中的作用

---

## 第一部分：从AE到VAE

### 自编码器（AE）

```
Encoder: x → z（压缩到潜在空间）
Decoder: z → x（从潜在空间重建）
```

### VAE的核心改进

```
z不再是一个确定向量，而是一个概率分布
Encoder输出均值μ和方差σ
从N(μ, σ²)采样得到z
```

---

## 第二部分：重参数化技巧

### 问题：采样操作不可导

```
z ~ N(μ, σ²)
梯度不能流过随机采样！
```

### 解决方案

```
z = μ + σ × ε, 其中 ε ~ N(0, 1)
```

```python
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
```

---

## 第三部分：VAE代码

```python
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU())
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, input_dim), nn.Sigmoid())

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(x_recon, x, mu, logvar):
    recon = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl
```

---

## ✅ 小结

1. **VAE**用概率分布代替确定向量
2. 核心：**重参数化技巧**让采样可导
3. 损失 = 重建损失 + KL散度

---

_继续学习：下一章「GAN生成对抗网络」_
