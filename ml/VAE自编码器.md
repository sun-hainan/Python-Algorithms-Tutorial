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

训练目标：最小化重建误差
```

### VAE的核心改进

```
AE的问题：z是一个确定的向量，没有随机性

VAE的改进：
- z不再是一个确定向量，而是一个概率分布
- Encoder输出均值μ和方差σ
- 从N(μ, σ²)采样得到z

这样就可以从N(0,1)随机采样z，生成新样本！
```

---

## 第二部分：VAE的数学推导

### 目标：最大化对数似然

```
我们希望模型生成的x'尽可能接近真实x

等价于：最大化 log P(x)

但直接计算 P(x) 很难（需要对整个潜在空间积分）

引入Encoder q(z|x)来近似真实的p(z|x)
```

### ELBO（Evidence Lower Bound）

```
log P(x) = L_ELBO + KL(q(z|x) || P(z|x))

优化下界 L_ELBO = E[log P(x|z)] - KL(q(z|x) || P(z))

其中：
- E[log P(x|z)]：重建损失（让解码器正确重建）
- KL(q(z|x) || P(z))：让后验分布接近先验（标准正态分布）
```

---

## 第三部分：重参数化技巧

### 问题：采样操作不可导

```
z ~ N(μ, σ²)

梯度不能流过随机采样操作！
```

### 解决方案

```
z = μ + σ × ε, 其中 ε ~ N(0, 1)

这样：
- μ和σ是可学习的参数
- ε是随机噪声
- z是确定性函数，可导！
```

```python
def reparameterize(mu, logvar):
    """
    重参数化

    mu: 均值
    logvar: log(方差)，用log是为了保证σ>0
    """
    std = torch.exp(0.5 * logvar)  # σ = exp(log(σ)) = exp(logvar/2)
    eps = torch.randn_like(std)     # ε ~ N(0, 1)
    return mu + eps * std            # z = μ + σ × ε
```

---

## 第四部分：VAE代码实现

```python
import torch
import torch.nn as nn

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

        # 输出均值和方差
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # 输出在 [0, 1]
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    """
    VAE损失 = 重建损失 + β × KL散度

    beta: KL散度的权重，beta>1会让潜在空间更正则化
    """
    # 重建损失（Binary Cross Entropy）
    recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')

    # KL散度：KL(N(μ,σ) || N(0,1))
    # = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_loss, recon_loss, kl_loss
```

---

## 第五部分：生成新样本

```python
# 训练后，从潜在空间采样生成新样本
model.eval()
with torch.no_grad():
    # 方式1：从标准正态分布采样
    z = torch.randn(16, latent_dim)
    generated = model.decode(z)

    # 方式2：在两个样本之间插值
    x1, x2 = data[0], data[1]
    mu1, _ = model.encode(x1)
    mu2, _ = model.encode(x2)

    # 插值
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        z_interp = alpha * mu1 + (1 - alpha) * mu2
        x_interp = model.decode(z_interp)
        # 可视化插值过程
```

---

## ✅ 小结

1. **VAE**用概率分布代替确定向量作为潜在表示
2. 核心：**重参数化技巧**让采样可导
3. 损失 = 重建损失 + KL散度
4. 可以从 N(0,1) 采样生成新样本
5. 生成图像较GAN模糊，但训练更稳定

---

_继续学习：下一章「GAN生成对抗网络」_
