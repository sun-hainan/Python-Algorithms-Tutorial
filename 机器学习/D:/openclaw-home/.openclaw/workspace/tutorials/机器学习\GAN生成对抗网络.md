# GAN生成对抗网络

_让两个网络互相"对抗"学习_

---

## 📖 学习目标

1. 理解决GAN的核心思想
2. 掌握 Generator 和 Discriminator 的对抗训练
3. 理解 GAN 的训练技巧

---

## 第一部分：GAN 的核心思想

### 🎯 核心思想

**让 Generator 和 Discriminator 互相"对抗"学习。**

```
GAN 结构：

                    真实图像
                       ↓
                   ┌───────┐
                   │   D   │ ← 判断真假
                   └───┬───┘
                       │
          生成图像 ←───┘

Generator (G): 生成假图像，试图骗过 D
Discriminator (D): 判断图像是真是假

对抗训练：G 试图生成越来越真的图像
         D 试图越来越准确地分辨真假
```

---

## 第二部分：对抗训练的数学

### 损失函数

```python
"""
GAN 的 min-max 游戏：

min_G max_D V(D, G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]

- D(x): 真实图像的判断概率
- D(G(z)): 生成图像的判断概率
- G 最小化这个值（让 D 认为是真的）
- D 最大化这个值（正确分类真假）
"""
```

---

## 第三部分：代码实现

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    """生成器：随机向量 → 图像"""
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh()  # 输出 [-1, 1]
        )

    def forward(self, z):
        img = self.net(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    """判别器：图像 → 真/假概率"""
    def __init__(self, img_shape):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 输出 [0, 1]
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.net(img_flat)
        return validity
```

---

## 第四部分：训练循环

```python
def train_gan(generator, discriminator, dataloader, latent_dim, epochs, lr=0.0002):
    """GAN 训练"""
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    for epoch in range(epochs):
        for batch_idx, real_images in enumerate(dataloader):
            batch_size = real_images.size(0)

            # 真实标签 = 1，假标签 = 0
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # ========== 训练 Discriminator ==========
            optimizer_D.zero_grad()

            # 真实图像损失
            real_output = discriminator(real_images)
            d_loss_real = criterion(real_output, real_labels)

            # 生成图像损失
            z = torch.randn(batch_size, latent_dim)
            fake_images = generator(z)
            fake_output = discriminator(fake_images.detach())  # detach 避免更新 G
            d_loss_fake = criterion(fake_output, fake_labels)

            # 总判别器损失
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # ========== 训练 Generator ==========
            optimizer_G.zero_grad()

            # G 试图让 D 认为是真实的
            fake_output = discriminator(fake_images)
            g_loss = criterion(fake_output, real_labels)

            g_loss.backward()
            optimizer_G.step()

        print(f"Epoch {epoch}: D_loss={d_loss.item():.4f}, G_loss={g_loss.item():.4f}")
```

---

## 第五部分：常见 GAN 变体

### DCGAN (Deep Convolutional GAN)

```python
class DCGenerator(nn.Module):
    """使用转置卷积的生成器"""
    def __init__(self, latent_dim, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0),  # 转置卷积
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, channels, 4, 2, 1),
            nn.Tanh()
        )
```

### WGAN-GP (Wasserstein GAN with Gradient Penalty)

```python
def wgan_gp_loss(D, real_images, fake_images, lambda_gp=10):
    """WGAN-GP 损失"""
    # 原始 WGAN 损失
    d_loss = -torch.mean(D(real_images)) + torch.mean(D(fake_images))

    # 梯度惩罚
    alpha = torch.rand(real_images.size(0), 1, 1, 1)
    interpolated = alpha * real_images + (1 - alpha) * fake_images
    gradient = torch.autograd.grad(D(interpolated), interpolated, torch.ones_like(D(interpolated)))[0]
    gradient_penalty = lambda_gp * ((gradient.norm(2, dim=1) - 1) ** 2).mean()

    return d_loss + gradient_penalty
```

---

## 第六部分：训练技巧

| 技巧 | 说明 |
|------|------|
| Label Smoothing | 真实标签用 0.9 代替 1 |
| Spectral Normalization | 限制判别器梯度 |
| TTUR | Generator 和 Discriminator 用不同学习率 |
| 逐步增长 | 从低分辨率开始，逐步增大 |

---

## 第七部分：名词解释

### Mode Collapse

```
定义：Generator 学会生成几种固定的样本，失去了多样性

原因：G 发现一种能骗过 D 的样本就不再探索

解决：WGAN、小批量判别、Unrolled GAN
```

### Wasserstein 距离

```
定义：把一个分布"搬"到另一个分布的最小代价

优点：比 JS 散度更平滑，解决训练不稳定问题
```

---

## ✅ 小结

1. **GAN**通过对抗训练学习数据分布
2. Generator 生成假图像，Discriminator 判别真假
3. 损失函数：min_G max_D V(D, G)
4. 常见问题：训练不稳定、Mode Collapse
5. 变体：DCGAN、WGAN、CGAN、CycleGAN 等

---

_继续学习：下一章「扩散模型DDPM」_
