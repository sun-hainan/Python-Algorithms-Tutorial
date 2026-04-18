# GAN生成对抗网络

_让两个网络互相博弈_

---

## 📖 学习目标

1. 理解对抗的思想
2. 掌握GAN的损失函数
3. 理解对抗训练的原理

---

## 第一部分：对抗的思想

### 伪造者与鉴定师

```
伪造者（Generator）：生成假币，试图骗过鉴定师
鉴定师（Discriminator）：判断是真币还是假币
```

### GAN结构

```
随机噪声 z ──→ Generator ──→ 生成图像 G(z)
                              │
                              ↓
真实图像 x ──→ Discriminator ──→ 判断真/假
```

---

## 第二部分：GAN的损失函数

```
min_G max_D V(D, G) = E[log D(x)] + E[log(1 - D(G(z)))]
```

---

## 第三部分：代码实现

```python
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
            nn.Linear(512, int(np.prod(img_shape)), nn.Tanh())

    def forward(self, z):
        return self.net(z).view(z.size(0), *self.img_shape)


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, img):
        return self.net(img.view(img.size(0), -1))
```

---

## 第四部分：训练过程

```python
def train_gan(generator, discriminator, dataloader, latent_dim, epochs, lr=0.0002):
    g_opt = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for real_images in dataloader:
            # 训练判别器
            d_opt.zero_grad()
            z = torch.randn(len(real_images), latent_dim)
            fake = generator(z).detach()
            d_loss = -torch.mean(torch.log(discriminator(real_images)) + torch.log(1 - discriminator(fake)))
            d_loss.backward()
            d_opt.step()

            # 训练生成器
            g_opt.zero_grad()
            z = torch.randn(len(real_images), latent_dim)
            fake = generator(z)
            g_loss = -torch.mean(torch.log(discriminator(fake)))
            g_loss.backward()
            g_opt.step()
```

---

## ✅ 小结

1. **GAN**通过对抗训练学习数据分布
2. Generator生成假图像，Discriminator判断真假
3. 损失函数是极小极大博弈

---

_继续学习：下一章「扩散模型DDPM」_
