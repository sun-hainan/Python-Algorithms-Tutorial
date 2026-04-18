# GAN生成对抗网络

_让两个网络互相博弈_

---

## 📖 学习目标

1. 理解决GAN的核心思想
2. 掌握对抗训练的原理
3. 理解GAN的损失函数

---

## 第一部分：对抗的思想

### 伪造者与鉴定师

```
伪造者（Generator）：生成假币，试图骗过鉴定师
鉴定师（Discriminator）：判断是真币还是假币

对抗过程：
1. 鉴定师看真币和假币，学习区分
2. 伪造者听到鉴定师的反馈，提高造假技术
3. 循环往复，直到伪造者能骗过鉴定师
```

### GAN结构

```
随机噪声 z ──→ Generator ──→ 生成图像 G(z)
                              │
                              ↓
真实图像 x ──→ Discriminator ──→ 判断真/假 D(x), D(G(z))
                              ↑
真实图像 x ──↗
```

---

## 第二部分：GAN的损失函数

### 极小极大博弈

```
min_G max_D V(D, G) = E[log D(x)] + E[log(1 - D(G(z)))]

D的目标：让V最大（正确分类真假）
G的目标：让V最小（让D以为G(z)是真）

当双方达到平衡时：
- D(x) = 0.5（对所有输入都判断为真概率50%）
- G生成了真实的数据分布
```

### 代码实现

```python
class Generator(nn.Module):
    """生成器"""
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
    """判别器"""
    def __init__(self, img_shape):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 输出 [0, 1]，真/假概率
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.net(img_flat)
        return validity
```

---

## 第三部分：训练过程

```python
def train_gan(generator, discriminator, dataloader, latent_dim, epochs, lr=0.0002):
    """GAN训练"""
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    for epoch in range(epochs):
        for batch_idx, real_images in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # ===== 训练判别器 =====
            optimizer_D.zero_grad()

            # 真图像损失
            real_output = discriminator(real_images)
            d_loss_real = criterion(real_output, real_labels)

            # 假图像损失
            z = torch.randn(batch_size, latent_dim)
            fake_images = generator(z).detach()  # detach避免更新G
            fake_output = discriminator(fake_images)
            d_loss_fake = criterion(fake_output, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # ===== 训练生成器 =====
            optimizer_G.zero_grad()

            z = torch.randn(batch_size, latent_dim)
            fake_images = generator(z)
            fake_output = discriminator(fake_images)

            # G想让D认为这是真图像
            g_loss = criterion(fake_output, real_labels)
            g_loss.backward()
            optimizer_G.step()

        print(f"Epoch {epoch}: D_loss={d_loss.item():.4f}, G_loss={g_loss.item():.4f}")
```

---

## 第四部分：常见问题

### Mode Collapse

```
问题：G只生成几种固定的样本，失去多样性

原因：G发现能骗过D的少数样本就不再探索

解决：WGAN、小批量判别、Unrolled GAN
```

### 训练不稳定

```
解决：
- 使用标签平滑（Label Smoothing）
- 使用TTUR（不同学习率）
- 梯度惩罚（WGAN-GP）
```

---

## ✅ 小结

1. **GAN**通过对抗训练学习数据分布
2. Generator生成假图像，Discriminator判断真假
3. 损失函数是极小极大博弈
4. 常见问题：Mode Collapse、训练不稳定
5. 变体：DCGAN、WGAN、CGAN、CycleGAN

---

_继续学习：下一章「扩散模型DDPM」_
