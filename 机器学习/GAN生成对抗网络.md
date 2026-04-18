# GAN生成对抗网络

> _让两个网络互相博弈，生成逼真图像的革命性技术_

---

## 🎯 先看一个生活中的例子

### 例子：伪造名画的犯罪团伙

```
场景：有一个伪造名画的犯罪团伙

他们有一个团队：
1. 伪造者（Forger）：负责伪造名画，试图骗过鉴定师
2. 鉴定师（Authenticator）：负责鉴定名画是真迹还是赝品

他们的博弈过程：

第一轮：
- 伪造者：画了一幅很差的赝品
- 鉴定师：一打眼就看出来是假的
- 伪造者意识到：画得太差了，需要改进

第二轮：
- 伪造者：改进了技术，画得好一点了
- 鉴定师：还是能看出来有些不对劲
- 伪造者继续改进...

第三轮、第四轮、...
- 伪造者：技术越来越好，越来越逼真
- 鉴定师：越来越难分辨真假

最终：
- 伪造者的画和真迹几乎无法区分！
- 甚至鉴定师也经常被骗！
```

**这就是 GAN 的核心思想！**

---

## 🤔 GAN的基本结构

### GAN 的两个网络

```
GAN = Generative Adversarial Network（生成对抗网络）

两个网络：
1. Generator（生成器）：伪造数据，试图骗过判别器
2. Discriminator（判别器）：判断数据是真实的还是生成的
```

### 网络结构图

```
随机噪声 z ──→ Generator ──→ 生成图像 G(z)
                              │
                              ↓
真实图像 x ──→ Discriminator ──→ 判断：真/假
                              │
                         反馈给 Generator
```

### 对抗的含义

```
Generator 的目标：生成越来越逼真的图像，骗过 Discriminator
Discriminator 的目标：越来越准确地分辨真假

两人互相对抗，互相提升：

Generator 变强 → 需要更高级的骗术
Discriminator 变强 → 需要更精准的鉴别

最终：双方都变得非常强！
```

---

## 📐 GAN的数学原理

### 极小极大博弈

```
GAN 的目标函数是一个极小极大博弈：

min_G max_D V(D, G) = E[log D(x)] + E[log(1 - D(G(z)))]

其中：
- D(x)：判别器对真实数据的判断（越接近1越真）
- D(G(z))：判别器对生成数据的判断
- G(z)：生成器从噪声生成的图像
```

### 直观理解

```
对 D（判别器）：
- 当输入是真实数据 x 时，D(x) 应该接近 1
- 当输入是生成数据 G(z) 时，D(G(z)) 应该接近 0
- 所以 D 要 maximize log D(x) + log(1 - D(G(z)))

对 G（生成器）：
- G 要 minimize log(1 - D(G(z)))
- 也就是说，要让 D(G(z)) 尽可能接近 1（骗过判别器）
```

### 训练过程

```
第一步：训练判别器 D（固定 G）
- 输入真实数据，期望 D(x) = 1
- 输入生成数据，期望 D(G(z)) = 0
- 让 D 学会分辨真假

第二步：训练生成器 G（固定 D）
- 输入噪声，生成 G(z)
- 期望 D(G(z)) = 1（骗过 D）
- 让 G 学会更逼真的生成

重复以上步骤，直到平衡
```

---

## 💻 代码实现

### PyTorch GAN

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 超参数
LATENT_DIM = 100
IMG_SHAPE = (1, 28, 28)  # MNIST 图片形状
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.0002
BETA1 = 0.5

class Generator(nn.Module):
    """生成器：从噪声生成图像"""
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim

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
            nn.Tanh()  # 输出范围 [-1, 1]
        )

    def forward(self, z):
        img = self.net(z)
        return img.view(img.size(0), *self.img_shape)


class Discriminator(nn.Module):
    """判别器：判断图像真假"""
    def __init__(self, img_shape):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 输出 [0, 1]，表示真假的概率
        )

    def forward(self, img):
        return self.net(img.view(img.size(0), -1))


# 初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator(LATENT_DIM, IMG_SHAPE).to(device)
discriminator = Discriminator(IMG_SHAPE).to(device)

# 优化器
opt_G = torch.optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, 0.999))
opt_D = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, 0.999))

# 损失函数
criterion = nn.BCELoss()


def train_step(real_imgs, generator, discriminator):
    """一步训练"""
    batch_size = real_imgs.size(0)
    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)

    # ===== 训练判别器 =====
    discriminator.zero_grad()

    # 真实图片：期望输出 1
    real_output = discriminator(real_imgs)
    d_loss_real = criterion(real_output, real_labels)

    # 生成图片：期望输出 0
    z = torch.randn(batch_size, LATENT_DIM).to(device)
    fake_imgs = generator(z)
    fake_output = discriminator(fake_imgs.detach())  # detach 避免反向传到 G
    d_loss_fake = criterion(fake_output, fake_labels)

    # 总判别器损失
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    opt_D.step()

    # ===== 训练生成器 =====
    generator.zero_grad()

    # 生成图片：期望骗过判别器（输出 1）
    fake_output = discriminator(fake_imgs)
    g_loss = criterion(fake_output, real_labels)
    g_loss.backward()
    opt_G.step()

    return d_loss.item(), g_loss.item()
```

---

## ⚠️ GAN 的训练难题

### Mode Collapse（模式坍塌）

```
问题：生成器只学会生成几种固定的样本！

举例：生成 MNIST 数字
- 本应生成 0-9 十种数字
- 但生成器只学会生成"7"这个数字
- 因为"7"最容易骗过判别器

表现：生成的图片多样性很差
```

### 训练不稳定

```
问题：判别器和生成器需要保持平衡

如果 D 太强：G 的梯度消失，难以学习
如果 G 太强：D 被骗得太惨，也无法学习

理想情况：双方步调一致，一起变强
```

### 解决方案

```
1. 标签平滑（Label Smoothing）
   - 不让 D 期望 0 或 1，而是 0.1 或 0.9
   - 让训练更稳定

2. 谱归一化（Spectral Normalization）
   - 限制 D 的 Lipschitz 常数
   - 让 D 不会太强

3. WGAN-GP
   - 改用 Wasserstein 距离
   - 避免梯度消失
```

---

## 📊 DCGAN：深度卷积GAN

```python
class DCGenerator(nn.Module):
    """深度卷积生成器"""
    def __init__(self, latent_dim, channels=1, features_g=64):
        super().__init__()

        self.net = nn.Sequential(
            # 输入: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, features_g * 8, 4, 1, 0),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(),
            # 4x4
            nn.ConvTranspose2d(features_g * 8, features_g * 4, 4, 2, 1),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(),
            # 8x8
            nn.ConvTranspose2d(features_g * 4, features_g * 2, 4, 2, 1),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(),
            # 16x16
            nn.ConvTranspose2d(features_g * 2, features_g, 4, 2, 1),
            nn.BatchNorm2d(features_g),
            nn.ReLU(),
            # 32x32
            nn.ConvTranspose2d(features_g, channels, 4, 2, 1),
            nn.Tanh()
            # 输出: channels x 32 x 32
        )

    def forward(self, z):
        return self.net(z)
```

---

## 📊 GAN的应用

### 图像生成

```
- 人脸生成（StyleGAN）
- 艺术风格迁移
- 游戏场景生成
```

### 图像修复

```
- 填充缺失的图像部分
- 老照片修复
- 去马赛克
```

### 数据增强

```
- 生成更多训练数据
- 解决数据不平衡问题
```

---

## ✅ 本章小结

| 概念 | 解释 |
|------|------|
| GAN | 生成对抗网络，两个网络互相对抗 |
| Generator | 生成器，从噪声生成假数据 |
| Discriminator | 判别器，判断真假 |
| 对抗训练 | Generator 骗 D，D 识别 G，轮流提升 |
| Mode Collapse | 生成器只生成少数模式 |
| DCGAN | 深度卷积 GAN，用卷积层代替全连接层 |

---

## 🔗 继续学习

GAN 是生成模型的一种重要方法。接下来让我们学习另一种生成模型——扩散模型！

👉 [扩散模型DDPM](./扩散模型DDPM.md)
