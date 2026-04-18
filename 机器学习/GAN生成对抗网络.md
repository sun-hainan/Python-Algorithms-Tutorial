# GAN生成对抗网络

_让两个网络互相"对抗"学习_

---

## 第一部分：GAN 的核心思想

### 核心思想

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
```

---

## 第二部分：损失函数

```
min_G max_D V(D, G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]

- D(x): 真实图像的判断概率
- D(G(z)): 生成图像的判断概率
- G 最小化这个值（让 D 认为是真的）
- D 最大化这个值（正确分类真假）
```

---

## 第三部分：代码实现

```python
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
            nn.Linear(512, int(np.prod(img_shape))), nn.Tanh())

    def forward(self, z):
        img = self.net(z)
        return img.view(img.size(0), *self.img_shape)

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

## 小结

1. **GAN**通过对抗训练学习数据分布
2. Generator 生成假图像，Discriminator 判别真假
3. 常见问题：训练不稳定、Mode Collapse

---

_继续学习：下一章「扩散模型DDPM」_
