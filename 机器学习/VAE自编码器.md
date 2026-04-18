# VAE自编码器

_变分自编码器：学习数据分布的生成模型_

---

## 第一部分：从 AE 到 VAE

### 自编码器 (AE)

```
Encoder: x → z（压缩）
Decoder: z → x（重建）
```

### 变分自编码器 (VAE)

```
Encoder: x → μ, σ（均值、方差）
       z ~ N(μ, σ)（采样）
Decoder: z → x（从分布采样重建）
```

---

## 第二部分：核心技巧 - 重参数化

```python
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
```

---

## 第三部分：代码实现

```python
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128))
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, input_dim), nn.Sigmoid())

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def vae_loss(x_recon, x, mu, logvar):
    recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss
```

---

## 小结

1. **VAE**用概率分布代替确定向量
2. 核心：**重参数化技巧**解决采样不可导问题
3. 损失 = 重建损失 + KL 散度
4. 潜在空间连续，可生成新样本

---

_继续学习：下一章「GAN生成对抗网络」_
