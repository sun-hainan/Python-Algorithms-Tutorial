# 扩散模型DDPM

_通过逐步去噪生成图像_

---

## 📖 学习目标

1. 理解决扩散模型的核心思想
2. 掌握前向扩散和反向去噪过程
3. 理解 DDPM 的训练目标

---

## 第一部分：扩散模型概述

### 🎯 核心思想

**扩散 = 两个过程：加噪（破坏）→ 去噪（生成）**

```
前向过程（Forward/Diffusion）：
x_0 → x_1 → x_2 → ... → x_T

每一步添加少量高斯噪声
直到 x_T ≈ 标准正态分布

反向过程（Reverse/Denoising）：
x_T → x_{T-1} → ... → x_1 → x_0

学习一个网络预测噪声，逐步去噪
```

---

## 第二部分：图解过程

```
前向扩散（训练阶段）：
x_0 (真实图像) → x_1 (加噪) → x_2 → ... → x_T (纯噪声)

反向去噪（生成阶段）：
x_T (纯噪声) → x_{T-1} (去噪一点) → ... → x_0 (生成图像)
```

---

## 第三部分：前向过程（已知）

```python
import torch
import numpy as np

def forward_diffusion(x_0, t, betas):
    """
    前向过程：逐步添加噪声

    x_0: 原始图像
    t: 时间步
    betas: 噪声调度（每步的 β）

    q(x_t | x_{t-1}) = N(x_t; sqrt(1-β_t) * x_{t-1}, β_t * I)
    """
    T = len(betas)

    # 计算累积的均值和方差
    alphas = 1 - betas
    alpha_bar = np.cumprod(alphas)

    # 采样噪声
    noise = torch.randn_like(x_0)

    # 闭式解：直接计算 x_t
    sqrt_alpha_bar = torch.sqrt(alpha_bar[t])
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar[t])

    x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
    return x_t, noise


# 测试
betas = np.linspace(1e-4, 0.02, 1000)  # T=1000
x_0 = torch.randn(1, 3, 64, 64)  # 原始图像
t = 500  # 中间时刻
x_t, noise = forward_diffusion(x_0, t, betas)
print(f"x_t 形状: {x_t.shape}")
```

---

## 第四部分：反向过程（学习）

```python
class DDPM(nn.Module):
    """DDPM 模型"""
    def __init__(self, latent_dim=256):
        super().__init__()
        # U-Net backbone（类似 ResNet，但有跳跃连接）
        self.backbone = UNet()

    def forward(self, x_t, t):
        """
        预测噪声

        输入：噪声图像 x_t 和时间步 t
        输出：预测的噪声 ε_θ(x_t, t)
        """
        return self.backbone(x_t, t)

    def p_sample(self, x_t, t, model):
        """
        从 x_t 采样 x_{t-1}

        使用预测的噪声重建
        """
        betas = np.linspace(1e-4, 0.02, 1000)
        alpha = 1 - betas[t]
        alpha_bar = np.cumprod(alpha)

        # 预测噪声
        eps_theta = model(x_t, t)

        # 计算均值和方差
        mean = (x_t - betas[t] / torch.sqrt(1 - alpha_bar[t]) * eps_theta) / torch.sqrt(alpha[t])
        variance = betas[t]

        # 采样
        if t > 0:
            noise = torch.randn_like(x_t)
            x_{t-1} = mean + torch.sqrt(variance) * noise
        else:
            x_{t-1} = mean

        return x_{t-1}

    @torch.no_grad()
    def generate(self, shape, T=1000):
        """从纯噪声生成图像"""
        x_t = torch.randn(shape)

        for t in reversed(range(T)):
            x_t = self.p_sample(x_t, t, self)
            if t % 100 == 0:
                print(f"Step {t}/{T}")

        return x_t
```

---

## 第五部分：U-Net backbone

```python
class UNet(nn.Module):
    """U-Net 结构用于去噪"""
    def __init__(self, in_channels=3, base_channels=128):
        super().__init__()

        # 编码器（下采样）
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1)
        )
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)

        # 时间嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(1, base_channels),
            nn.SiLU(),
            nn.Linear(base_channels, base_channels * 2)
        )

        # ... 更多层

    def forward(self, x, t):
        # 编码
        h1 = self.enc1(x)
        h2 = self.down1(h1)
        # ...

        # 时间嵌入
        t_emb = self.time_mlp(t)

        # 解码 + 跳跃连接
        # ...

        return noise_pred
```

---

## 第六部分：训练过程

```python
def train_ddpm(model, dataloader, optimizer, epochs=100):
    """DDPM 训练"""
    for epoch in range(epochs):
        for batch in dataloader:
            x_0 = batch  # 真实图像

            # 随机采样时间步
            t = torch.randint(0, 1000, (x_0.size(0),))

            # 前向加噪
            x_t, noise = forward_diffusion(x_0, t, betas)

            # 预测噪声
            noise_pred = model(x_t, t)

            # MSE 损失
            loss = nn.functional.mse_loss(noise_pred, noise)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

---

## 第七部分：名词解释

### 噪声调度 (Noise Schedule)

```
定义：每一步添加多少噪声

常见策略：
- 线性：β_t 线性增加
- 余弦：β_t = cos²(πt/2T)
- SDE：随机微分方程视角
```

### 闭式解

```
DDPM 的优点：前向过程有闭式解，可以直接计算任意时刻的噪声图像

x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε

不需要一步步模拟扩散过程
```

---

## ✅ 小结

1. **DDPM**通过两个过程学习：前向扩散（加噪）和反向去噪
2. 前向过程有闭式解，可以直接计算任意时刻
3. 反向过程通过 U-Net 学习去噪
4. 生成过程：纯噪声 → 逐步去噪 → 图像
5. 相比 GAN：更稳定、生成多样性更好

---

_继续学习：下一章「Stable Diffusion」_
