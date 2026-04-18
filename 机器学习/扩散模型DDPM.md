# 扩散模型DDPM

_通过逐步去噪生成图像_

---

## 第一部分：扩散模型概述

### 核心思想

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

## 第二部分：前向过程（已知）

```python
def forward_diffusion(x_0, t, betas):
    """
    前向过程：逐步添加噪声
    q(x_t | x_{t-1}) = N(x_t; sqrt(1-β_t) * x_{t-1}, β_t * I)
    """
    alphas = 1 - betas
    alpha_bar = np.cumprod(alphas)
    noise = torch.randn_like(x_0)
    sqrt_alpha_bar = torch.sqrt(alpha_bar[t])
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar[t])
    x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
    return x_t, noise
```

---

## 第三部分：反向过程（学习）

```python
class DDPM(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = UNet()  # U-Net 用于去噪

    def forward(self, x_t, t):
        """预测噪声"""
        return self.backbone(x_t, t)

    @torch.no_grad()
    def generate(self, shape, T=1000):
        """从纯噪声生成图像"""
        x_t = torch.randn(shape)
        for t in reversed(range(T)):
            # 预测噪声
            eps_theta = self.forward(x_t, t)
            # 去噪步骤...
        return x_0
```

---

## 小结

1. **DDPM**通过两个过程学习：前向扩散（加噪）和反向去噪
2. 前向过程有闭式解
3. 反向过程通过 U-Net 学习去噪
4. 相比 GAN：更稳定、生成多样性更好

---

_继续学习：下一章「Stable Diffusion」_
