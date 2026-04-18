# 扩散模型DDPM

_通过逐步去噪生成图像_

---

## 📖 学习目标

1. 理解扩散模型的两个过程
2. 掌握前向扩散和反向去噪
3. 理解决DDPM的训练目标

---

## 第一部分：两个过程

```
前向扩散（Forward）：
x₀ → x₁ → ... → x_T（逐渐加噪）

反向去噪（Reverse）：
x_T → x_{T-1} → ... → x₀（学习去噪）
```

---

## 第二部分：前向过程（已知）

### 闭式解

```python
def forward_diffusion(x_0, t, betas):
    alphas = 1 - betas
    alpha_bar = np.cumprod(alphas)
    noise = np.random.randn_like(x_0)
    sqrt_alpha_bar = np.sqrt(alpha_bar[t])
    sqrt_one_minus = np.sqrt(1 - alpha_bar[t])
    return sqrt_alpha_bar * x_0 + sqrt_one_minus * noise, noise
```

---

## 第三部分：反向过程（学习）

```python
class DDPM(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone  # 通常是U-Net

    def forward(self, x_t, t):
        return self.backbone(x_t, t)

    @torch.no_grad()
    def generate(self, shape, T=1000):
        x_t = torch.randn(shape)
        for t in reversed(range(T)):
            noise_pred = self.forward(x_t, t)
            # 简化的去噪步骤
            alpha = 1 - betas[t]
            alpha_bar = np.prod(1 - betas[:t+1])
            x_t = (x_t - np.sqrt(1-alpha_bar) * noise_pred) / np.sqrt(alpha_bar)
        return x_t
```

---

## ✅ 小结

1. **DDPM**通过前向扩散（加噪）和反向去噪学习
2. 前向过程有闭式解
3. 反向过程通过U-Net学习去噪
4. 训练稳定，但采样慢

---

_继续学习：下一章「Stable Diffusion」_
