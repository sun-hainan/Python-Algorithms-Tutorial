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
前向扩散（Forward/Diffusion）：
x₀(真实图像) → x₁(加噪) → x₂ → ... → xₜ → ... → x_T(纯噪声)

反向去噪（Reverse/Denoising）：
x_T(纯噪声) → x_{T-1}(去噪) → ... → x₁ → x₀(生成图像)

核心：学习反向过程，从噪声恢复图像
```

---

## 第二部分：前向过程（已知）

### 逐步加噪

```python
def forward_diffusion(x_0, t, betas):
    """
    前向过程：直接计算任意时刻t的噪声图像

    利用闭式解，不需要一步步模拟
    q(x_t | x_0) = N(x_t; √ᾱ_t * x_0, (1-ᾱ_t) * I)

    其中 ᾱ_t = Πs=1^t α_s, α_s = 1 - β_s
    """
    alphas = 1 - betas
    alpha_bar = np.cumprod(alphas)

    # 采样噪声
    noise = np.random.randn_like(x_0)

    # 闭式解
    sqrt_alpha_bar = np.sqrt(alpha_bar[t])
    sqrt_one_minus_alpha_bar = np.sqrt(1 - alpha_bar[t])

    x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

    return x_t, noise
```

### 为什么前向过程有闭式解？

```
每一步：x_t = √(1-β_t) * x_{t-1} + √β_t * ε

展开后：
x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε

这就是闭式解！可以一步从x_0算到x_t
```

---

## 第三部分：反向过程（学习）

### 预测噪声

```python
class DDPM(nn.Module):
    """DDPM模型"""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone  # 通常是U-Net

    def forward(self, x_t, t, text_emb=None):
        """
        预测噪声

        输入：噪声图像x_t，时间步t，文本嵌入（可选）
        输出：预测的噪声
        """
        return self.backbone(x_t, t, text_emb)

    @torch.no_grad()
    def generate(self, shape, T=1000):
        """
        从纯噪声生成图像
        """
        # 从纯噪声开始
        x_t = torch.randn(shape)

        # 逐步去噪
        for t in reversed(range(T)):
            # 预测噪声
            noise_pred = self.forward(x_t, t)

            # 计算去噪后的图像
            # （简化版，实际需要更复杂的采样）
            alpha = 1 - betas[t]
            alpha_bar = np.prod(1 - betas[:t+1])

            x_t = (x_t - np.sqrt(1-alpha_bar) * noise_pred) / np.sqrt(alpha_bar)

        return x_t
```

---

## 第四部分：DDPM训练

```python
def train_ddpm(model, dataloader, optimizer, epochs=100):
    """DDPM训练"""
    for epoch in range(epochs):
        for batch in dataloader:
            x_0 = batch  # 真实图像

            # 随机采样时间步
            t = torch.randint(0, 1000, (x_0.size(0),))

            # 前向加噪（直接得到x_t）
            x_t, noise = forward_diffusion_torch(x_0, t, betas)

            # 预测噪声
            noise_pred = model(x_t, t)

            # MSE损失
            loss = nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

---

## 第五部分：DDPM vs GAN

| 对比 | DDPM | GAN |
|------|------|-----|
| 训练 | 稳定 | 不稳定 |
| 采样 | 慢（多步）| 快（一步）|
| 分布覆盖 | 完整 | 可能有mode collapse |
| 质量 | 高 | 高 |
| 灵活性 | 可控生成 | 较难控制 |

---

## ✅ 小结

1. **DDPM**通过两个过程：前向扩散（加噪）和反向去噪
2. 前向过程有闭式解，可直接计算任意时刻
3. 反向过程通过U-Net学习去噪
4. 训练稳定，但采样慢
5. 是Stable Diffusion等模型的基础

---

_继续学习：下一章「Stable Diffusion」_
