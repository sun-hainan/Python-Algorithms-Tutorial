# Stable Diffusion

_在隐空间舞蹈的图像生成器_

---

## 📖 学习目标

1. 理解决Stable Diffusion的核心思想
2. 理解CLIP、VAE、U-Net三大组件
3. 掌握文生图的工作流程

---

## 第一部分：为什么需要Latent Diffusion？

```
普通Diffusion的问题：
- 在像素空间操作
- 512×512×3 = 786,432 像素
- 每一步都要在这个巨大的空间计算
- 非常慢！

Latent Diffusion的解决方案：
- 先用VAE压缩到隐空间
- 512×512×3 → 64×64×4 = 16,384（压缩48倍！）
- 在隐空间做扩散
- 最后用VAE解码回像素空间
```

---

## 第二部分：三大组件

### 1. CLIP Text Encoder

```
作用：把文本转换成模型能理解的向量

"a cute cat" → [0.1, 0.5, -0.3, ...]

让生成过程"听懂"你要什么
```

### 2. U-Net + Scheduler

```
作用：在隐空间去噪

输入：噪声隐向量 + 文本嵌入
输出：预测的噪声

Scheduler控制噪声调度的策略
```

### 3. VAE

```
编码器：图像 → 隐向量（压缩）
解码器：隐向量 → 图像（重建）
```

---

## 第三部分：文生图流程

```python
def text_to_image(prompt, pipe):
    """
    文生图流程
    """
    # 1. 文本编码
    text_embeddings = pipe.encode_prompt(prompt)

    # 2. 初始化纯噪声（隐空间）
    latents = torch.randn(1, 4, 64, 64)  # 64×64×4隐空间

    # 3. 迭代去噪
    for t in pipe.timesteps:
        # U-Net预测噪声
        noise_pred = pipe.unet(latents, t, text_embeddings)

        # 更新隐向量
        latents = scheduler.step(noise_pred, t, latents)

    # 4. VAE解码
    image = pipe.vae.decode(latents)

    return image
```

---

## 第四部分：ControlNet

```
ControlNet：用额外条件控制生成

输入：边缘图、姿态图、深度图等
输出：符合条件的新图像

让生成更可控！
```

---

## ✅ 小结

1. **Stable Diffusion**在隐空间做扩散，大幅加速
2. CLIP提供文本理解，U-Net负责去噪，VAE负责编解码
3. 通过Scheduler控制采样步数和策略
4. ControlNet等扩展让控制更精细

---

_继续学习：下一章「LoRA」_
