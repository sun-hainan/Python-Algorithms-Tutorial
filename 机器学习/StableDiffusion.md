# Stable Diffusion

_在隐空间舞蹈的图像生成器_

---

## 📖 学习目标

1. 理解Latent Diffusion的核心思想
2. 理解CLIP、VAE、U-Net三大组件
3. 掌握文生图的工作流程

---

## 第一部分：为什么需要Latent Diffusion？

```
普通Diffusion的问题：
- 在像素空间操作
- 512×512×3 = 786,432 像素
- 每一步计算巨大，非常慢！

Latent Diffusion：
- 先用VAE压缩到隐空间
- 512×512×3 → 64×64×4（压缩48倍！）
- 在隐空间做扩散
- 最后用VAE解码
```

---

## 第二部分：三大组件

```
1. CLIP Text Encoder
   "a cute cat" → 文本嵌入向量

2. U-Net + Scheduler
   噪声隐向量 + 文本嵌入 → 预测噪声

3. VAE
   编码器：图像 → 隐向量
   解码器：隐向量 → 图像
```

---

## 第三部分：文生图流程

```python
def text_to_image(prompt, pipe):
    # 1. CLIP编码文本
    text_emb = pipe.encode_prompt(prompt)

    # 2. 初始化纯噪声（隐空间）
    latents = torch.randn(1, 4, 64, 64)

    # 3. 迭代去噪
    for t in pipe.timesteps:
        noise_pred = pipe.unet(latents, t, text_emb)
        latents = scheduler.step(noise_pred, t, latents)

    # 4. VAE解码
    image = pipe.vae.decode(latents)
    return image
```

---

## 第四部分：ControlNet

```
ControlNet：用额外条件控制生成

输入：边缘图、姿态图、深度图
输出：符合条件的新图像
```

---

## ✅ 小结

1. **Stable Diffusion**在隐空间做扩散，大幅加速
2. CLIP理解文本，U-Net去噪，VAE编解码
3. ControlNet让生成更可控

---

_继续学习：下一章「LoRA」_
