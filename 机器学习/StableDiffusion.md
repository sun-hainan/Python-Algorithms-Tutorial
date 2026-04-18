# Stable Diffusion

_开源的图像生成扩散模型_

---

## 第一部分：Stable Diffusion 概述

### 核心思想

**在隐空间进行扩散，大幅降低计算成本。**

```
普通 Diffusion：在像素空间进行
Stable Diffusion：在压缩的隐空间进行

优势：计算量减少 64 倍！
```

---

## 第二部分：架构组件

```
Stable Diffusion 三大组件：

1. Text Encoder (CLIP)
   "a cute cat" → text embeddings

2. UNet + Scheduler
   噪声隐向量 + text emb → 去噪预测

3. VAE Decoder
   隐向量 → 图像
```

---

## 第三部分：工作流程

```python
def generate_image(prompt, pipe):
    """
    文生图流程
    1. 文本编码
    2. 初始化纯噪声
    3. 迭代去噪
    4. VAE 解码
    """
    # 1. 文本编码
    text_embeddings = pipe.encode_prompt(prompt)

    # 2. 初始化噪声
    latents = torch.randn(1, 4, 64, 64)

    # 3. 迭代去噪
    for t in pipe.timesteps:
        noise_pred = pipe.unet(latents, t, text_embeddings)
        latents = pipe.scheduler.step(noise_pred, t, latents)

    # 4. VAE 解码
    image = pipe.vae.decode(latents)
    return image
```

---

## 第四部分：应用

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
prompt = "a cute cat sitting on a sofa, photorealistic"
image = pipe(prompt, num_inference_steps=20).images[0]
image.save("cat.png")
```

---

## 小结

1. **Stable Diffusion**在隐空间做扩散，大幅降低计算量
2. 三大组件：CLIP（文本）、U-Net（去噪）、VAE（编解码）
3. 通过 Scheduler 控制去噪过程
4. 可用 LoRA 高效微调

---

_继续学习：下一章「LoRA」_
