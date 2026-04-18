# Stable Diffusion

_开源的图像生成扩散模型_

---

## 📖 学习目标

1. 理理解 Stable Diffusion 的架构
2. 掌握 Latent Diffusion 的核心思想
3. 了解文生图的工作流程

---

## 第一部分：Stable Diffusion 概述

### 🎯 核心思想

**在隐空间进行扩散，大幅降低计算成本。**

```
普通 Diffusion：在像素空间进行
  x_0 (64x64x3) → x_1 → ... → x_T (纯噪声)

Stable Diffusion：在压缩的隐空间进行
  z_0 (8x8x4) → z_1 → ... → z_T (纯噪声)
  然后用 VAE 解码成图像

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

## 第三部分：Latent Diffusion 原理

```python
class LatentDiffusionModel:
    """隐扩散模型"""
    def __init__(self):
        # 1. 文本编码器
        self.text_encoder = CLIPTextEncoder()

        # 2. VAE（压缩图像到隐空间）
        self.vae = AutoencoderKL()

        # 3. U-Net（去噪）
        self.unet = UNet()

        # 4. 调度器
        self.scheduler = DDPMScheduler()

    @torch.no_grad()
    def generate(self, prompt, num_inference_steps=50):
        """
        文生图流程

        1. 文本编码
        2. 初始化纯噪声
        3. 迭代去噪
        4. VAE 解码
        """
        # 1. 文本编码
        text_embeddings = self.text_encoder.encode(prompt)

        # 2. 初始化噪声
        latents = torch.randn(1, 4, 64, 64)  # 隐向量形状

        # 3. 迭代去噪
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.scheduler.timesteps:
            # U-Net 预测噪声
            noise_pred = self.unet(latents, t, text_embeddings)

            # 采样
            latents = self.scheduler.step(noise_pred, t, latents)

        # 4. VAE 解码
        image = self.vae.decode(latents)

        return image
```

---

## 第四部分：各组件详解

### 1. CLIP Text Encoder

```python
from transformers import CLIPTextModel

class CLIPTextEncoder:
    """CLIP 文本编码器"""
    def __init__(self):
        self.model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

    def encode(self, text):
        """文本 → embedding"""
        inputs = tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.last_hidden_state  # (batch, seq_len, 768)
```

### 2. VAE

```python
from diffusers import AutoencoderKL

class VAE:
    """变分自编码器"""
    def __init__(self):
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            subfolder="vae"
        )

    def encode(self, image):
        """图像 → 隐向量（压缩）"""
        with torch.no_grad():
            latent = self.vae.encode(image).latent_dist.sample()
            latent = latent * 0.18215  # scaling factor
        return latent

    def decode(self, latent):
        """隐向量 → 图像（重建）"""
        latent = latent / 0.18215
        with torch.no_grad():
            image = self.vae.decode(latent).sample
        return image
```

### 3. Scheduler（调度器）

```python
class DDIMScheduler:
    """DDIM 调度器：加速采样"""
    def __init__(self, num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012):
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps) ** 2
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def set_timesteps(self, num_inference_steps):
        """设置推理步数"""
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.linspace(999, 0, num_inference_steps)

    def step(self, model_output, timestep, sample):
        """单步去噪"""
        t = timestep
        pred_original_sample = (
            sample - self.alpha_bar[t] ** 0.5 * model_output
        ) / (1 - self.alpha_bar[t]) ** 0.5

        # DDIM 采样公式
        # ...
```

---

## 第五部分：ControlNet 控制生成

```python
class ControlNet:
    """ControlNet：用额外条件控制生成"""
    def __init__(self):
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny"
        )

    def generate_with_pose(self, pose_image, prompt):
        """
        根据姿态图像生成对应人物
        """
        # 检测边缘/姿态
        canny_image = detect_canny(pose_image)

        # 带条件的生成
        output = pipe(
            prompt=prompt,
            image=canny_image,  # 控制条件
            strength=0.8  # 条件强度
        )

        return output.images[0]
```

---

## 第六部分：LoRA 微调

```python
# 用 LoRA 高效微调 Stable Diffusion
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model

# 加载模型
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")

# 配置 LoRA
lora_config = LoraConfig(
    r=16,  # 秩
    lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v", "to_out"]
)

# 应用 LoRA
pipe.unet = get_peft_model(pipe.unet, lora_config)

# 训练（只需训练 LoRA 参数）
trainer = LoRATrainer(...)
trainer.train()

# 保存 LoRA
pipe.save_pretrained("./lora_weights")
```

---

## 第七部分：实际应用

### 文生图

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

prompt = "a cute cat sitting on a sofa, photorealistic, 8k"
image = pipe(prompt, num_inference_steps=20).images[0]
image.save("cat.png")
```

### 图生图

```python
from diffusers import StableDiffusionImg2ImgPipeline

pipe = StableDiffusionImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

init_image = Image.open("sketch.png").convert("RGB")
prompt = "a beautiful landscape, oil painting"
image = pipe(prompt=prompt, image=init_image, strength=0.75).images[0]
```

---

## 第八部分：名词解释

### Latent Space

```
定义：VAE 压缩后的隐空间，比像素空间小 64 倍

好处：计算更高效，生成更平滑
```

### Guidance Scale

```
定义：文本条件的引导强度

- 值越大：越严格遵循 prompt
- 值太小：可能偏离 prompt
- 通常 7-12 比较好
```

---

## ✅ 小结

1. **Stable Diffusion**在隐空间做扩散，大幅降低计算量
2. 三大组件：CLIP（文本）、U-Net（去噪）、VAE（编解码）
3. 通过 Scheduler 控制去噪过程
4. 可用 LoRA 高效微调
5. 应用：文生图、图生图、ControlNet 控制

---

_继续学习：下一章「LoRA」_
