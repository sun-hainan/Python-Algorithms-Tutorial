# Stable Diffusion

> _在隐空间中舞蹈的图像生成器，让AI画图走进千家万户_

---

## 🎯 先看一个生活中的例子

### 例子：建筑师的草图

假设你是一位建筑师，客户说："给我设计一个有未来感的图书馆。"

你会怎么做？

```
方案1（传统）：
- 你从零开始画
- 画错了要擦掉重画
- 非常耗时

方案2（Stable Diffusion 思路）：
1. 你先画一个简单的草图（压缩信息）
2. 在草图上修改细节
3. 把草图变成完整的建筑图

Stable Diffusion 就是在"隐空间"中画草图，
比在像素空间中操作快得多！
```

---

## 🤔 为什么需要 Latent Diffusion？

### 普通 Diffusion 的问题

```
普通 DDPM 的问题：
- 在像素空间操作
- 512×512×3 的 RGB 图像 = 786,432 像素
- 每一步都要处理这么多像素
- 非常慢！需要高端 GPU！
```

### Latent Diffusion 的解决方案

```
关键思想：在"隐空间"中做扩散！

1. 用 VAE 把图像压缩到隐空间
   - 512×512×3 → 64×64×4（压缩 48 倍！）
   - 像素数从 786K 降到 16K

2. 在隐空间做扩散
   - 处理 64×64 的隐向量，比 512×512 快很多

3. 最后用 VAE 解码回像素空间
   - 把隐向量变回图像
```

### 速度对比

| 方法 | 图像尺寸 | 处理像素 | 单步耗时 | 总耗时（50步）|
|------|---------|---------|---------|--------------|
| 像素级 DDPM | 512×512 | 786K | 10秒 | 500秒 |
| Latent DDPM | 64×64 | 16K | 0.1秒 | 5秒 |

**速度提升 100 倍！**

---

## 🏗️ Stable Diffusion 的三大组件

### 1. CLIP Text Encoder

```
作用：把文字转换成向量，让模型"看懂"文字

"A cute cat sitting on a couch"
    ↓ CLIP
[0.12, -0.34, 0.56, ...]  ← 文本嵌入向量
```

### 2. U-Net + Scheduler

```
作用：在隐空间中执行去噪

输入：
- 当前的隐向量（噪声图）
- 文本嵌入（条件信息）

输出：
- 预测的噪声

Scheduler 控制噪声去除的节奏
```

### 3. VAE（自动编码器）

```
Encoder：图像 → 隐向量
Decoder：隐向量 → 图像
```

---

## 📐 Stable Diffusion 的工作流程

### 文生图（Text-to-Image）

```
Step 1: 文本编码
"一只可爱的猫" → CLIP → 文本向量

Step 2: 初始化
随机噪声（隐空间）→ 作为起点

Step 3: 迭代去噪（通常 20-50 步）
for each step:
    noise_pred = U-Net(noise, text_vector, step)
    noise = noise - noise_pred

Step 4: VAE 解码
隐向量 → VAE → 最终图像
```

---

## 💻 代码实现

### 使用 Hugging Face Diffusers

```python
from diffusers import StableDiffusionPipeline
import torch

# 加载预训练模型
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# 生成图像
prompt = "a cute cat sitting on a couch, photorealistic, 8k"
image = pipe(
    prompt,
    num_inference_steps=50,  # 去噪步数
    guidance_scale=7.5,       # 文本引导强度
    height=512,
    width=512
).images[0]

# 保存
image.save("generated_cat.png")
```

### 引导图像生成（Image-to-Image）

```python
from diffusers import StableDiffusionImg2ImgPipeline

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# 输入草图
init_image = Image.open("sketch.png").convert("RGB")
init_image = init_image.resize((512, 512))

# 基于草图生成
result = pipe(
    prompt="a beautiful landscape, high quality",
    image=init_image,
    strength=0.75,  # 变化强度（0=保持原图，1=完全重画）
    guidance_scale=7.5
).images[0]

result.save("landscape.png")
```

---

## 🎨 ControlNet：精确控制生成

### 什么是 ControlNet？

```
ControlNet：用额外的条件来控制生成

输入：
- 文本描述
- 额外的条件图（边缘图、姿态图、深度图等）

输出：
- 符合条件图约束的图像
```

### 常见的 ControlNet 类型

```
1. Canny Edge
   - 输入：边缘检测图
   - 输出：符合边缘的图像

2. Pose（姿态）
   - 输入：人体骨骼图
   - 输出：摆出对应姿势的人物

3. Depth（深度）
   - 输入：深度图
   - 输出：符合深度的 3D 场景

4. Semantic Segmentation
   - 输入：语义分割图
   - 输出：符合分割的图像
```

### ControlNet 代码

```python
from controlnet import ControlNetModel
from diffusers import StableDiffusionControlNetPipeline
import torch

# 加载 ControlNet（以 Canny 为例）
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# 边缘图
canny_image = Image.open("edges.png")

# 生成
result = pipe(
    prompt="a person walking in the park",
    image=canny_image,
    controlnet_conditioning_scale=0.5
).images[0]
```

---

## 📊 Stable Diffusion 的参数

### num_inference_steps

```
去噪步数越多，质量越高，但速度越慢

推荐：
- 草图 → 20-25 步
- 正式生成 → 50 步
- 高质量 → 80-100 步
```

### guidance_scale

```
文本引导强度（CFG）

- 低值 (1-5): 更创意、自由发挥
- 中值 (7-9): 平衡
- 高值 (12-15): 严格遵循文本描述

太高会导致过度饱和和 artifacts
```

### strength（Image-to-Image）

```
变化强度

- 0.1-0.3: 微调
- 0.4-0.6: 中等变化
- 0.7-1.0: 剧烈变化，可能面目全非
```

---

## 📊 Stable Diffusion 版本

| 版本 | 特点 |
|------|------|
| v1.4 | 基础模型 |
| v1.5 | 改进版，最常用 |
| v2.0 | 新训练，改进质量 |
| SDXL | 更大模型，1024分辨率 |
| SDXL Turbo | 1-4步极速生成 |
| LCM LoRA | 4-8步快速生成 |

---

## ✅ 本章小结

| 概念 | 解释 |
|------|------|
| Latent Diffusion | 在隐空间做扩散，大幅加速 |
| CLIP | 文本编码器，让模型理解文字 |
| U-Net | 去噪网络 |
| VAE | 图像和隐向量之间的转换 |
| guidance_scale | 文本引导强度 |
| ControlNet | 用额外条件精确控制生成 |

---

## 🔗 继续学习

Stable Diffusion 展示了如何高效地生成图像。接下来让我们学习如何用 LoRA 高效微调大模型！

👉 [LoRA](./LoRA.md)
