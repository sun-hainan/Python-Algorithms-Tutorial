# LoRA

_低秩适配：高效微调大模型的方法_

---

## 📖 学习目标

1. 理解决LoRA的核心思想
2. 掌握低秩分解的原理
3. 能用LoRA微调自己的模型

---

## 第一部分：LoRA 概述

### 🎯 核心思想

**不微调全部参数，只微调低秩分解后的少量参数。**

```
全参数微调的问题：
- GPT-3 有 1750 亿参数
- 微调需要更新所有参数
- 显存和存储成本极高

LoRA 的解决方案：
- 冻结原模型权重
- 添加少量新参数
- 只训练新参数
```

---

## 第二部分：低秩分解

### 数学原理

```
原始权重矩阵: W_0 ∈ R^(d×k)

LoRA 添加: ΔW = BA
其中 B ∈ R^(d×r), A ∈ R^(r×k), r << min(d,k)

更新公式: W = W_0 + ΔW = W_0 + BA

参数量对比:
- 全量: d × k
- LoRA: (d + k) × r

例如: d=4096, k=4096, r=8
- 全量: 16,777,216 参数
- LoRA: 65,536 参数
- 压缩比: 256 倍！
```

---

## 第三部分：代码实现

```python
import torch
import torch.nn as nn
import numpy as np

class LoRALinear(nn.Module):
    """带 LoRA 的线性层"""
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 冻结原始权重
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features),
            requires_grad=False
        )

        # LoRA 新增的可训练参数
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x):
        """
        原权重 + LoRA 权重
        W = W_0 + BA * scaling
        """
        original = torch.nn.functional.linear(x, self.weight)
        lora = x @ self.lora_A.T @ self.lora_B.T * self.scaling
        return original + lora


class LoRALayer:
    """通用 LoRA 实现"""
    def __init__(self, model, rank=4, alpha=1.0, target_modules=None):
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules or ['q_proj', 'v_proj']

        # 替换目标层
        self.lora_layers = {}
        self.original_layers = {}

        for name, module in model.named_modules():
            if any(tm in name for tm in self.target_modules):
                if isinstance(module, nn.Linear):
                    # 保存原始层
                    self.original_layers[name] = module

                    # 创建 LoRA 版本
                    lora_layer = LoRALinear(
                        module.in_features,
                        module.out_features,
                        rank=rank,
                        alpha=alpha
                    )
                    lora_layer.weight.data = module.weight.data.clone()
                    lora_layer.weight.requires_grad = False

                    self.lora_layers[name] = lora_layer

                    # 替换
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    setattr(model.get_submodule(parent_name), child_name, lora_layer)
```

---

## 第四部分：训练 LoRA

```python
def train_with_lora(model, train_loader, rank=4, alpha=1.0, lr=1e-4, epochs=3):
    """
    用 LoRA 训练模型
    """
    # 1. 应用 LoRA
    lora_layer = LoRALayer(model, rank=rank, alpha=alpha)

    # 2. 只优化 LoRA 参数
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr
    )

    # 3. 训练
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            outputs = model(batch['input'])
            loss = nn.functional.cross_entropy(outputs, batch['labels'])

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: Loss = {total_loss / len(train_loader):.4f}")

    return model


def save_lora_weights(model, path):
    """只保存 LoRA 权重"""
    lora_state = {}
    for name, param in model.named_parameters():
        if 'lora' in name:
            lora_state[name] = param
    torch.save(lora_state, path)
    print(f"保存 LoRA 权重到 {path}")


def load_lora_weights(model, path):
    """加载 LoRA 权重"""
    lora_state = torch.load(path)
    model.load_state_dict(lora_state, strict=False)
    return model
```

---

## 第五部分：QLora - 更高效的版本

```python
"""
QLoRA = Quantization + LoRA

思想：
1. 把预训练模型量化为 4-bit
2. 只用 LoRA 微调少量参数
3. 极大减少显存占用
"""

from transformers import BitsAndBytesConfig

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # Normal Float 4
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True  # 双重量化
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantization_config=bnb_config,
    device_map="auto"
)

# 应用 LoRA
model = get_peft_model(model, lora_config)

# 训练
trainer = transformers.Trainer(...)
trainer.train()
```

---

## 第六部分：LoRA 在 Stable Diffusion 中的应用

```python
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model

# 1. 加载预训练模型
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
)

# 2. 配置 LoRA（针对 attention 层）
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v", "to_out"],
    lora_dropout=0.05
)

# 3. 应用 LoRA
pipe.unet = get_peft_model(pipe.unet, lora_config)

# 4. 准备数据
# 需要准备风格图像和描述文本

# 5. 训练
# 使用 DreamBooth 或 Textual Inversion 技术微调

# 6. 保存
pipe.save_pretrained("./my_style_lora")
```

---

## 第七部分：其他微调方法对比

| 方法 | 参数量 | 显存 | 速度 | 效果 |
|------|--------|------|------|------|
| 全量微调 | 100% | 非常高 | 慢 | 最好 |
| LoRA | ~1% | 高 | 快 | 很好 |
| QLoRA | ~0.5% | 中等 | 中 | 很好 |
| Adapter | ~1-5% | 中 | 快 | 一般 |
| Prefix Tuning | ~0.1% | 低 | 快 | 一般 |

---

## 第八部分：名词解释

### 秩 (Rank)

```
定义：矩阵中线性无关的行/列的最大数量

LoRA 中：
- r 越小，压缩越多，参数量越少
- r 越大，表达能力越强，效果越好
- 通常 r=4~16 比较常用
```

### 秩亏 (Rank Deficiency)

```
定义：当矩阵的秩小于其维度时

LoRA 假设：大模型的权重更新是低秩的
即 ΔW 可以用低秩矩阵近似
```

---

## ✅ 小结

1. **LoRA**通过低秩分解大幅减少可训练参数
2. 冻结原模型权重，只训练 A、B 矩阵
3. 参数量从 d×k 降到 (d+k)×r
4. 可与量化结合（QLoRA）进一步节省显存
5. 广泛应用于 LLM 和 Stable Diffusion 的微调

---

_生成模型部分结束！_
