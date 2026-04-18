# LoRA

_高效微调大模型的秘诀_

---

## 📖 学习目标

1. 理解决LoRA的核心思想
2. 理解低秩分解的原理
3. 掌握LoRA的应用场景

---

## 第一部分：全量微调的问题

```
大模型的参数量：
- GPT-2: 15亿
- GPT-3: 1750亿
- LLaMA-2: 700亿

全量微调的问题：
- 需要更新所有参数
- 显存巨大
- 存储多个微调版本成本高
```

---

## 第二部分：LoRA的核心思想

### 低秩分解

```
原始权重：W₀ ∈ R^(d×k)

LoRA添加：ΔW = BA
其中 B ∈ R^(d×r), A ∈ R^(r×k)
r << min(d, k)

更新后：W = W₀ + ΔW = W₀ + BA

参数量：
- 全量：d × k
- LoRA：(d + k) × r

例如：d=4096, k=4096, r=8
- 全量：16,777,216
- LoRA：65,536
- 压缩256倍！
```

---

## 第三部分：重参数化

```python
class LoRALinear(nn.Module):
    """LoRA线性层"""
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 冻结原始权重
        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)

        # LoRA的可训练参数
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x):
        # 原始输出 + LoRA输出
        original = nn.functional.linear(x, self.weight)
        lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return original + lora
```

---

## 第四部分：QLoRA

```
QLoRA = Quantization + LoRA

把预训练模型量化为4-bit，大幅减少显存
然后只训练LoRA参数

效果：
- 65B的模型可以在单卡80G显存微调
- 训练速度和质量都很好
```

---

## 第五部分：应用

### 微调LLM

```python
# 用LoRA微调ChatGLM
from peft import LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("chatglm-6b")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query_key_value", "dense"],
    lora_dropout=0.05
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 只有0.1%的参数是可训练的！
```

### 微调Stable Diffusion

```python
from diffusers import StableDiffusionPipeline
from peft import LoraConfig

pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-2-1")

# LoRA只加在attention层
lora_config = LoraConfig(r=16, target_modules=["to_q", "to_k", "to_v", "to_out"])
pipe.unet = get_peft_model(pipe.unet, lora_config)

# 训练LoRA...
```

---

## ✅ 小结

1. **LoRA**通过低秩分解大幅减少可训练参数
2. 冻结原模型，只训练A、B矩阵
3. 参数量从 d×k 降到 (d+k)×r
4. 结合量化（QLoRA）进一步节省显存
5. 广泛用于LLM和Stable Diffusion的微调

---

_继续学习：下一章「梯度下降」_
