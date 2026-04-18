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
大模型参数量：
- GPT-2: 15亿
- GPT-3: 1750亿

全量微调：更新所有参数，显存巨大
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
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features), requires_grad=False)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

    def forward(self, x):
        original = nn.functional.linear(x, self.weight)
        lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return original + lora
```

---

## 第四部分：QLoRA

```
QLoRA = Quantization + LoRA

把模型量化为4-bit，只训练LoRA参数
65B模型可在单卡80G显存微调！
```

---

## 第五部分：应用

```python
from peft import LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("chatglm-6b")
lora_config = LoraConfig(r=8, target_modules=["query_key_value"])
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 只有0.1%的参数可训练！
```

---

## ✅ 小结

1. **LoRA**通过低秩分解大幅减少可训练参数
2. 冻结原模型，只训练A、B矩阵
3. 结合量化（QLoRA）进一步节省显存

---

_机器学习详细教程完结！_
