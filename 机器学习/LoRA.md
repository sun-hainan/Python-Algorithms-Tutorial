# LoRA

_低秩适配：高效微调大模型的方法_

---

## 第一部分：LoRA 概述

### 核心思想

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

```
原始权重矩阵: W_0 ∈ R^(d×k)

LoRA 添加: ΔW = BA
其中 B ∈ R^(d×r), A ∈ R^(r×k), r << min(d,k)

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
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x):
        original = nn.functional.linear(x, self.weight)
        lora = x @ self.lora_A.T @ self.lora_B.T * self.scaling
        return original + lora
```

---

## 第四部分：QLora

```python
"""
QLoRA = Quantization + LoRA
把预训练模型量化为 4-bit，只用 LoRA 微调
"""

from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

---

## 小结

1. **LoRA**通过低秩分解大幅减少可训练参数
2. 冻结原模型权重，只训练 A、B 矩阵
3. 参数量从 d×k 降到 (d+k)×r
4. 可与量化结合（QLoRA）进一步节省显存
5. 广泛应用于 LLM 和 Stable Diffusion 的微调

---

_生成模型部分结束！_
