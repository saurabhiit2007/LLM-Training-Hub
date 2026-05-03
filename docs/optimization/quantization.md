# Quantization for LLM Training

## 1. Overview

**Quantization** reduces the precision of model weights, activations, and gradients from high-precision formats (FP32, BF16) to lower-precision formats (INT8, INT4) to save memory and accelerate computation.

**Key distinction:**
- **Quantization-Aware Training (QAT):** Quantization applied during training
- **Post-Training Quantization (PTQ):** Quantization applied after training (more common for LLMs)
- **Training with Quantized Components:** Using quantized optimizers, activations during training (not quantizing the final model)

This guide focuses on **quantization techniques used during LLM training** to reduce memory and enable larger models.

---

## 2. Why Quantization for Training?

### 2.1 Memory Savings

**Memory consumption by precision:**

| Precision | Bytes per Parameter | 7B Model Size |
|-----------|-------------------|---------------|
| FP32 | 4 | 28 GB |
| BF16/FP16 | 2 | 14 GB |
| INT8 | 1 | 7 GB |
| INT4 | 0.5 | 3.5 GB |

**Typical training memory (7B model, no quantization):**
- Weights (BF16): 14 GB
- Gradients (BF16): 14 GB
- Optimizer states (FP32): 56 GB
- **Total:** 84 GB (before activations)

**With quantization:**
- 8-bit optimizer states: 56 GB → 14 GB (75% reduction)
- 4-bit base model (QLoRA): 14 GB → 3.5 GB (75% reduction)

### 2.2 Speed Benefits

Lower precision enables:
- Faster memory transfers (less data movement)
- Hardware acceleration (INT8 ops faster than FP16 on some hardware)
- Larger batch sizes (more memory available)

**Caveat:** Speed gains are hardware-dependent and often modest for training.

---

## 3. Key Quantization Techniques for Training

### 3.1 Quantized Optimizers (8-bit Adam)

Store optimizer states in INT8 instead of FP32 — 4× memory reduction with no quality loss. See [Memory-Efficient Optimizers](memory_efficient_optimizers.md) for the full derivation, block-wise quantization details, and implementation.

---

### 3.2 4-bit Quantization (QLoRA)

**QLoRA** (Quantized LoRA) combines:
1. **4-bit NormalFloat (NF4)** quantization of base model
2. **LoRA adapters** (only train small rank decomposition matrices)
3. **Double quantization** (quantize the quantization constants)
4. **Paged optimizers** for memory efficiency

#### 3.2.1 NormalFloat 4-bit (NF4)

**Key insight:** Neural network weights follow a normal distribution, not uniform.

**Standard quantization (uniform):**
- Divides range into equal intervals
- Wastes precision where weights are dense (near zero)

**NF4 (information-theoretic optimal):**
- Bins chosen to have equal number of weights per bin
- More precision near zero (where most weights are)
- Less precision at extremes

**NF4 quantization bins (16 values for 4-bit):**

Designed so that bins have equal expected number of values from a standard normal distribution.

**Example bins:**
```
[-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
  0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0]
```

#### 3.2.2 Double Quantization

**Problem:** Even quantization constants (scale factors) consume memory.

**Solution:** Quantize the quantization constants themselves.

**Example:**
- Block size: 64 parameters
- Each block needs a FP32 scale factor (4 bytes)
- For 7B parameters: $\frac{7 \times 10^9}{64} \times 4 = 437$ MB just for scales
- Double quantization: 437 MB → 109 MB (quantize scales to 8-bit)

**Total memory savings:**
- Base model: 14 GB → 3.5 GB (4-bit weights)
- Quantization constants: 0.44 GB → 0.11 GB (double quantization)

#### 3.2.3 QLoRA Memory Breakdown

**Fine-tuning 7B model with QLoRA:**

| Component | Memory |
|-----------|--------|
| Base model (4-bit NF4) | 3.5 GB |
| LoRA adapters (trainable) | ~0.2 GB (rank 16) |
| Gradients (LoRA only) | ~0.2 GB |
| Optimizer states (paged 8-bit) | ~0.4 GB |
| Activations (checkpointed) | ~2 GB |
| **Total** | **~6.3 GB** |

**Enables 7B fine-tuning on consumer GPUs (RTX 3090, RTX 4090 with 24GB).**

**Implementation:**

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,      # Double quantization
    bnb_4bit_quant_type="nf4",           # NormalFloat 4-bit
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load base model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# Add LoRA adapters (only these are trained)
lora_config = LoraConfig(
    r=16,                    # LoRA rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)

model = get_peft_model(model, lora_config)
```

---

### 3.3 Block-wise k-bit Quantization

**Concept:** Divide weights into blocks and quantize each block independently with its own scale factor.

**Why block-wise?**
- Different layers have different weight magnitudes
- Even within a layer, weight distributions vary
- Per-tensor quantization loses too much precision
- Per-parameter quantization is too expensive

**How it works:**

1. **Divide tensor into blocks** (typically 64-256 elements per block)
2. **Compute scale factor per block:**
   $$
   s = \frac{\max(|w_{\text{block}}|)}{2^{k-1} - 1}
   $$
   Where $k$ is number of bits (e.g., 4 or 8)

3. **Quantize each element:**
   $$
   w_{\text{quant}} = \text{round}\left(\frac{w}{s}\right)
   $$

4. **Store:** Quantized weights + scale factors

**Example (8-bit, block size 64):**

```python
def quantize_blockwise(tensor, block_size=64, bits=8):
    """
    Block-wise quantization to k-bit integers.
    """
    # Reshape into blocks
    numel = tensor.numel()
    tensor_flat = tensor.flatten()
    
    # Pad to multiple of block_size
    padding = (block_size - numel % block_size) % block_size
    if padding > 0:
        tensor_flat = torch.cat([tensor_flat, torch.zeros(padding)])
    
    tensor_blocks = tensor_flat.reshape(-1, block_size)
    
    # Compute per-block scale (absmax)
    absmax = tensor_blocks.abs().max(dim=1, keepdim=True).values
    scale = absmax / (2**(bits-1) - 1)
    
    # Avoid division by zero
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    
    # Quantize
    max_int = 2**(bits-1) - 1
    min_int = -2**(bits-1)
    quantized = (tensor_blocks / scale).round().clamp(min_int, max_int)
    
    return quantized.to(torch.int8), scale, numel

def dequantize_blockwise(quantized, scale, original_numel):
    """
    Reconstruct FP32 tensor from quantized blocks.
    """
    dequantized = quantized.float() * scale
    dequantized_flat = dequantized.flatten()[:original_numel]
    return dequantized_flat
```

**Trade-off: Block size**

| Block Size | Precision | Memory Overhead | Best For |
|------------|-----------|----------------|----------|
| Small (16-32) | High | High (more scales) | Critical layers |
| Medium (64-128) | Medium | Medium | General use |
| Large (256-512) | Low | Low | Less critical layers |

**Typical choice:** 64 for 8-bit, 128 for 4-bit.

---

### 3.4 Mixed-Precision Quantization

**Concept:** Use different precisions for different components.

**Common patterns:**

| Component | Precision | Reason |
|-----------|-----------|--------|
| **Forward pass** | INT8/BF16 | Speed + memory |
| **Backward pass** | BF16/FP32 | Gradient precision |
| **Weights** | INT4/INT8 | Memory savings |
| **Activations** | BF16 | Numerical stability |
| **Optimizer states** | INT8 | Memory savings |
| **Master weights** | FP32 | Accumulate updates accurately |

**QLoRA example:**
- Base model: 4-bit NF4
- LoRA adapters: BF16
- Gradients: BF16
- Optimizer states: 8-bit
- Computation: BF16 (dequantize on-the-fly)

---

## 4. Quantization Challenges and Solutions

### 4.1 Outlier Features

**Problem:** A few extreme values (outliers) dominate the quantization range, forcing lower precision for majority of values.

**Example:**
- 99% of weights in range [-1, 1]
- 1% of weights in range [-10, 10]
- Quantization range must cover [-10, 10], wasting precision on the 99%

**Solutions:**

**1. Per-channel quantization:**
- Separate scale factor per output channel
- Isolates outliers to specific channels

**2. Outlier extraction (LLM.int8()):**
- Keep outlier weights in FP16
- Quantize rest to INT8
- Mixed precision matmul

**3. SmoothQuant:**
- Migrate difficulty from weights to activations
- Apply scaling to smooth distributions

### 4.2 Gradient Precision

**Problem:** Gradients during backprop need higher precision than forward pass.

**Solution:** Always use at least BF16 for gradients, even if weights are 4-bit.

**QLoRA approach:**
- 4-bit weights dequantized to BF16 before computation
- Gradients computed in BF16
- Only LoRA adapter gradients exist (base model frozen)

### 4.3 Quantization Noise Accumulation

**Problem:** Quantization error accumulates over training steps.

**Mitigation:**
- Use higher precision for optimizer states (even if weights are quantized)
- Maintain FP32 master weights
- Use larger learning rates to overcome noise
- Monitor training closely for divergence

---

## 5. Quantization Variants Comparison

### 5.1 Overview Table

**Training-time quantization:**

| Technique | Bits | What's Quantized | Memory Savings | Quality Loss | Use Case |
|-----------|------|-----------------|----------------|--------------|----------|
| **8-bit Adam** | 8 | Optimizer states | 75% (optimizer) | None | Standard training |
| **QLoRA** | 4 | Base model | 75% (model) | Minimal (LoRA) | Fine-tuning on consumer GPUs |
| **LLM.int8()** | 8 | Weights + acts | 50% | <1% | Inference with bitsandbytes |

Post-training inference quantization methods (GPTQ, AWQ, GGUF) are covered in the [LLM Inference Speed](../../LLM-Inference-Speed/docs/quantization/) repo.

### 5.2 When to Use Each

```
Memory not constrained (80GB+ GPU):
└─> Standard BF16 training with 8-bit Adam (optional)

Memory constrained (24-40GB GPU):
└─> QLoRA (4-bit base + LoRA adapters)

Extremely constrained (<16GB):
└─> QLoRA + gradient checkpointing + small batch size
```

---

## 7. Summary

**Key Takeaways:**

### Quantization for Training
- **8-bit Adam:** 4× optimizer state memory reduction, no quality loss, widely used
- **QLoRA:** 4-bit base model + LoRA adapters, enables fine-tuning on consumer GPUs
- **Block-wise quantization:** Sweet spot between precision and memory (block size 64-128)
- **NF4:** Information-theoretically optimal for normal distributions (neural network weights)

### Memory Savings
- 8-bit optimizer: 75% reduction in optimizer states
- 4-bit model: 75% reduction in model size
- Combined (QLoRA): ~90% total memory reduction

### Quality vs Compression
- 8-bit: No degradation for optimizer states
- 4-bit + LoRA: ~2-5% quality loss (compared to full fine-tuning)
- 4-bit full model (post-training): ~1-3% quality loss with good methods (GPTQ, AWQ)

### When to Use
- **Training large models:** 8-bit Adam (standard practice)
- **Fine-tuning on limited hardware:** QLoRA
- **Inference:** Post-training quantization (GPTQ, AWQ, not covered here)
- **Serving at scale:** Mixed precision + quantization

### Implementation Tips
1. Always use block-wise quantization (not per-tensor)
2. Start with block size 64, tune if needed
3. Monitor training loss closely for divergence
4. Keep gradients in BF16 even if weights are quantized
5. Use double quantization for 4-bit (reduces overhead)

---

## 8. Further Reading

- **8-bit Optimizers:** [8-bit Optimizers via Block-wise Quantization (2021)](https://arxiv.org/abs/2110.02861)
- **QLoRA:** [QLoRA: Efficient Finetuning of Quantized LLMs (2023)](https://arxiv.org/abs/2305.14314)
- **NF4:** [Optimal Brain Quantization (includes NF4 derivation)](https://arxiv.org/abs/2208.07339)
- **LLM.int8():** [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale (2022)](https://arxiv.org/abs/2208.07339)
- **GPTQ:** [GPTQ: Accurate Post-Training Quantization for GPT (2023)](https://arxiv.org/abs/2210.17323)
- **AWQ:** [Activation-aware Weight Quantization (2023)](https://arxiv.org/abs/2306.00978)
