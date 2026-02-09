# 🔢 Quantization for LLM Training

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

**Concept:** Store optimizer states (momentum, variance) in INT8 instead of FP32.

**How it works:**
1. During optimizer step, **dequantize** INT8 states to FP32
2. Apply Adam update in FP32
3. **Quantize** updated states back to INT8

**Memory savings:** 4× reduction in optimizer state memory (FP32 → INT8)

**Implementation (bitsandbytes):**

```python
import bitsandbytes as bnb

# Standard Adam: 56 GB for 7B model
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 8-bit Adam: 14 GB for 7B model
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-4)
```

**Why it works:**
- Optimizer states are exponential moving averages
- Accumulate slowly over many steps
- Tolerate quantization noise well

**Result:** No quality degradation in practice (validated on GPT-3 scale models).

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

| Technique | Bits | What's Quantized | Memory Savings | Quality Loss | Use Case |
|-----------|------|-----------------|----------------|--------------|----------|
| **8-bit Adam** | 8 | Optimizer states | 75% (optimizer) | None | Standard training |
| **QLoRA** | 4 | Base model | 75% (model) | Minimal (LoRA) | Fine-tuning on consumer GPUs |
| **LLM.int8()** | 8 | Weights + acts | 50% (inference) | <1% | Inference |
| **GPTQ** | 4 | Weights | 75% | 1-3% | Post-training (inference) |
| **AWQ** | 4 | Weights | 75% | <1% | Post-training (inference) |
| **Mixed 8-4** | 4-8 | Weights (mixed) | 60-70% | <1% | Inference |

**Note:** GPTQ and AWQ are post-training quantization methods, not used during training.

### 5.2 When to Use Each

**During Training:**

```
Memory not constrained (80GB+ GPU):
└─> Standard BF16 training with 8-bit Adam (optional)

Memory constrained (24-40GB GPU):
└─> QLoRA (4-bit base + LoRA adapters)

Extremely constrained (<16GB):
└─> QLoRA + gradient checkpointing + small batch size
```

**After Training (Inference):**

```
Real-time inference:
└─> AWQ (4-bit, preserves salient weights)

Batch inference:
└─> GPTQ (4-bit, good compression)

High accuracy needed:
└─> LLM.int8() (8-bit with outlier handling)
```

---

## 6. Interview Questions

### Q1: Explain how 8-bit Adam maintains training quality despite 4× less precision.

**Answer:**

**Key technique: Block-wise quantization with dynamic scaling**

8-bit Adam doesn't naively round FP32 states to INT8. Instead:

1. **Block-level adaptation:**
   - Divide optimizer states into blocks (e.g., 2048 elements)
   - Each block gets its own scale factor
   - Adapts to local value distributions

2. **Example:**
   ```
   Block 1 (early layers): values in [0.001, 0.01]
     scale₁ = 0.01 / 127 = 7.87e-5
   
   Block 2 (late layers): values in [0.1, 1.0]
     scale₂ = 1.0 / 127 = 7.87e-3
   ```
   Different blocks can represent vastly different ranges.

3. **Why it works:**
   - Optimizer states (momentum, variance) have local structure
   - Nearby parameters have similar gradient statistics
   - Block-wise quantization exploits this locality
   - Errors average out over thousands of optimizer steps

4. **Dequantization during updates:**
   - States are INT8 in memory
   - Dequantized to FP32 during Adam step
   - Updated in FP32, then re-quantized
   - No cumulative error in weight updates

**Empirical validation:** Papers show 8-bit Adam matches FP32 Adam on models up to 175B parameters.

**Follow-up:** Why not use 4-bit for optimizer states?  
**Answer:** 4-bit works for inference (single forward pass) but training accumulates quantization error over 10,000+ optimizer steps. The precision loss becomes significant and causes divergence.

---

### Q2: Walk through the memory calculation for QLoRA vs full fine-tuning of a 13B model.

**Answer:**

**Setup:** 13B parameter model, single GPU

---

**Full Fine-Tuning (BF16 + 8-bit Adam):**

| Component | Calculation | Memory |
|-----------|-------------|--------|
| Weights | $13 \times 10^9 \times 2$ | 26 GB |
| Gradients | $13 \times 10^9 \times 2$ | 26 GB |
| Adam states (8-bit) | $13 \times 10^9 \times 2$ | 26 GB |
| Activations (checkpointed) | (batch-dependent) | ~5 GB |
| **Total** | - | **~83 GB** |

Requires 80GB A100 at minimum.

---

**QLoRA (4-bit base + LoRA rank 64):**

| Component | Calculation | Memory |
|-----------|-------------|--------|
| Base model (4-bit NF4) | $13 \times 10^9 \times 0.5$ | 6.5 GB |
| Quantization overhead | ~1% of model | 0.1 GB |
| LoRA adapters (trainable) | $13B \times \frac{64}{4096} \times 2 \times 2$ | 0.8 GB |
| Gradients (LoRA only) | Same as LoRA adapters | 0.8 GB |
| Optimizer (8-bit, LoRA only) | $0.8 \times 2$ | 1.6 GB |
| Activations (checkpointed) | (smaller batch) | ~2 GB |
| **Total** | - | **~11.8 GB** |

Fits comfortably on 16GB consumer GPUs (RTX 4080, RTX 4090).

---

**Comparison:**

| Metric | Full Fine-Tuning | QLoRA |
|--------|-----------------|-------|
| GPU Memory | 83 GB | 11.8 GB |
| GPU Required | A100 80GB ($3/hr) | RTX 4090 24GB (owned) |
| Throughput | 1.0× (baseline) | 0.8× (slightly slower) |
| Final Quality | 100% | ~95-98% (LoRA limitation) |
| **Cost (100 hours)** | **$300** | **$0 (owned GPU)** |

**Key insight:** QLoRA sacrifices minimal quality for 7× memory reduction, enabling fine-tuning on consumer hardware.

**Follow-up:** Why are LoRA adapters in BF16 and not quantized?  
**Answer:** LoRA adapters are small (~1% of model) and actively trained. Quantizing them would hurt training dynamics for minimal memory savings. The 4-bit savings comes from the frozen base model.

---

### Q3: Why does NF4 outperform uniform INT4 for neural network weights?

**Answer:**

**Key insight:** Neural network weights follow a **normal distribution**, not a uniform distribution.

---

**Distribution analysis:**

After training, weights in LLMs typically have:
- Mean ≈ 0
- Standard deviation ≈ 0.1-0.5
- Most weights concentrated near zero
- Few outliers at extremes

**Uniform INT4 quantization:**

Divides the range [-max, max] into 16 equal intervals:

```
[-max, ..., -0.6max, -0.4max, -0.2max, 0, 0.2max, 0.4max, 0.6max, ..., max]
       ↑                                            ↑
    Few weights here                           Few weights here
                         ↑
                   Most weights here (wasted precision!)
```

**Problem:** Wastes many bins on the tails where few weights exist.

---

**NF4 quantization:**

Chooses bins so each bin contains **equal number of weights** (assuming normal distribution).

```
More bins near zero (high density):
[-1.0, -0.70, -0.53, -0.39, -0.28, -0.18, -0.09, 0,
  0.08, 0.16, 0.25, 0.34, 0.44, 0.56, 0.72, 1.0]
  
Notice: Bins are closer together near 0, farther apart at extremes.
```

**Benefit:**
- 99% of weights get more precision (concentrated near zero)
- 1% of outliers get less precision (acceptable)

---

**Quantitative comparison (7B LLaMA):**

| Method | Avg Quantization Error | Perplexity Increase |
|--------|----------------------|-------------------|
| FP32 (baseline) | 0 | 0% |
| Uniform INT4 | 0.023 | +5.2% |
| NF4 | 0.008 | +0.4% |

**NF4 reduces quantization error by ~3× compared to uniform INT4.**

---

**Information theory explanation:**

Optimal quantization minimizes:
$$
E = \sum_{i} p_i \cdot (x_i - q_i)^2
$$

Where $p_i$ is probability of value $x_i$, $q_i$ is quantized value.

For normal distribution, equal-probability bins minimize this error.

**Follow-up:** Could we use even more optimal bins for each layer?  
**Answer:** Yes (layer-wise quantization), but NF4 is a good universal choice that works across layers without per-layer calibration. Simplicity vs optimality trade-off.

---

### Q4: Compare block-wise quantization with per-tensor and per-channel quantization.

**Answer:**

**Three granularity levels:**

---

**1. Per-Tensor Quantization**

- **Single scale factor** for entire tensor
- Example: Whole weight matrix uses same scale

**Pros:**
- Simplest implementation
- Minimal memory overhead
- Fast quantization/dequantization

**Cons:**
- Poor precision if weight magnitudes vary across tensor
- Outliers force large scale → low precision for typical values

**Example (weight matrix with outliers):**
```
99% of weights in [-0.5, 0.5]
1% of weights in [-5.0, 5.0]

Per-tensor scale: 5.0 / 127 = 0.039
Quantization error for typical weight 0.1: ±0.039 (39% error!)
```

---

**2. Per-Channel Quantization**

- **One scale factor per output channel** (row in weight matrix)
- Example: Each row of weight matrix has its own scale

**Pros:**
- Better than per-tensor (isolates outliers to specific channels)
- Moderate memory overhead
- Common in hardware accelerators

**Cons:**
- Still suffers if outliers exist within a channel
- Not as fine-grained as block-wise

**Example (per row):**
```
Row 1: weights in [-0.5, 0.5] → scale = 0.00394
Row 2: weights in [-2.0, 2.0] → scale = 0.0157
Row 3: weights in [-5.0, 5.0] → scale = 0.0394

Better precision for Row 1, still coarse for Row 3 if it has dense values near zero.
```

---

**3. Block-wise Quantization**

- **One scale factor per block** (e.g., 64-256 consecutive elements)
- Example: Weight matrix divided into 64-element blocks

**Pros:**
- Fine-grained adaptation to local distributions
- Handles outliers well (isolated to their block)
- Best precision for given memory budget

**Cons:**
- More memory overhead (many scale factors)
- Slightly slower dequantization

**Example (64-element blocks):**
```
Block 1: weights in [-0.2, 0.2] → scale = 0.00157
Block 2: weights in [-0.5, 0.5] → scale = 0.00394
Block 3: weights in [-5.0, 5.0] → scale = 0.0394

Each block optimizes for its local distribution.
```

---

**Comparison Table:**

| Granularity | Scales per Tensor | Precision | Memory Overhead | Speed |
|-------------|------------------|-----------|----------------|-------|
| **Per-Tensor** | 1 | Low | Minimal | Fastest |
| **Per-Channel** | $C$ (num channels) | Medium | Low | Fast |
| **Block-wise** | $N/B$ (B=block size) | High | Medium | Medium |

For 7B model (weight matrix 4096×4096):
- Per-tensor: 1 scale (4 bytes)
- Per-channel: 4096 scales (16 KB)
- Block-wise (size 64): 262,144 scales (1 MB)

---

**Practical choice:**

**Training (8-bit Adam):** Block-wise (64-128) for optimizer states  
**Inference (GPTQ, AWQ):** Per-channel or hybrid  
**Hardware accelerators:** Per-channel (hardware support)

**Follow-up:** What about per-parameter quantization?  
**Answer:** Not practical—scale factors would be same size as quantized values! No memory savings. Block-wise is the sweet spot.

---

### Q5: You're training a 70B model on 8× A100s. Design a quantization strategy.

**Answer:**

**Goal:** Maximize model size and quality within 8× 80GB = 640 GB total memory.

---

**Option 1: No Quantization + ZeRO-3**

| Component | Memory per GPU |
|-----------|---------------|
| Model (BF16, sharded) | 140 GB / 8 = 17.5 GB |
| Gradients (sharded) | 17.5 GB |
| Optimizer (FP32, sharded) | 70 GB |
| Activations | ~10 GB |
| **Total** | **~115 GB per GPU** ❌ (exceeds 80 GB) |

**Doesn't fit.**

---

**Option 2: ZeRO-3 + 8-bit Adam**

| Component | Memory per GPU |
|-----------|---------------|
| Model (BF16, sharded) | 17.5 GB |
| Gradients (sharded) | 17.5 GB |
| Optimizer (8-bit, sharded) | 35 GB |
| Activations | ~10 GB |
| **Total** | **~80 GB per GPU** ✅ (barely fits!) |

**This works.**

---

**Option 3: ZeRO-3 + CPU Offload (Optimizer)**

| Component | Memory (GPU) | Memory (CPU) |
|-----------|-------------|-------------|
| Model (BF16, sharded) | 17.5 GB | 0 |
| Gradients (sharded) | 17.5 GB | 0 |
| Optimizer (FP32, offloaded) | 2 GB (buffers) | 560 GB |
| Activations | ~10 GB | 0 |
| **Total** | **~47 GB per GPU** ✅ | **70 GB per node** |

**Also works, but 2-3× slower due to CPU-GPU transfers.**

---

**Option 4: Gradient Checkpointing + 8-bit Adam**

| Component | Memory per GPU |
|-----------|---------------|
| Model (BF16, sharded) | 17.5 GB |
| Gradients (sharded) | 17.5 GB |
| Optimizer (8-bit, sharded) | 35 GB |
| Activations (checkpointed) | ~3 GB |
| **Total** | **~73 GB per GPU** ✅ |

**Best option: headroom for larger batches.**

---

**Recommended Strategy:**

```
Hardware: 8× A100 80GB

Parallelism:
├─ ZeRO-3 (shard model, gradients, optimizer)
└─ Optional: Tensor Parallelism (2-way) if needed

Quantization:
├─ 8-bit Adam optimizer states
└─ BF16 for model weights and gradients

Memory optimizations:
├─ Gradient checkpointing (every layer)
├─ Flash Attention 2 (reduce attention memory)
└─ Activation offloading (if still OOM)

Expected memory: ~73 GB per GPU
Training speed: ~85% of baseline (checkpointing overhead)
```

---

**Alternative (LoRA for faster iteration):**

If full fine-tuning is too slow for iteration:

```
Strategy: QLoRA + ZeRO-3

Components:
├─ Base model: 4-bit NF4 (35 GB / 8 = 4.4 GB per GPU)
├─ LoRA adapters: BF16 (rank 128, ~3 GB total / 8 = 0.4 GB per GPU)
├─ Gradients: LoRA only (0.4 GB per GPU)
├─ Optimizer: 8-bit (0.8 GB per GPU)
└─ Activations: checkpointed (~3 GB per GPU)

Total: ~9 GB per GPU

Benefits:
├─ 8× less memory (could train on 8× 24GB GPUs!)
├─ Faster iteration (smaller updates)
└─ Can test multiple LoRA configs simultaneously

Trade-off:
└─ ~95-98% quality vs full fine-tuning
```

---

**Decision matrix:**

| Goal | Strategy |
|------|----------|
| **Maximum quality** | Option 4 (BF16 + 8-bit Adam + checkpointing) |
| **Fast iteration** | QLoRA (can use smaller/cheaper GPUs) |
| **Minimum cost** | Option 3 (CPU offload, use cheaper CPUs) |

**For production:** Option 4 (best quality-speed balance).

**Follow-up:** What if we need to train 175B?  
**Answer:** Either (1) scale to 64 GPUs with ZeRO-3 + 8-bit Adam, or (2) use pipeline parallelism + ZeRO to split across nodes, or (3) QLoRA if LoRA adapters are acceptable.

---

### Q6: Explain the trade-off between quantization block size and precision.

**Answer:**

**Block size** in block-wise quantization determines the granularity of scale factors.

---

**Small blocks (16-32 elements):**

**Pros:**
- Very fine-grained adaptation
- Each block optimizes for local distribution
- High precision even with outliers

**Cons:**
- Many scale factors → high memory overhead
- Example: 7B params, block=32 → 219M scales = 876 MB (just for scales!)
- Slower dequantization (more scales to load)

**When to use:** Critical layers (e.g., output projection) where precision matters most.

---

**Medium blocks (64-128 elements):**

**Pros:**
- Good precision-memory balance
- Moderate number of scales
- Industry standard

**Cons:**
- May miss very local outliers

**Memory overhead example (7B params, block=64, 8-bit):**
- Scales: 7B / 64 × 4 bytes = 437 MB (~3% overhead)

**When to use:** Default choice for most scenarios.

---

**Large blocks (256-512 elements):**

**Pros:**
- Minimal memory overhead
- Fast dequantization
- Fewer scale factors to manage

**Cons:**
- Coarse adaptation
- Precision suffers if block contains diverse values
- Approaches per-channel quantization

**When to use:** Less critical layers or when memory is extremely tight.

---

**Quantitative example (weight variance within blocks):**

Consider a layer with weights ranging [-0.5, 0.5] but one outlier per 128 elements at [-5.0, 5.0]:

| Block Size | Scale (max/127) | Error on typical weight (0.1) |
|------------|----------------|-------------------------------|
| **16** | 0.5/127 = 0.004 | ±0.004 (4% error) |
| **64** | 0.5/127 = 0.004 | ±0.004 (4% error) |
| **128** | 5.0/127 = 0.039 | ±0.039 (39% error!) |
| **256** | 5.0/127 = 0.039 | ±0.039 (39% error!) |

**Small blocks isolate the outlier, large blocks spread its impact.**

---

**Optimal block size formula (rule of thumb):**

For $k$-bit quantization:
$$
B_{\text{opt}} \approx 2^{k+2} \text{ to } 2^{k+4}
$$

Examples:
- 8-bit: $B = 64$ to 256 (typically 64-128)
- 4-bit: $B = 64$ to 256 (typically 128)

---

**Practical recommendation:**

```
Default: block_size = 64
├─ Works well for 8-bit and 4-bit
├─ ~3-5% memory overhead for scales
└─ Good precision-performance balance

Tuning:
├─ If precision critical: decrease to 32
├─ If memory critical: increase to 128
└─ Monitor quantization error on validation set
```

**Follow-up:** Can we use different block sizes for different layers?  
**Answer:** Yes! Some implementations use smaller blocks for sensitive layers (e.g., final output projection) and larger blocks for less sensitive layers (e.g., early embeddings). This is a hyperparameter optimization problem.

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
