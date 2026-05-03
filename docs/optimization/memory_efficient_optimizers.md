# Memory-Efficient Optimizers for LLM Training

## 1. Overview

Training large language models requires storing not just model parameters, but also **optimizer states**—the additional memory used by optimizers like Adam to track momentum and variance. For billion-parameter models, optimizer states often consume **more memory than the model itself**.

### Memory Breakdown for a 7B Parameter Model

| Component | Size (FP32) | Size (BF16/FP16) | Multiplier |
|-----------|-------------|------------------|------------|
| Model parameters | 28 GB | 14 GB | 1× |
| Gradients | 28 GB | 14 GB | 1× |
| **Optimizer states (Adam)** | **56 GB** | **56 GB** | **2×** |
| Activations (batch-dependent) | ~40 GB | ~20 GB | Variable |
| **Total** | **~152 GB** | **~104 GB** | - |

**Key insight:** Optimizer states remain in FP32 even during mixed-precision training, making them the primary memory bottleneck.

Memory-efficient optimizers address this by:
1. **Reducing precision** of optimizer states (8-bit optimizers)
2. **Offloading** states to CPU memory (Paged Adam, ZeRO-Offload)
3. **Changing the algorithm** to use less state (Adafactor, LION)
4. **Combining approaches** (QLoRA = quantization + paging + LoRA)

---

## 2. Why Standard Adam Is Memory-Hungry

### 2.1 Adam Update Rule

For each parameter $\theta_t$ at step $t$:

$$
\begin{align}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t & \text{(first moment)} \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 & \text{(second moment)} \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t} & \text{(bias correction)} \\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^t} & \text{(bias correction)} \\
\theta_t &= \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{align}
$$

### 2.2 Memory Requirements

For **N parameters**, Adam stores:
- Parameters $\theta$: **4N bytes** (FP32) or **2N bytes** (BF16)
- First moment $m$: **4N bytes** (always FP32)
- Second moment $v$: **4N bytes** (always FP32)

**Total optimizer state memory: 8N bytes**, regardless of parameter precision.

**Example:**
- 7B model in BF16: 14 GB parameters + 56 GB optimizer states = **70 GB** just for training setup
- Without optimizer states: 14 GB parameters + 14 GB gradients = **28 GB**

This 2.5× memory overhead makes standard Adam prohibitive for large models.

---

## 3. Memory-Efficient Optimizer Strategies

### 3.1 Taxonomy

```
Memory-Efficient Optimizers
│
├── Precision Reduction
│   ├── 8-bit Adam (bitsandbytes)
│   ├── FP8 Adam (Transformer Engine)
│   └── 4-bit Adam (QLoRA)
│
├── State Offloading
│   ├── Paged Adam (bitsandbytes)
│   ├── ZeRO-Offload (DeepSpeed)
│   └── CPUAdam (DeepSpeed)
│
├── Algorithmic Changes
│   ├── Adafactor (Google)
│   ├── LION (Google)
│   └── SM3 (Google)
│
└── Hybrid Approaches
    ├── QLoRA (4-bit + Paging + LoRA)
    └── ZeRO-Infinity (offload + NVMe)
```

---

## 4. Precision Reduction: 8-bit Optimizers

### 4.1 Core Idea

Store optimizer states in **8-bit integers** instead of 32-bit floats, reducing memory by **4×**.

**Challenge:** Naive quantization destroys training dynamics.

**Solution (Dynamic Block-wise Quantization):**
1. Divide optimizer states into blocks (e.g., 2048 elements)
2. Compute block-specific scaling factors
3. Quantize each block independently
4. Dequantize during optimizer step

### 4.2 Block-wise Quantization Example

```python
# Simplified 8-bit quantization
def quantize_blockwise(tensor, block_size=2048):
    blocks = tensor.reshape(-1, block_size)
    absmax = blocks.abs().max(dim=1, keepdim=True).values
    scale = absmax / 127.0  # INT8 range
    
    quantized = (blocks / scale).round().clamp(-128, 127).to(torch.int8)
    return quantized, scale

def dequantize_blockwise(quantized, scale):
    return (quantized.float() * scale).flatten()
```

### 4.3 Implementation (bitsandbytes)

```python
import bitsandbytes as bnb

# Standard Adam (56 GB for 7B model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 8-bit Adam (14 GB for 7B model = 4× reduction)
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-4)
```

**Memory savings:**
- 7B model: 56 GB → 14 GB (75% reduction)
- No performance degradation in practice
- Slightly slower optimizer step (10-15%)

### 4.4 Why 8-bit Works for Optimizer States

Optimizer states are **exponential moving averages** that:
- Accumulate slowly over many steps
- Don't require high precision at each step
- Tolerate quantization noise

**Empirical result:** 8-bit Adam matches FP32 Adam's convergence on models up to 175B parameters.

---

## 5. State Offloading: Paged Adam

### 5.1 Motivation

Even with 8-bit quantization, optimizer states for 100B+ models exceed GPU memory. **Paged Adam** moves optimizer states to CPU RAM and streams them to GPU as needed.

### 5.2 How Paged Adam Works

**Inspired by virtual memory paging in operating systems:**

1. **Optimizer states live in CPU memory** (host RAM)
2. **Small "pages" transferred to GPU** during optimizer step
3. **Adam update applied** for that page's parameters
4. **Updated states copied back to CPU**
5. **Next page processed**

Only a small fraction of optimizer states is on GPU at any time.

```
CPU Memory (512 GB)              GPU Memory (40 GB)
┌─────────────────────┐          ┌──────────────┐
│ m, v for all params │  ──→     │ Active page  │
│ (56 GB for 7B)      │  ←──     │ (~2 GB)      │
└─────────────────────┘          └──────────────┘
        Persistent                   Transient
```

### 5.3 Step-by-Step Operation

```python
# Pseudocode for Paged Adam
for batch in dataloader:
    # 1. Forward/backward on GPU (gradients computed)
    loss = model(batch)
    loss.backward()
    
    # 2. Optimizer step with paging
    for page in optimizer_state_pages:
        # Transfer page to GPU
        m_page, v_page = load_page_from_cpu(page)
        
        # Apply Adam update
        params_page = get_params_for_page(page)
        apply_adam_update(params_page, m_page, v_page, grads_page)
        
        # Write back to CPU
        save_page_to_cpu(page, m_page, v_page)
```

### 5.4 Memory Benefits

**GPU memory becomes nearly constant** with respect to model size:

| Model Size | Standard Adam (GPU) | Paged Adam (GPU) | Paged Adam (CPU) |
|------------|---------------------|------------------|------------------|
| 1B params | 8 GB | ~1 GB | 8 GB |
| 7B params | 56 GB | ~1 GB | 56 GB |
| 13B params | 104 GB | ~1 GB | 104 GB |
| 70B params | 560 GB | ~1 GB | 560 GB |

**Enables single-GPU fine-tuning** of models that wouldn't fit otherwise.

### 5.5 Performance Tradeoffs

**Bottleneck:** PCIe bandwidth (typically 16-32 GB/s)

**Impact on throughput:**
- Standard Adam: ~1ms optimizer step
- Paged Adam: ~50-100ms optimizer step (50-100× slower)

**When overhead is acceptable:**
- Fine-tuning (fewer total steps)
- Small batch sizes (more time in forward/backward)
- Parameter-efficient methods (LoRA reduces parameter count)

**Not suitable for:**
- Large-scale pretraining
- High-throughput multi-GPU training
- Latency-critical applications

### 5.6 Implementation

```python
import bitsandbytes as bnb

# Paged Adam (optimizer states in CPU memory)
optimizer = bnb.optim.PagedAdam(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
)

# Can combine with 8-bit quantization
optimizer = bnb.optim.PagedAdam8bit(model.parameters(), lr=1e-4)
```

### 5.7 Paged Adam vs ZeRO-Offload

| Aspect | Paged Adam | ZeRO-Offload |
|--------|------------|--------------|
| **Scope** | Optimizer states only | Parameters + gradients + optimizer |
| **Granularity** | Page-level | Tensor-level |
| **Use case** | Single GPU fine-tuning | Multi-GPU distributed training |
| **Implementation** | bitsandbytes | DeepSpeed |
| **Complexity** | Simple drop-in | Requires distributed setup |
| **Throughput** | ~50× slower optimizer | Better pipelining |

**Decision guide:**
- Single GPU + memory constraints → **Paged Adam**
- Multi-GPU + need full model training → **ZeRO-Offload**

---

## 6. Algorithmic Changes: Adafactor and LION

Some optimizers redesign the algorithm to use less state.

### 6.1 Adafactor (Google)

**Key idea:** Don't store full second moment matrix; use **factored approximation**.

**Standard Adam second moment:**
```
v ∈ R^(d_model × d_model)  # Full matrix
Memory: O(d²)
```

**Adafactor second moment:**
```
row_var ∈ R^(d_model)      # Row variances
col_var ∈ R^(d_model)      # Column variances
v ≈ row_var ⊗ col_var      # Outer product approximation
Memory: O(d)
```

**Implementation:**
```python
from transformers import Adafactor

optimizer = Adafactor(
    model.parameters(),
    lr=1e-3,
    scale_parameter=True,      # Scale learning rate by parameter norm
    relative_step=False,       # Use fixed learning rate
    warmup_init=False,
)
```

**Pros:**
- ~2× memory reduction vs Adam
- Works well for Transformers (designed for T5)

**Cons:**
- Slightly different convergence behavior
- Less well-tested than Adam
- Sensitive to hyperparameters

### 6.2 LION (EvoLved Sign Momentum)

**Key idea:** Store only the **sign** of momentum, not the full value.

```python
# LION update (simplified)
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
theta_t = theta_{t-1} - lr * sign(m_t)  # Only sign matters!
```

**Memory:** Same as SGD with momentum (1× state vs Adam's 2×)

**Benchmarks:** Matches or beats Adam on many tasks with proper tuning.

```python
# Using LION optimizer (google/automl)
from lion_pytorch import LION

optimizer = LION(model.parameters(), lr=1e-4, betas=(0.9, 0.99))
```

---

## 7. Hybrid Approaches: QLoRA

**QLoRA** stacks 4-bit NF4 base model quantization, Paged Adam optimizer, and LoRA adapters to reduce a 7B fine-tune from ~112 GB to ~4 GB — enabling consumer GPU training.

See [QLoRA](../parameter_efficient_fine_tuning/qlora.md) for the full derivation, memory breakdown, and implementation.

---

## 8. Comparison Table

### 8.1 Memory Efficiency

| Optimizer | Memory per Parameter | Speedup vs Adam | Best Use Case |
|-----------|---------------------|-----------------|---------------|
| Adam (FP32) | 12 bytes | 1× (baseline) | Standard training |
| Adam (BF16 params) | 10 bytes | 1× | Mixed-precision training |
| 8-bit Adam | 6 bytes | 0.9× | Large model training |
| Paged Adam | ~2 bytes (GPU) | 0.01-0.02× | Single-GPU fine-tuning |
| Paged 8-bit Adam | ~0.5 bytes (GPU) | 0.01-0.02× | Extreme memory constraint |
| Adafactor | 6 bytes | 1.2× | T5/encoder-decoder models |
| LION | 6 bytes | 1.1× | Vision models, some LLMs |
| SGD + Momentum | 6 bytes | 1.5× | Not recommended for LLMs |

### 8.2 Decision Matrix

| Scenario | Recommendation | Reason |
|----------|---------------|--------|
| 7B model, 80GB GPU | Standard Adam (BF16) | Memory not a constraint |
| 7B model, 24GB GPU | 8-bit Adam | 4× memory saving, minimal slowdown |
| 7B model, 16GB GPU | Paged 8-bit Adam + LoRA | Extreme memory efficiency |
| 70B model, multi-GPU | ZeRO-3 + 8-bit Adam | Distributed memory optimization |
| Fine-tuning only | QLoRA | Best memory/quality tradeoff |
| T5/Flan models | Adafactor | Designed for these architectures |
| Rapid experimentation | LION | Simpler, fewer hyperparameters |

---

## 11. Quick Reference

### Memory Hierarchy for Optimizers

```
Least Memory ────────────────────────► Most Memory
│                                                  │
QLoRA          Paged     8-bit      Adam         Adam
(4-bit +       8-bit     Adam       (BF16)      (FP32)
LoRA +         Adam                              
Paging)                                           
│                                                  │
~1 GB/7B      ~14 GB    ~28 GB     ~56 GB      ~56 GB
```

### Performance Hierarchy

```
Fastest ────────────────────────────► Slowest
│                                              │
Adam          8-bit       Paged        QLoRA
(FP32/BF16)   Adam        Adam         (complex)
│                                              │
1× baseline   0.9×        0.01-0.02×   0.01-0.02×
```

### Checklist for Choosing an Optimizer

- [ ] What GPU do you have? (memory capacity)
- [ ] Single or multi-GPU? (offload strategies differ)
- [ ] Full fine-tuning or LoRA? (LoRA reduces parameter count)
- [ ] Throughput vs memory priority? (speed/memory tradeoff)
- [ ] Supported hardware? (BF16 needs A100+, FP8 needs H100)

---

## 12. Further Reading

- **8-bit Optimizers:** [bitsandbytes Paper (2021)](https://arxiv.org/abs/2110.02861)
- **Paged Optimizers:** [bitsandbytes Documentation](https://github.com/TimDettmers/bitsandbytes)
- **QLoRA:** [QLoRA Paper (2023)](https://arxiv.org/abs/2305.14314)
- **Adafactor:** [Adafactor Paper (2018)](https://arxiv.org/abs/1804.04235)
- **LION:** [LION Paper (2023)](https://arxiv.org/abs/2302.06675)
- **ZeRO:** [ZeRO Paper (2019)](https://arxiv.org/abs/1910.02054)
- **GaLore:** [GaLore Paper (2024)](https://arxiv.org/abs/2403.03507)
