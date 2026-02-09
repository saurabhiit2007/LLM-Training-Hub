# 🎯 Mixed-Precision Training for LLMs

## 1. Overview

**Mixed-precision training** uses lower-precision numeric formats (FP16, BF16) for computation while maintaining higher precision (FP32) for critical operations. This reduces memory usage and accelerates training without significantly impacting model quality.

### Why Mixed-Precision?

Training large language models faces three key bottlenecks:
- **Memory**: Model parameters, gradients, and optimizer states
- **Compute**: Matrix multiplications during forward/backward passes
- **Communication**: Gradient synchronization in distributed training

Mixed-precision addresses all three by using 16-bit formats strategically.

---

## 2. Core Concepts

### 2.1 Precision Formats Comparison

| Format | Bits | Exponent | Mantissa | Range | Precision | Best For |
|--------|------|----------|----------|-------|-----------|----------|
| FP32 | 32 | 8 | 23 | ±3.4e38 | High | Master weights |
| FP16 | 16 | 5 | 10 | ±65,504 | Medium | Fast compute (with loss scaling) |
| BF16 | 16 | 8 | 7 | ±3.4e38 | Lower | Stable training |
| TF32 | 19* | 8 | 10 | ±3.4e38 | Medium | NVIDIA A100+ internal |

*TF32 uses 32 bits in memory but only 19 bits matter for computation

### 2.2 Memory Savings

For a model with N parameters:

| Component | FP32 | Mixed-Precision | Savings |
|-----------|------|-----------------|---------|
| Parameters | 4N | 2N | 50% |
| Gradients | 4N | 2N | 50% |
| Activations | 4N | 2N | 50% |
| Optimizer states | Varies | Same (FP32) | 0% |

**Overall memory reduction**: ~40-50% depending on optimizer

---

## 3. Mixed-Precision Strategies

### 3.1 Automatic Mixed Precision (AMP)

Modern frameworks automatically choose precision for each operation:

**PyTorch Implementation**
```python
from torch.cuda.amp import autocast, GradScaler

model = MyLLM().cuda()
optimizer = torch.optim.AdamW(model.parameters())
scaler = GradScaler()  # For FP16 only

for inputs, labels in dataloader:
    optimizer.zero_grad()
    
    # Forward pass in mixed precision
    with autocast(dtype=torch.float16):  # or torch.bfloat16
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    
    # Backward pass with gradient scaling (FP16 only)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**HuggingFace Transformers**
```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    bf16=True,  # Use BF16 (recommended for A100+)
    # fp16=True,  # Or use FP16 (for V100, older GPUs)
    fp16_full_eval=False,  # Keep eval in FP32 if needed
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
```

### 3.2 FP16 vs BF16: When to Use Which?

#### **FP16 (Half Precision)**

**Pros:**
- Wider hardware support (V100, T4, older GPUs)
- Slightly higher precision for small values
- Mature tooling

**Cons:**
- Limited range causes gradient underflow/overflow
- Requires loss scaling
- More tuning needed

**Use when:**
- Training on V100 or older NVIDIA GPUs
- Model is small-to-medium (< 1B parameters)

#### **BF16 (Brain Float 16)**

**Pros:**
- Same range as FP32 (no overflow/underflow)
- No loss scaling needed
- Drop-in replacement for FP32
- More stable for large models

**Cons:**
- Requires A100, H100, or newer hardware
- Lower precision for very small values (rarely matters)

**Use when:**
- Training on A100, H100, or TPUs
- Training large models (> 1B parameters)
- You want stability without tuning

**Quick Decision Tree:**
```
Do you have A100/H100? 
  ├─ Yes → Use BF16
  └─ No → Use FP16 with loss scaling
```

---

## 4. Loss Scaling (FP16 Only)

### 4.1 Why Loss Scaling?

FP16's minimum normal value is ~6e-5. Gradients smaller than this underflow to zero:

```python
# Example: gradient underflow in FP16
gradient_fp32 = 1e-7  # Common for deep layers
gradient_fp16 = torch.tensor(gradient_fp32, dtype=torch.float16)
print(gradient_fp16)  # Output: 0.0 (underflow!)
```

### 4.2 How It Works

1. **Scale up** the loss before backward pass
2. **Gradients scale** proportionally
3. **Scale down** gradients before optimizer step

```python
scale_factor = 1024  # or dynamic

# Forward
loss = criterion(outputs, labels)
scaled_loss = loss * scale_factor

# Backward
scaled_loss.backward()

# Unscale before optimizer
for param in model.parameters():
    if param.grad is not None:
        param.grad /= scale_factor

optimizer.step()
```

### 4.3 Dynamic Loss Scaling

Automatically adjusts scale factor:

```python
scaler = GradScaler(
    init_scale=2**16,      # Initial scale
    growth_factor=2.0,     # Increase by 2x if stable
    backoff_factor=0.5,    # Decrease by 0.5x if overflow
    growth_interval=2000,  # Check every N iterations
)
```

**Algorithm:**
- If no overflow for N steps → increase scale
- If overflow detected → decrease scale, skip update

---

## 5. Best Practices

### 5.1 What Stays in FP32?

Critical operations that should remain in FP32:
- **Master weights** (copy of parameters)
- **Optimizer states** (momentum, variance for Adam)
- **Loss computation** (optional, helps stability)
- **Batch normalization** (variance calculation)
- **Softmax** (numerical stability)

### 5.2 Common Pitfalls

| Issue | Symptom | Solution |
|-------|---------|----------|
| Gradient overflow | Loss becomes NaN | Increase loss scaling / Use BF16 |
| Gradient underflow | Loss plateaus early | Decrease loss scaling / Use BF16 |
| Poor convergence | Higher final loss | Check optimizer precision, use BF16 |
| OOM errors | Out of memory | Verify mixed precision is active |

### 5.3 Verification

```python
# Check if mixed precision is working
def check_precision(model):
    for name, param in model.named_parameters():
        print(f"{name}: {param.dtype}")
        if param.grad is not None:
            print(f"{name}.grad: {param.grad.dtype}")
        break  # Check first layer

# Monitor during training
with autocast():
    # Should see FP16/BF16
    print(f"Activations: {outputs.dtype}")
```

---

## 6. Advanced Topics

### 6.1 TF32 (NVIDIA A100+)

TF32 is automatically used for matrix multiplications on A100/H100:
- Stores as FP32
- Computes with 10-bit mantissa
- ~8x faster than FP32
- Transparent to user

```python
# Enable TF32 (enabled by default on A100+)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### 6.2 FP8 (H100, Cutting Edge)

Next-generation format for H100:
- 8-bit floating point
- Requires special scaling strategies
- 2-4x faster than FP16/BF16

```python
# Transformer Engine (NVIDIA)
import transformer_engine.pytorch as te

layer = te.Linear(768, 3072, params_dtype=torch.float8_e4m3)
```

### 6.3 Mixed Precision with FSDP

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision

mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,    # Parameters
    reduce_dtype=torch.float32,     # Gradient reduction
    buffer_dtype=torch.float32,     # Buffers (BatchNorm, etc.)
)

model = FSDP(model, mixed_precision=mp_policy)
```

---

## 7. Interview Questions

### Q1: Why does BF16 not need loss scaling while FP16 does?

**Answer:** BF16 has an 8-bit exponent (same as FP32), giving it the same numeric range (±3.4e38). This means gradients rarely underflow to zero. FP16 has only a 5-bit exponent with range ±65,504, so small gradients (< 6e-5) underflow without scaling.

**Follow-up:** What about precision loss?  
BF16 has 7 mantissa bits vs FP16's 10, but training is more sensitive to range than precision. Small rounding errors are tolerable; underflow is not.

---

### Q2: Explain how automatic mixed precision (AMP) works internally.

**Answer:** AMP maintains two copies of model weights:
1. **FP16/BF16 copy** for forward/backward (faster compute)
2. **FP32 master copy** for optimizer updates (numerical stability)

During training:
- Forward pass: FP16/BF16
- Loss computation: Often FP32
- Backward pass: FP16/BF16 gradients
- Optimizer step: FP32 master weights → convert to FP16/BF16

**Follow-up:** Why keep FP32 master weights?  
Repeated small updates in FP16 can round to zero. FP32 accumulates these tiny changes accurately.

---

### Q3: You're training a 7B model on A100s and loss becomes NaN. What do you check?

**Answer (structured approach):**

1. **Check precision settings:**
   ```python
   # If using FP16
   bf16=False, fp16=True  # Switch to BF16 on A100
   ```

2. **Inspect gradient norms:**
   ```python
   total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   if total_norm > 1000 or torch.isnan(total_norm):
       print("Gradient explosion!")
   ```

3. **Check learning rate:** Too high can cause instability
4. **Verify loss scaling:** If using FP16, check scaler state
5. **Data sanity:** NaN in inputs will propagate

**Most likely solution for A100:** Switch from FP16 to BF16.

---

### Q4: What's the memory breakdown for mixed-precision training of a 1B parameter model?

**Answer:**

| Component | FP32 (GB) | Mixed (BF16) | Calculation |
|-----------|-----------|--------------|-------------|
| Parameters | 4.0 | 2.0 | 1B × 4 bytes or 2 bytes |
| Gradients | 4.0 | 2.0 | Same as parameters |
| Optimizer (AdamW) | 8.0 | 8.0 | 2 states × 4 bytes (stays FP32) |
| Activations* | ~12.0 | ~6.0 | Batch-dependent |
| **Total** | **~28 GB** | **~18 GB** | **36% reduction** |

*Assumes batch_size=4, seq_len=2048

**Key insight:** Optimizer states don't benefit from mixed precision, limiting overall savings.

---

### Q5: How would you implement mixed precision for a custom CUDA kernel?

**Answer:**

```python
import torch
from torch.cuda.amp import custom_fwd, custom_bwd

class MyFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd  # Cast inputs to autocast dtype
    def forward(ctx, input, weight):
        # Runs in FP16/BF16 if autocast is enabled
        output = my_cuda_kernel(input, weight)
        ctx.save_for_backward(input, weight)
        return output
    
    @staticmethod
    @custom_bwd  # Match forward precision
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = my_cuda_kernel_backward(grad_output, weight)
        grad_weight = my_cuda_kernel_backward(grad_output, input)
        return grad_input, grad_weight
```

**Key points:**
- `@custom_fwd` auto-casts inputs
- `@custom_bwd` ensures gradient dtype matches
- Kernel itself must support FP16/BF16 operations

---

### Q6: Why might mixed precision actually hurt performance in some cases?

**Answer (common scenarios):**

1. **Small models on modern GPUs:**
   - Overhead of type casting > compute savings
   - Memory not a bottleneck

2. **CPU training:**
   - No tensor cores
   - FP16/BF16 ops may be slower than FP32

3. **Batch size = 1:**
   - Can't amortize precision conversion costs
   - Common in inference, not training

4. **Memory-bound operations:**
   - Attention with large sequence lengths
   - Speedup comes from compute, not memory

**Benchmark example:**
```python
# Small model might be slower in mixed precision
model = SmallTransformer(d_model=256, n_layers=6)  # 10M params

# Always profile before deciding
with torch.profiler.profile() as prof:
    with autocast():
        outputs = model(inputs)
print(prof.key_averages().table())
```

---

### Q7: What's the difference between `torch.autocast` and model.half()?

**Answer:**

| Aspect | `model.half()` | `torch.autocast()` |
|--------|----------------|-------------------|
| **Scope** | Converts entire model permanently | Context manager for specific operations |
| **Control** | All ops in FP16 | Selective (matmul in FP16, softmax in FP32) |
| **Safety** | Can cause instability | Automatically handles sensitive ops |
| **Master weights** | No | Yes (via optimizer) |
| **Recommended** | Inference only | Training |

**Example of the problem with `.half()`:**
```python
model = model.half()  # Everything in FP16

# Numerical instability in softmax
attention_scores = torch.randn(1000, 1000, dtype=torch.float16)
attention_probs = F.softmax(attention_scores, dim=-1)  # May overflow!

# vs autocast (safe)
with autocast():
    attention_probs = F.softmax(attention_scores, dim=-1)  # Auto-promoted to FP32
```

---

## 8. Recent Updates (2024-2025)

### 8.1 FP8 Support (NVIDIA H100)

**Transformer Engine** (released 2023, stable in 2024):
```python
import transformer_engine.pytorch as te

# Automatic FP8 precision
layer = te.Linear(in_features=1024, out_features=4096)
layer = te.LayerNormLinear(1024, 4096)  # Fused ops

# 30-40% faster than BF16 on H100
```

**Key differences from FP16/BF16:**
- Two FP8 formats: E4M3 (training) and E5M2 (gradients)
- Requires per-tensor scaling (like FP16 loss scaling)
- Software support still maturing

### 8.2 PyTorch 2.0+ Compile with Mixed Precision

```python
model = torch.compile(model, mode="max-autotune")

# Automatically finds best precision for each op
with autocast(dtype=torch.bfloat16):
    outputs = model(inputs)
```

**Benefits:**
- Fuses precision casts with kernels
- 10-20% additional speedup over standard AMP

### 8.3 Microsoft DeepSpeed ZeRO-Offload with Mixed Precision

```python
ds_config = {
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},  # Mixed precision on GPU, optimizer on CPU
    }
}
```

Enables training 10B+ models on consumer GPUs.

---

## 9. Quick Reference

### Decision Matrix

| Scenario | Recommendation | Reason |
|----------|---------------|--------|
| A100/H100 training | BF16 | No loss scaling, stable |
| V100/T4 training | FP16 + loss scaling | No BF16 support |
| Small model (<100M) | FP32 | Minimal benefit |
| Inference | FP16 + `model.half()` | Simpler, no training concerns |
| H100 cutting edge | FP8 | Maximum speed |
| Stability issues | BF16 or FP32 | More numerical headroom |

### Common Errors and Fixes

```python
# Error: "Mixed precision requires CUDA"
# Fix: Check device
assert torch.cuda.is_available()

# Error: "overflow" in FP16
# Fix: Switch to BF16 or adjust loss scaling
scaler = GradScaler(init_scale=2**10)  # Lower initial scale

# Error: "NaN loss"
# Fix: Gradient clipping + precision check
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Error: "No speedup observed"
# Fix: Ensure tensor cores are used
assert inputs.shape[-1] % 8 == 0  # Dimensions must be multiples of 8
```

---

## 10. Further Reading

- [NVIDIA Mixed Precision Training Guide](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/)
- [PyTorch AMP Documentation](https://pytorch.org/docs/stable/amp.html)
- [BF16 Original Paper (2019)](https://arxiv.org/abs/1905.12322)
- [FP8 Formats for Deep Learning (2022)](https://arxiv.org/abs/2209.05433)
- [Hugging Face Performance Docs](https://huggingface.co/docs/transformers/perf_train_gpu_one)
