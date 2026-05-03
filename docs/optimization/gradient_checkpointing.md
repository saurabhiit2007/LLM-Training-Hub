# Gradient Checkpointing for LLM Training

## 1. Overview

**Gradient checkpointing** (also called **activation checkpointing** or **activation recomputation**) is a memory optimization technique that trades **compute for memory** by recomputing intermediate activations during the backward pass instead of storing them.

### The Memory Problem

During backpropagation, PyTorch stores all intermediate activations from the forward pass to compute gradients. For large models, **activations consume more memory than parameters**.

**Memory breakdown for 7B LLaMA (batch_size=8, seq_len=2048):**

| Component | Memory | Percentage |
|-----------|--------|------------|
| Parameters (BF16) | 14 GB | 12% |
| Gradients | 14 GB | 12% |
| Optimizer states | 56 GB | 47% |
| **Activations** | **~35 GB** | **29%** |
| **Total** | **~119 GB** | **100%** |

Without gradient checkpointing, even a single forward/backward pass can exceed GPU memory.

---

## 2. How Backpropagation Uses Activations

### 2.1 Standard Backpropagation

```python
# Forward pass
x1 = layer1(x0)      # Store x1 for backward
x2 = layer2(x1)      # Store x2 for backward
x3 = layer3(x2)      # Store x3 for backward
loss = criterion(x3, y)

# Backward pass
grad_x3 = grad_loss
grad_x2 = layer3.backward(grad_x3, x2)  # Uses stored x2
grad_x1 = layer2.backward(grad_x2, x1)  # Uses stored x1
grad_x0 = layer1.backward(grad_x1, x0)  # Uses stored x0
```

**Memory:** All activations $x_1, x_2, x_3, ...$ must be kept in memory.

For a Transformer with **L layers**, each layer stores:

- Query, Key, Value projections
- Attention scores
- Attention outputs
- MLP intermediate activations

**Total activation memory scales as:** $O(L \times B \times S \times d)$

- $L$ = number of layers
- $B$ = batch size
- $S$ = sequence length
- $d$ = hidden dimension

### 2.2 Why Activations Dominate

**Example: LLaMA-7B single layer (batch=4, seq_len=2048)**

| Component | Shape | Memory (BF16) |
|-----------|-------|---------------|
| Parameters | (4096, 4096) | 32 MB |
| Attention activations | (4, 2048, 4096) | 64 MB |
| MLP activations | (4, 2048, 11008) | 176 MB |
| **Per-layer total** | - | **240 MB** |
| **32 layers** | - | **7.7 GB** |

With batch_size=8, this doubles to ~15 GB just for activations.

---

## 3. Gradient Checkpointing: Core Idea

### 3.1 The Trade-off

**Instead of storing all activations:**

1. Store activations only at **checkpoint boundaries** (e.g., every K layers)
2. During backward pass, **recompute** activations between checkpoints

```python
# Forward pass (checkpoint every 2 layers)
x1 = layer1(x0)
x2 = layer2(x1)      # Checkpoint: Save x2
del x1               # Free memory

x3 = layer3(x2)
x4 = layer4(x3)      # Checkpoint: Save x4
del x3               # Free memory

# Backward pass for layers 3-4
x3 = layer3(x2)      # RECOMPUTE x3
grad_x4 = grad_loss
grad_x3 = layer4.backward(grad_x4, x3)
grad_x2 = layer3.backward(grad_x3, x2)

# Backward pass for layers 1-2
x1 = layer1(x0)      # RECOMPUTE x1
grad_x2 = ...
grad_x1 = layer2.backward(grad_x2, x1)
grad_x0 = layer1.backward(grad_x1, x0)
```

**Result:**

- Memory: $O(L/K)$ instead of $O(L)$
- Compute: +33% (one extra forward pass per checkpoint segment)

---

## 4. Implementation Strategies

### 4.1 PyTorch Native Checkpointing

```python
import torch
from torch.utils.checkpoint import checkpoint

class TransformerLayer(nn.Module):
    def forward(self, x):
        # Normal forward pass
        attn_out = self.attention(x)
        mlp_out = self.mlp(attn_out)
        return mlp_out

class TransformerWithCheckpointing(nn.Module):
    def __init__(self, num_layers=32):
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer() for _ in range(num_layers)])
    
    def forward(self, x):
        for layer in self.layers:
            # Checkpoint each layer
            x = checkpoint(layer, x, use_reentrant=False)
        return x
```

**Key parameter: `use_reentrant=False`**

- Newer, more memory-efficient implementation
- Avoids issues with control flow and RNGs
- Recommended for all new code

### 4.2 Selective Checkpointing

**Not all layers benefit equally. Checkpoint strategically:**

```python
class SelectiveCheckpointTransformer(nn.Module):
    def __init__(self, num_layers=32, checkpoint_every=4):
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer() for _ in range(num_layers)])
        self.checkpoint_every = checkpoint_every
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i % self.checkpoint_every == 0:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)  # No checkpointing
        return x
```

**Trade-off tuning:**

- `checkpoint_every=1`: Max memory savings (~50%), +33% compute
- `checkpoint_every=4`: Moderate savings (~30%), +10% compute
- `checkpoint_every=∞`: No savings, baseline speed

### 4.3 HuggingFace Transformers

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Train normally
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
for batch in dataloader:
    outputs = model(**batch)
    outputs.loss.backward()
    optimizer.step()
```

**Under the hood:** Checkpoints every Transformer block by default.

### 4.4 DeepSpeed Integration

```python
ds_config = {
    "train_batch_size": 16,
    "gradient_accumulation_steps": 4,
    "activation_checkpointing": {
        "partition_activations": True,        # Shard across GPUs
        "cpu_checkpointing": True,            # Offload to CPU
        "contiguous_memory_optimization": True,
        "number_checkpoints": 32,             # Checkpoint every layer
    }
}

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config,
)
```

**Advanced features:**

- **Activation partitioning**: Shard checkpoints across GPUs
- **CPU offloading**: Store checkpoints in CPU memory
- **Contiguous memory**: Reduce fragmentation

---

## 5. Memory Savings Analysis

### 5.1 Theoretical Bounds

For a model with $L$ layers and checkpointing every $K$ layers:

**Activation memory:**
$$
M_{\text{checkpoint}} = O\left(\frac{L}{K} + K\right) \times B \times S \times d
$$

**Optimal checkpointing interval:**
$$
K_{\text{opt}} = \sqrt{L}
$$

This minimizes total memory while keeping recomputation reasonable.

**Example: 32-layer model**

- No checkpointing: $M \propto 32$
- Checkpoint every 6 layers: $M \propto 32/6 + 6 \approx 11.3$ (65% reduction)
- Optimal ($\sqrt{32} \approx 6$): Best memory/compute trade-off

### 5.2 Real-World Measurements

**LLaMA-7B (32 layers, batch=4, seq_len=2048) on A100:**

| Strategy | Activation Memory | Peak Memory | Throughput | Compute Overhead |
|----------|-------------------|-------------|------------|------------------|
| No checkpointing | 15.2 GB | 85 GB | 1.0× | 0% |
| Checkpoint every 8 layers | 5.8 GB | 75 GB | 0.92× | +8% |
| Checkpoint every 4 layers | 4.1 GB | 70 GB | 0.85× | +15% |
| Checkpoint every layer | 1.9 GB | 62 GB | 0.75× | +33% |

**Sweet spot:** Checkpoint every 4-8 layers for most use cases.

### 5.3 Scaling with Sequence Length

Activation memory scales **quadratically** with sequence length due to attention:

$$
M_{\text{attn}} \propto B \times S^2 \times H
$$

Where $H$ = number of attention heads.

**Impact:**

| Sequence Length | Activation Memory (32 layers) | With Checkpointing (every layer) |
|-----------------|-------------------------------|----------------------------------|
| 512 | 1.2 GB | 0.3 GB |
| 2048 | 15.2 GB | 1.9 GB |
| 8192 | 240 GB | 30 GB |
| 32768 | **3.8 TB** | **480 GB** |

**For long-context training, gradient checkpointing is essential.**

---

## 6. Best Practices

### 6.1 When to Use Gradient Checkpointing

✅ **Use when:**

- Training large models (>1B parameters)
- Long sequence lengths (>2048)
- Limited GPU memory
- Batch size is already at minimum (can't reduce further)

❌ **Don't use when:**

- Small models (<100M parameters)
- GPU memory is not a bottleneck
- Inference (no backward pass needed)
- Extremely tight latency requirements

### 6.2 Combining with Other Techniques

**Recommended stack for memory efficiency:**

```python
# 1. Mixed precision
from torch.cuda.amp import autocast, GradScaler

# 2. Gradient checkpointing
model.gradient_checkpointing_enable()

# 3. Gradient accumulation
accumulation_steps = 4

# 4. Memory-efficient optimizer
import bitsandbytes as bnb
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-4)

scaler = GradScaler()
for i, batch in enumerate(dataloader):
    with autocast(dtype=torch.bfloat16):
        outputs = model(**batch)
        loss = outputs.loss / accumulation_steps
    
    scaler.scale(loss).backward()
    
    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

**Memory savings stack:**

- Mixed precision: -40%
- Gradient checkpointing: -50% (activations)
- 8-bit optimizer: -75% (optimizer states)
- **Combined: Enables 3-4× larger models**

### 6.3 Debugging Checkpointing Issues

**Common problems:**

1. **RNG state mismatch**
   ```python
   # Problem: Dropout behaves differently during recomputation
   # Solution: Use use_reentrant=False
   checkpoint(fn, x, use_reentrant=False)
   ```

2. **In-place operations**
   ```python
   # Problem: In-place ops break gradient computation
   x += residual  # ❌ Breaks checkpointing
   
   # Solution: Use out-of-place operations
   x = x + residual  # ✅ Works with checkpointing
   ```

3. **Custom CUDA kernels**
   ```python
   # Must define backward pass explicitly
   class MyCheckpointedFunction(torch.autograd.Function):
       @staticmethod
       def forward(ctx, x):
           ctx.save_for_backward(x)
           return my_cuda_kernel_forward(x)
       
       @staticmethod
       def backward(ctx, grad_output):
           x, = ctx.saved_tensors
           return my_cuda_kernel_backward(grad_output, x)
   ```

---

## 7. Advanced Topics

### 7.1 CPU Offloading (DeepSpeed)

For extreme memory constraints, offload checkpoints to CPU:

```python
# DeepSpeed config
"activation_checkpointing": {
    "cpu_checkpointing": True,  # Store checkpoints in CPU RAM
    "contiguous_memory_optimization": True,
}
```

**Trade-off:**

- Memory: Can handle arbitrarily large models
- Speed: ~2-3× slower due to PCIe transfers

**Use case:** Training 70B+ models on consumer GPUs.

### 7.2 Selective Activation Checkpointing (SAC)

**Idea:** Only checkpoint memory-heavy operations (attention), not cheap ones (LayerNorm).

```python
class SmartCheckpointLayer(nn.Module):
    def forward(self, x):
        # Checkpoint expensive attention
        attn_out = checkpoint(self.attention, x, use_reentrant=False)
        
        # Don't checkpoint cheap operations
        norm_out = self.layer_norm(attn_out)
        
        # Checkpoint expensive MLP
        mlp_out = checkpoint(self.mlp, norm_out, use_reentrant=False)
        
        return mlp_out
```

**Benefit:** 80% of memory savings with only 15% compute overhead.

### 7.3 Gradient Checkpointing + FSDP

**Fully Sharded Data Parallel (FSDP)** + checkpointing enables massive models:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# Define which layers to checkpoint
auto_wrap_policy = transformer_auto_wrap_policy(
    transformer_layer_cls={TransformerBlock},
)

model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    use_orig_params=True,  # Enable gradient checkpointing
)

# Enable checkpointing
for module in model.modules():
    if isinstance(module, TransformerBlock):
        module.gradient_checkpointing = True
```

**Enables training 175B+ models on 8× A100s.**

---

## 10. Quick Reference

### Decision Tree

```
Is GPU memory sufficient?
├─ Yes → Don't use checkpointing (faster training)
└─ No  → Can you reduce batch size?
    ├─ No (already batch_size=1) → Enable checkpointing
    └─ Yes → Try batch reduction first
        └─ Still OOM? → Enable checkpointing
```

### Checkpointing Intervals

| Model Size | Recommended Interval | Memory Savings | Compute Overhead |
|------------|---------------------|----------------|------------------|
| <1B | Every 8-16 layers | 30-40% | +5-10% |
| 1-10B | Every 4-8 layers | 50-60% | +10-20% |
| 10-100B | Every 1-2 layers | 70-80% | +25-35% |
| >100B | Every layer + CPU offload | 85-90% | +50-100% |

### Compatibility Matrix

| Technique | Compatible | Notes |
|-----------|-----------|-------|
| Mixed precision | ✅ | No interaction |
| Flash Attention | ✅ | Complementary (use both) |
| FSDP | ✅ | Enables massive models |
| DDP | ✅ | Works normally |
| DeepSpeed ZeRO | ✅ | Can combine with CPU offload |
| Model parallelism | ✅ | Checkpoint per device |
| Quantization | ✅ | Checkpoint quantized activations |

---

## 11. Further Reading

- **Gradient Checkpointing Original Paper:** [Training Deep Nets with Sublinear Memory Cost (2016)](https://arxiv.org/abs/1604.06174)
- **PyTorch Documentation:** [torch.utils.checkpoint](https://pytorch.org/docs/stable/checkpoint.html)
- **Flash Attention:** [Flash Attention 2 (2023)](https://arxiv.org/abs/2307.08691)
- **DeepSpeed Activation Checkpointing:** [DeepSpeed Docs](https://www.deepspeed.ai/tutorials/megatron/)
- **Megatron-LM:** [Efficient Large-Scale Language Model Training (2021)](https://arxiv.org/abs/2104.04473)
