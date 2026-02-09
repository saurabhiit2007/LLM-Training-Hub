# ⚡ Gradient Checkpointing for LLM Training

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

## 8. Interview Questions

### Q1: Explain the memory vs compute trade-off in gradient checkpointing.

**Answer:**

**Without checkpointing:**
- Store all $L$ layer activations during forward pass
- Backward pass uses stored activations (1× forward compute)
- Memory: $O(L \times B \times S \times d)$

**With checkpointing (every $K$ layers):**
- Store activations only at checkpoints ($L/K$ checkpoints)
- Backward pass recomputes activations between checkpoints
- Memory: $O((L/K) \times B \times S \times d)$
- Compute: 1× forward + ($K/L$)× recomputation = $1 + K/L$ forward passes

**Example:** 32-layer model, checkpoint every 4 layers
- Memory: $32/4 = 8$ checkpoints → 75% memory reduction
- Compute: $1 + 4/32 = 1.125$× → 12.5% slower

**Follow-up:** Why not checkpoint every single layer?  
**Answer:** Checkpointing overhead (memory allocation, bookkeeping) becomes significant. Empirically, checkpointing every 1-8 layers is optimal.

---

### Q2: Why does activation memory scale quadratically with sequence length?

**Answer:**

**Attention mechanism computes:**
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**Memory bottleneck:** Attention scores $QK^T$

**Shape:**
- $Q$: $(B, S, d)$ — batch, sequence, dimension
- $K^T$: $(B, d, S)$
- $QK^T$: $(B, S, S)$ — **square matrix for each sequence**

**Per-layer attention memory:**
$$
M_{\text{attn}} = B \times H \times S^2 \times 2 \text{ bytes (BF16)}
$$

Where $H$ = number of attention heads.

**Scaling example (32 heads, batch=4):**

| Seq Length | $S^2$ | Memory (BF16) |
|------------|-------|---------------|
| 512 | 262K | 64 MB |
| 2048 | 4.2M | 1 GB |
| 8192 | 67M | 16 GB |
| 32768 | 1.07B | **256 GB** per layer |

**With 32 layers at seq_len=32768: ~8TB of activation memory without checkpointing!**

**Follow-up:** How does Flash Attention help?  
**Answer:** Flash Attention recomputes attention on-the-fly in blocks, never materializing the full $S \times S$ matrix. This is orthogonal to gradient checkpointing and can be combined for maximum savings.

---

### Q3: You're training LLaMA-13B on a single A100 (80GB). Design your checkpointing strategy.

**Answer:**

**Model specs:**
- 13B parameters (40 layers)
- Target: batch_size=4, seq_len=4096

**Memory budget (without checkpointing):**

| Component | Memory |
|-----------|--------|
| Parameters (BF16) | 26 GB |
| Gradients | 26 GB |
| Optimizer (8-bit Adam) | 13 GB |
| Activations (40 layers) | ~60 GB |
| **Total** | **125 GB** ❌ (exceeds 80GB) |

**Strategy:**

1. **Enable gradient checkpointing every layer:**
   ```python
   model.gradient_checkpointing_enable()
   ```
   Activation memory: 60 GB → ~8 GB (87% reduction)

2. **Use Flash Attention 2:**
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       "meta-llama/Llama-2-13b-hf",
       attn_implementation="flash_attention_2",
   )
   ```
   Additional ~30% reduction in attention memory

3. **Reduce batch size if needed:**
   ```python
   # If still OOM, use batch_size=2 with gradient accumulation
   batch_size = 2
   gradient_accumulation_steps = 2  # Effective batch = 4
   ```

**Final memory:**

| Component | Memory |
|-----------|--------|
| Parameters | 26 GB |
| Gradients | 26 GB |
| Optimizer | 13 GB |
| Activations (checkpointed + Flash) | ~5 GB |
| **Total** | **70 GB** ✅ |

**Trade-offs:**
- Throughput: ~25-30% slower than no checkpointing
- Quality: No impact (same training dynamics)

---

### Q4: Compare gradient checkpointing vs reducing batch size for memory savings.

**Answer:**

| Approach | Memory Saved | Throughput Impact | Convergence Impact |
|----------|--------------|-------------------|-------------------|
| Gradient checkpointing | 50-80% (activations) | -25% to -33% | None |
| Reduce batch size 4→2 | ~40% (activations + grads) | -50% | **May slow convergence** |
| Reduce batch size 4→1 | ~60% (activations + grads) | -75% | **Often slower convergence** |

**Detailed analysis:**

**Gradient checkpointing:**
- ✅ Saves memory without changing training dynamics
- ✅ Allows larger effective batch size via gradient accumulation
- ⚠️ Compute overhead proportional to checkpoint frequency

**Reducing batch size:**
- ✅ Saves memory from activations + gradients
- ❌ Smaller batch = noisier gradients
- ❌ May require more steps to converge
- ❌ Lower GPU utilization (less parallelism)

**Recommendation:**
1. **First**: Enable gradient checkpointing (free quality-wise)
2. **Then**: Reduce batch size if still OOM
3. **Use gradient accumulation** to recover effective batch size:
   ```python
   batch_size = 1  # Fits in memory
   accumulation_steps = 8  # Effective batch = 8
   ```

**Follow-up:** When is reducing batch size better?  
**Answer:** Very small models where checkpointing overhead > benefit, or when you want faster iteration during debugging (batch_size=1 is simplest).

---

### Q5: Explain how `use_reentrant=False` differs from the old checkpointing API.

**Answer:**

### **Old API (`use_reentrant=True`, default before PyTorch 2.0):**

**Mechanism:** Uses Python context manager trick to "re-enter" autograd

**Problems:**
1. **RNG state inconsistency:**
   ```python
   # Forward pass
   x = F.dropout(x, p=0.1)  # RNG state A
   
   # Recomputation during backward
   x = F.dropout(x, p=0.1)  # RNG state B (different!)
   ```
   Different dropout masks → incorrect gradients

2. **Control flow issues:**
   ```python
   if condition:  # Evaluated during forward
       x = layer1(x)
   else:
       x = layer2(x)
   # Recomputation may take different branch!
   ```

3. **Nested checkpointing breaks:**
   ```python
   checkpoint(checkpoint(fn, x))  # ❌ Fails with reentrant
   ```

### **New API (`use_reentrant=False`, recommended):**

**Mechanism:** Uses proper autograd hook system

**Improvements:**
1. **RNG state is saved and restored:**
   ```python
   # PyTorch internally does:
   rng_state = torch.get_rng_state()
   output = fn(x)  # Forward
   
   # During backward:
   torch.set_rng_state(rng_state)
   output = fn(x)  # Exact same RNG → same dropout
   ```

2. **Control flow is saved:**
   - Decision trees stored in autograd graph
   - Recomputation follows exact same path

3. **Memory efficient:**
   - No Python frame overhead
   - Better for deeply nested checkpoints

**Migration:**
```python
# Old (deprecated)
from torch.utils.checkpoint import checkpoint
out = checkpoint(fn, x)  # use_reentrant=True by default

# New (recommended)
out = checkpoint(fn, x, use_reentrant=False)
```

**PyTorch 2.0+ warns if using old API.**

---

### Q6: How would you implement gradient checkpointing from scratch?

**Answer:**

```python
import torch

class GradientCheckpoint(torch.autograd.Function):
    """
    Simplified gradient checkpointing implementation.
    """
    @staticmethod
    def forward(ctx, run_function, *args):
        """
        Forward pass: just run the function and save inputs.
        
        Args:
            run_function: The function to checkpoint
            args: Inputs to the function
        """
        # Save the function and inputs for backward
        ctx.run_function = run_function
        ctx.save_for_backward(*args)
        
        # Run forward pass without saving intermediate activations
        with torch.no_grad():
            outputs = run_function(*args)
        
        return outputs
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        Backward pass: recompute forward, then compute gradients.
        """
        # Retrieve saved inputs
        inputs = ctx.saved_tensors
        
        # Recompute forward pass WITH gradient tracking
        with torch.enable_grad():
            # Detach inputs to avoid double backprop
            detached_inputs = [x.detach().requires_grad_(True) for x in inputs]
            
            # Recompute forward
            outputs = ctx.run_function(*detached_inputs)
        
        # Now compute gradients
        torch.autograd.backward(outputs, grad_outputs)
        
        # Get gradients w.r.t inputs
        grads = [x.grad for x in detached_inputs]
        
        return (None,) + tuple(grads)  # None for run_function

def checkpoint(function, *args):
    """
    Wrapper for gradient checkpointing.
    """
    return GradientCheckpoint.apply(function, *args)

# Example usage
class ExpensiveLayer(torch.nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim, dim * 4)
        self.linear2 = torch.nn.Linear(dim * 4, dim)
    
    def forward(self, x):
        return self.linear2(torch.nn.functional.gelu(self.linear1(x)))

# Test
layer = ExpensiveLayer()
x = torch.randn(4, 512, 1024, requires_grad=True)

# Without checkpointing
y1 = layer(x)
y1.sum().backward()

# With checkpointing
x.grad = None
y2 = checkpoint(layer, x)
y2.sum().backward()

print("Outputs match:", torch.allclose(y1, y2))
print("Gradients match:", torch.allclose(x.grad, x.grad))
```

**Key concepts:**
1. **`forward()` saves only inputs, not intermediate activations**
2. **`backward()` recomputes forward with gradients enabled**
3. **Detaching inputs prevents double backprop through checkpointed region**

**Production differences (PyTorch's implementation):**
- Handles RNG state properly
- Optimized memory management
- Supports nested checkpointing
- Better error messages

---

### Q7: What's the memory footprint during the recomputation phase?

**Answer:**

**Key insight:** During recomputation, we temporarily need memory for **both** the segment being recomputed and the checkpoint boundaries.

**Memory timeline for checkpointing every 4 layers (32 total):**

```
Forward pass:
├─ Layers 1-4:  Store checkpoint at layer 4   [~2 GB]
├─ Layers 5-8:  Store checkpoint at layer 8   [~2 GB]
├─ Layers 9-12: Store checkpoint at layer 12  [~2 GB]
└─ ...

Backward pass (layer 29-32):
├─ Load checkpoint at layer 28               [2 GB]
├─ Recompute layers 29-32                    [2 GB temp]
├─ Compute gradients                         [2 GB grads]
├─ Free recomputed activations
└─ Peak memory: 2 + 2 + 2 = 6 GB

Backward pass (layer 25-28):
├─ Load checkpoint at layer 24               [2 GB]
├─ Recompute layers 25-28                    [2 GB temp]
├─ Compute gradients                         [2 GB grads]
└─ Peak memory: 2 + 2 + 2 = 6 GB
```

**Memory during recomputation:**
$$
M_{\text{peak}} = M_{\text{checkpoint}} + M_{\text{recompute}} + M_{\text{gradients}}
$$

For $K$ layers between checkpoints:
$$
M_{\text{peak}} \approx 2K \times M_{\text{per-layer}}
$$

**Important:** Checkpointing **does not reduce peak memory to zero**. There's still a temporary spike during recomputation.

**Practical implication:**
- If OOM during backward pass, reduce checkpoint interval
- Monitor peak memory, not just average

---

## 9. Recent Updates (2024-2025)

### 9.1 Flash Attention 3 + Gradient Checkpointing

**Flash Attention 3** (H100) combines:
- Flash Attention 2 (no $S^2$ materialization)
- Hardware-aware tiling
- **Built-in gradient checkpointing**

```python
from flash_attn import flash_attn_func

# Automatic gradient checkpointing in Flash Attention
output = flash_attn_func(
    q, k, v,
    causal=True,
    deterministic=True,  # Uses checkpointing for backward
)
```

**Benefit:** No manual checkpointing needed for attention layers.

### 9.2 PyTorch 2.0 `torch.compile()` + Checkpointing

```python
model = torch.compile(model, mode="reduce-overhead")
model.gradient_checkpointing_enable()

# torch.compile optimizes checkpoint recomputation
# Fuses operations, reducing overhead by ~20%
```

### 9.3 Megatron-LM Sequence Parallelism

**Combines sequence parallelism with selective checkpointing:**

```python
# Only checkpoint activations not sharded across GPUs
"sequence_parallel": True,
"checkpoint_activations": True,
"checkpoint_num_layers": 1,  # Checkpoint every layer
```

**Benefit:** Activations sharded + checkpointed = extreme memory efficiency.

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
