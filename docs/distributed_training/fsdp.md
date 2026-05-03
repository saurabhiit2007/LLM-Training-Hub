# FSDP: Fully Sharded Data Parallel

## 1. Overview

**FSDP** is PyTorch's native implementation of the **ZeRO-3** algorithm.

**Key difference from DeepSpeed**: Built directly into PyTorch, offering:

- Better integration with PyTorch ecosystem
- Compatibility with `torch.compile`
- More Pythonic API
- Native support for advanced features like HSDP

---

## 2. Core Concept

Like ZeRO-3, FSDP shards **all model states** across GPUs:

- Parameters (Ψ/N per GPU)
- Gradients (Ψ/N per GPU)
- Optimizer states (12Ψ/N per GPU)

**Memory**: ~2Ψ/N per GPU (excluding activations)

---

## 3. How FSDP Works

### 1. Sharding Strategy

**Full Sharding** (default, equivalent to ZeRO-3):
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD
)
```

Each GPU stores 1/N of all parameters.

---

### 2. Forward Pass Pattern

```python
# For each FSDP module (e.g., one transformer layer):

# 1. All-Gather parameters
full_params = all_gather(param_shards)  # Reconstruct

# 2. Compute
output = layer(input, full_params)

# 3. Free parameters immediately
del full_params  # Keep only local shard
```

**Key insight**: Parameters are **materialized only when needed**, then freed.

---

### 3. Backward Pass Pattern

```python
# For each FSDP module:

# 1. All-Gather parameters (again)
full_params = all_gather(param_shards)

# 2. Compute gradients
grad_input, grad_params = backward(output_grad, full_params)

# 3. Free parameters
del full_params

# 4. Reduce-Scatter gradients
grad_shard = reduce_scatter(grad_params)  # Each GPU gets its shard
```

---

## 4. Wrapping Strategies

**Critical for performance**: How you wrap your model determines memory efficiency.

### 1. Auto Wrap (Recommended)

Automatically wraps layers based on policy:

```python
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={LlamaDecoderLayer}  # Wrap each transformer block
)

model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy
)
```

**Result**: Each `LlamaDecoderLayer` is independently sharded and unsharded.

---

### 2. Manual Wrap

```python
from torch.distributed.fsdp import FSDP

# Wrap each layer individually
for i, layer in enumerate(model.layers):
    model.layers[i] = FSDP(layer)

# Wrap the whole model
model = FSDP(model)
```

---

### 3. Size-Based Wrap

```python
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

# Wrap any module with >100M parameters
auto_wrap_policy = functools.partial(
    size_based_auto_wrap_policy,
    min_num_params=100_000_000
)

model = FSDP(model, auto_wrap_policy=auto_wrap_policy)
```

---

## 5. Why Wrapping Matters

### Bad Wrapping (No Sub-Module Wrapping)

```python
# Wrap entire model as one unit
model = FSDP(model)  # No auto_wrap_policy
```

**Problem**: All parameters gathered at once

- Peak memory = full model size (defeats purpose of FSDP!)
- No parameter overlap

---

### Good Wrapping (Layer-by-Layer)

```python
# Wrap each transformer layer
model = FSDP(model, auto_wrap_policy=transformer_auto_wrap_policy)
```

**Benefit**: 

- Layer 1 gathers → computes → frees
- Layer 2 gathers → computes → frees
- Peak memory = single layer + activations

---

## 6. FSDP vs DeepSpeed

| Feature | FSDP | DeepSpeed ZeRO-3 |
|---------|------|-----------------|
| **Ecosystem** | PyTorch native | Third-party (Microsoft) |
| **Configuration** | Python API | JSON config file |
| **torch.compile** | ✅ Full support (FSDP2) | ❌ Limited |
| **CPU Offload** | ✅ Basic | ✅ Advanced (+ NVMe) |
| **Hybrid Sharding** | ✅ HSDP built-in | ❌ Manual setup |
| **Ease of Use** | More Pythonic | Requires config tuning |
| **Throughput** | Better for <20B models | Better for 100B+ models |
| **Optimized Kernels** | Standard PyTorch | Custom CUDA kernels |

**When to use FSDP**:

- PyTorch-native workflow
- Using `torch.compile` for speedup
- Models <20B parameters
- Want simple Python API

**When to use DeepSpeed**:

- Extreme scale (100B+ parameters)
- Need NVMe offload
- Want maximum optimization

---

## 7. FSDP2: The Modern Version

**Introduced in PyTorch 2.x**, built on DTensors (Distributed Tensors).

### Key Improvements

1. **No Parameter Flattening**:
   - Original FSDP flattens params into 1D buffer
   - FSDP2 keeps original shapes
   - Better compatibility with `torch.compile`

2. **Faster**:
   - ~10-30% throughput improvement
   - Better kernel fusion with `torch.compile`

3. **Cleaner API**:
```python
from torch.distributed._composable.fsdp import fully_shard

# Apply to each layer
for layer in model.layers:
    fully_shard(layer)

fully_shard(model)  # Wrap whole model
```

---

## 8. Hybrid Sharding (HSDP)

**Problem**: All-to-all communication across 1000s of GPUs is slow.

**Solution**: **Hybrid Sharding** = Shard within nodes, replicate across nodes.

```
Node 0: [GPU 0-7] ← 8-way FSDP (shard parameters)
Node 1: [GPU 8-15] ← 8-way FSDP (same parameters, different data)
...
        ↕ Data Parallel replication
```

**Benefits**:

- Fast NVLink communication within nodes (900 GB/s)
- Avoid slow inter-node all-gather (100 Gb/s Ethernet)
- Scale to thousands of GPUs

### Configuration

```python
from torch.distributed.fsdp import ShardingStrategy

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
    device_mesh=device_mesh  # Define node topology
)
```

**Use case**: Multi-node training where network is the bottleneck.

---

## 9. Memory Calculation Example

**7B model, 8 GPUs, FP16, Adam**

### Standard DP (No FSDP)

- Parameters: 14 GB
- Gradients: 14 GB
- Optimizer: 84 GB
- **Total per GPU: 112 GB** ❌

### FSDP (Full Sharding)

- Parameters: 14/8 = 1.75 GB
- Gradients: 14/8 = 1.75 GB
- Optimizer: 84/8 = 10.5 GB
- **Total per GPU: 14 GB** ✅

**Plus activations**: ~10-20 GB depending on batch size.

---

## 11. Practical Implementation

### Basic FSDP Setup

```python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# Auto-wrap each transformer layer
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={LlamaDecoderLayer}
)

# Create FSDP model
model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=torch.distributed.fsdp.MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    ),
    device_id=torch.cuda.current_device(),
)

# Training loop (same as normal PyTorch)
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch).loss
    loss.backward()
    optimizer.step()
```

---

### With CPU Offload

```python
from torch.distributed.fsdp import CPUOffload

model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    cpu_offload=CPUOffload(offload_params=True),  # Offload params to CPU
)
```

**Trade-off**: Memory ↓↓, Speed ↓

---

### With Activation Checkpointing

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)

# Wrap layers with checkpointing
for layer in model.layers:
    layer = checkpoint_wrapper(
        layer,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT
    )
    layer = FSDP(layer)

model = FSDP(model)
```

**Benefit**: Further reduce activation memory.

---

## 12. Debugging Tips

### Issue: OOM During Training

**Solutions**:

1. **Check wrapping**:
   ```python
   # Print FSDP structure
   print(model)
   # Each transformer layer should be wrapped
   ```

2. **Reduce batch size**: Activations scale with batch

3. **Enable activation checkpointing**:
   ```python
   model.gradient_checkpointing_enable()
   ```

4. **Use CPU offload** (last resort):
   ```python
   cpu_offload=CPUOffload(offload_params=True)
   ```

---

### Issue: Slow Training

**Checklist**:

1. **Communication overhead**: Profile with PyTorch profiler
   ```python
   with torch.profiler.profile() as prof:
       model(batch)
   print(prof.key_averages().table(sort_by="cuda_time_total"))
   ```

2. **Wrapping granularity**: Too fine → overhead, too coarse → memory

3. **Use FSDP2 + torch.compile**:
   ```python
   model = torch.compile(fully_shard(model))
   ```

4. **HSDP for multi-node**:
   ```python
   sharding_strategy=ShardingStrategy.HYBRID_SHARD
   ```

---

## 14. Integration with Hugging Face

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    bf16=True,  # Use BF16 for stability
    fsdp="full_shard auto_wrap",  # Enable FSDP
    fsdp_config={
        "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"]
    },
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

---
## 15. Key Takeaways

1. **FSDP is PyTorch's native ZeRO-3** - shards all model states
2. **Wrapping policy is critical** - determines memory efficiency
3. **Memory scales linearly**: Ψ/N per GPU (14 GB for 7B on 8 GPUs)
4. **FSDP2 is better** - works with `torch.compile`, 10-30% faster
5. **HSDP for multi-node** - shard within nodes, replicate across
6. **Use for models that don't fit on single GPU** (7B+ typically)
7. **Communication overhead** - 2× All-Gather per layer (vs DDP's 1× All-Reduce per step)
8. **Trade-off**: More memory savings than ZeRO-2, more communication than ZeRO-2

---