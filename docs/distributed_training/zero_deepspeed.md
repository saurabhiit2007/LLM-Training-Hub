# ZeRO and DeepSpeed

## 1. The Core Problem

**Standard Data Parallelism** replicates everything on each GPU:

- Model parameters (Ψ)
- Gradients (Ψ)
- Optimizer states (12Ψ for Adam)

**Total per GPU**: ~16Ψ bytes

For a 7B model:

- 7B × 16 bytes = **112 GB** (doesn't fit on 80GB GPU!)

**ZeRO's insight**: This redundancy is unnecessary. Share the memory burden.

---

## 2. ZeRO: Zero Redundancy Optimizer

**Goal**: Eliminate memory redundancy while keeping the efficiency of Data Parallelism.

### The Three Stages

| Stage | Shards | Memory per GPU | Communication | Example (7B, 8 GPUs) |
|-------|--------|----------------|---------------|---------------------|
| **ZeRO-1** | Optimizer states only | ~12Ψ | Minimal (gather optimizer shards) | 14 + 14 + 84/8 = **38.5 GB** |
| **ZeRO-2** | Optimizer + Gradients | ~8Ψ | All-Reduce gradient shards | 14 + 14/8 + 84/8 = **26.25 GB** |
| **ZeRO-3** | Optimizer + Gradients + Parameters | ~2Ψ | Gather params on-the-fly | 14/8 + 14/8 + 84/8 = **14 GB** |

> **Note**: These numbers exclude activations, which can be substantial.

---

## 3. Stage-by-Stage Breakdown

### ZeRO-1: Optimizer State Sharding

**What's sharded**: Only optimizer states (momentum, variance)

**Each GPU stores**:

- Full parameters (2Ψ)
- Full gradients (2Ψ)
- 1/N of optimizer states (12Ψ / N)

**Communication**:

- Standard All-Reduce for gradients
- Gather optimizer state shards when needed for update

**Memory savings**: 4× (from 16Ψ to 12Ψ)

**Use case**: Drop-in improvement over DDP with minimal overhead

---

### ZeRO-2: + Gradient Sharding

**What's sharded**: Optimizer states + gradients

**Each GPU stores**:

- Full parameters (2Ψ)
- 1/N of gradients (2Ψ / N)
- 1/N of optimizer states (12Ψ / N)

**Communication**:

- **Reduce-Scatter** instead of All-Reduce during backward
  - Each GPU receives only its gradient shard
- Gather optimizer shards for update (same as ZeRO-1)

**Memory savings**: 8× (from 16Ψ to ~8Ψ)

**Use case**: Models up to ~13B on 8×A100 (40GB)

---

### ZeRO-3: + Parameter Sharding

**What's sharded**: Everything (optimizer states + gradients + parameters)

**Each GPU stores**:

- 1/N of parameters (2Ψ / N)
- 1/N of gradients (2Ψ / N)
- 1/N of optimizer states (12Ψ / N)

**Communication**:

**Forward pass**:

1. All-Gather to reconstruct layer parameters
2. Compute forward
3. Discard parameters

**Backward pass**:

1. All-Gather to reconstruct layer parameters (again)
2. Compute backward
3. Reduce-Scatter gradients to owning GPU
4. Discard parameters

**Memory savings**: N× (linear scaling with GPUs)

**Trade-off**: Significantly more communication (2× All-Gather per layer)

**Use case**: Very large models (70B+ parameters)

---

## 4. ZeRO-3 Communication Pattern

```python
# Pseudo-code for one layer forward/backward

# Forward
params = all_gather(param_shard)  # Reconstruct full parameters
output = layer(input, params)
del params  # Free memory immediately

# Backward  
params = all_gather(param_shard)  # Reconstruct again
grad_input, grad_params = backward(output_grad, params)
del params

grad_shard = reduce_scatter(grad_params)  # Each GPU gets its shard
```

**Key insight**: Parameters are materialized **only when needed**, then immediately freed.

---

## 5. ZeRO Offload

**Problem**: Even ZeRO-3 may not fit very large models on GPU.

**Solution**: Offload to CPU RAM (or even NVMe).

### CPU Offload

**What's offloaded**:

- Optimizer states (always)
- Parameters (optional)
- Gradients (optional)

**Workflow**:
```
GPU: Compute-heavy operations (forward, backward)
  ↕ PCIe transfer
CPU: Store optimizer states, occasionally parameters
```

**When to use**:

- Training 10B+ models on consumer GPUs (e.g., RTX 3090)
- Limited GPU memory
- Have sufficient CPU RAM

**Trade-off**: GPU memory ↓↓, Training speed ↓ (PCIe bandwidth: ~25 GB/s vs GPU memory: ~2000 GB/s)

---

### NVMe Offload (ZeRO-Infinity)

**Extreme offloading** to SSD storage.

**Use case**: Trillion-parameter models (research, not production)

**Trade-off**: GPU memory ↓↓↓, Training speed ↓↓↓ (NVMe: ~7 GB/s)

---

## 6. DeepSpeed: The Library

**DeepSpeed** is Microsoft's implementation of ZeRO with many optimizations.

### Key Features

1. **ZeRO Stages 1-3**: As described above

2. **ZeRO-Offload**: CPU and NVMe offloading

3. **Optimized Kernels**: Custom CUDA kernels for:
   - Fused Adam optimizer
   - Fused layer norm
   - Communication-computation overlap

4. **Mixed Precision**: FP16, BF16, FP8 support

5. **Gradient Accumulation**: Built-in with ZeRO

6. **Activation Checkpointing**: Integrated memory optimization

---

### Configuration Example

```json
{
  "train_batch_size": 256,
  "gradient_accumulation_steps": 16,
  "gradient_clipping": 1.0,
  
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 16
  },
  
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 500000000,
    "stage3_prefetch_bucket_size": 500000000,
    "stage3_param_persistence_threshold": 1000000
  }
}
```

---

## 8. Memory Calculation Example

**7B model, 8 GPUs, FP16, Adam optimizer**

### Without ZeRO (Standard DP)
Per GPU:

- Parameters: 14 GB
- Gradients: 14 GB
- Optimizer: 84 GB
- **Total: 112 GB** ❌ (doesn't fit on 80GB A100)

### With ZeRO-1
Per GPU:

- Parameters: 14 GB
- Gradients: 14 GB
- Optimizer: 84/8 = 10.5 GB
- **Total: 38.5 GB** ✅

### With ZeRO-2
Per GPU:

- Parameters: 14 GB
- Gradients: 14/8 = 1.75 GB
- Optimizer: 84/8 = 10.5 GB
- **Total: 26.25 GB** ✅✅

### With ZeRO-3
Per GPU:

- Parameters: 14/8 = 1.75 GB
- Gradients: 14/8 = 1.75 GB
- Optimizer: 84/8 = 10.5 GB
- **Total: 14 GB** ✅✅✅

> **Note**: These exclude activations! With batch_size=32, sequence_length=2048, activations can be 10-20 GB.

---

## 9. Practical Tips

### 1. Start with ZeRO-2
```python
# Usually sufficient for most models
# Less communication overhead than ZeRO-3
ds_config = {
    "zero_optimization": {"stage": 2}
}
```

### 2. Monitor Communication
```python
# Profile with DeepSpeed timer
from deepspeed.runtime.utils import see_memory_usage
see_memory_usage("After forward", force=True)
```

### 3. Tune Bucket Sizes
```python
# Larger buckets = less overhead, less overlap
# Smaller buckets = more overhead, more overlap
{
    "reduce_bucket_size": 500_000_000,  # 500MB
    "allgather_bucket_size": 500_000_000
}
```

### 4. Enable Communication Overlap
```python
{
    "overlap_comm": true,  # Essential for ZeRO-3
    "contiguous_gradients": true
}
```

---

## 10. Integration with Hugging Face

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    fp16=True,
    deepspeed="ds_config.json",  # DeepSpeed config file
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

---

## Key Takeaways

1. **ZeRO eliminates redundancy** in Data Parallelism by sharding model states
2. **Three stages**: Optimizer (Z1), + Gradients (Z2), + Parameters (Z3)
3. **ZeRO-2 is the sweet spot** for most use cases (good memory savings, low overhead)
4. **ZeRO-3 for extreme scale** (70B+) - trades communication for memory
5. **Offloading enables training on limited GPUs** but significantly slows training
6. **DeepSpeed is the reference implementation** with optimized kernels
7. **Communication-computation overlap is critical** for ZeRO-3 efficiency
8. **Memory savings scale linearly with GPUs** (Z3: 112GB → 14GB on 8 GPUs)

---