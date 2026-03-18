## 1. The Memory-Compute-Communication Triangle

Distributed training is fundamentally constrained by a three-way trade-off:

- **Memory**: Limits model and batch size per GPU
- **Compute**: Determines forward/backward pass speed
- **Communication**: Determines synchronization speed

> Improving one dimension often degrades another. Real systems balance all three.

---

---

## 2. Why We Need Parallelism

### Memory Requirements (7B Parameter Model Example)

For mixed precision training with Adam optimizer:

- **Parameters (FP16)**: 7B × 2 bytes = 14 GB
- **Gradients (FP16)**: 7B × 2 bytes = 14 GB
- **Optimizer States (FP32)**:

  - Master weights: 7B × 4 = 28 GB
  - Momentum: 7B × 4 = 28 GB
  - Variance: 7B × 4 = 28 GB
  - Subtotal: 84 GB

**Total**: ~112 GB (excludes activations!)

This doesn't fit on an 80GB A100 → **Need sharding strategies**

---

---

## 3. The Four Main Parallelism Types

| Type | What It Splits | When to Use | Communication |
|------|---------------|-------------|---------------|
| **Data Parallelism (DP)** | Training batches | Default baseline | All-Reduce gradients once per step |
| **Tensor Parallelism (TP)** | Individual layers | When layers don't fit on one GPU | All-Reduce/All-Gather in fwd/bwd pass |
| **Pipeline Parallelism (PP)** | Model depth (layers) | Very deep models | Activation passing between stages |
| **ZeRO / FSDP** | Model states (params, grads, optimizer) | Large models, memory constrained | Reduce-Scatter, All-Gather |

---

---

## 4. Hybrid Parallelism in Practice

Real systems combine multiple strategies:

```
┌─────────────────────────────────────┐
│  Data Parallel (across nodes)      │
│  ┌───────────────────────────────┐ │
│  │ Tensor Parallel (within node)│ │
│  │ ┌───────────────────────────┐ │ │
│  │ │ Pipeline Parallel (depth) │ │ │
│  │ └───────────────────────────┘ │ │
│  └───────────────────────────────┘ │
└─────────────────────────────────────┘
```

**Example scaling strategy (1 GPU → 1024 GPUs)**:

1. Start with Data Parallelism
2. Add ZeRO-2 when optimizer states don't fit
3. Add Tensor Parallelism (8-way) within nodes for very large layers
4. Add Pipeline Parallelism for extreme depth
5. Scale Data Parallelism across nodes

---

---

## 5. Training vs Inference Optimization

### Training (Throughput-Oriented)
- Goal: **Maximum tokens/second**
- Large batch sizes
- Heavy parallelism with communication overlap
- Higher per-batch latency acceptable

**Common choices**:

- Data Parallelism with large global batches
- Pipeline Parallelism with micro-batching
- Aggressive communication-computation overlap

---

### Inference (Latency-Oriented)
- Goal: **Fast response per request**
- Small/dynamic batch sizes
- Minimize synchronization

**Common choices**:

- Model replication over sharding (for small models)
- Limited/no Pipeline Parallelism (bubble overhead)
- Kernel fusion and caching

> **Key Insight**: Training amortizes communication over large batches; inference cannot hide communication latency easily.

---

---

## 6. Memory Reduction Techniques

### Activation Checkpointing
- **What**: Discard activations during forward, recompute in backward
- **Trade-off**: Memory ↓, Compute ↑ (20-40% overhead)
- **When**: Memory-constrained, compute is not bottleneck

### Mixed Precision Training
- **What**: FP16/BF16 compute, FP32 optimizer states
- **Benefit**: 2× memory savings, faster compute
- **Risk**: Numerical instability (use loss scaling)

### Gradient Accumulation
- **What**: Accumulate gradients over K steps before optimizer update
- **Benefit**: Simulates larger batch size, reduces sync frequency
- **Effective batch**: K × mini_batch_size

---

---

## 7. Critical Communication Primitives

| Operation | Purpose | Example Use |
|-----------|---------|-------------|
| **All-Reduce** | Sum and broadcast result | Gradient averaging in DP |
| **All-Gather** | Collect shards to all GPUs | Gather parameter shards in ZeRO-3 |
| **Reduce-Scatter** | Sum then split result | Distribute gradient shards |
| **Broadcast** | Send from one to all | Parameter initialization |

---

---

## 8. Common Interview Questions

**Q1: "How would you scale training from 1 GPU to 256 GPUs?"**

**Answer**:

1. **1-8 GPUs**: Start with Data Parallelism (DDP)
2. **8-64 GPUs**: Add ZeRO-2 to shard optimizer states and gradients
3. **64-256 GPUs**: Consider:

   - ZeRO-3/FSDP for full parameter sharding
   - 8-way Tensor Parallelism within nodes
   - 4-way Pipeline Parallelism if model is very deep
   - Data Parallelism across remaining dimension

---

**Q2: "What's the difference between model parallelism and data parallelism?"**

**Answer**:

- **Data Parallelism**: Replicate entire model on each GPU, split batch
  
  - Communication: Once per iteration (gradient sync)
  - Limitation: Model must fit on single GPU
  
- **Model Parallelism**: Split model itself across GPUs

  - **Tensor Parallelism**: Split individual layers
  - **Pipeline Parallelism**: Split by depth
  - Communication: Multiple times per forward/backward pass
  - Enables training models larger than single GPU memory

---

**Q3: "Why can't we just use Data Parallelism for everything?"**

**Answer**:

Memory constraints. Standard DP requires:

- Full model parameters on each GPU
- Full gradients on each GPU
- Full optimizer states on each GPU

For large models (>10B parameters), this exceeds single GPU memory even before considering activations.

---

---

## 9. Key Takeaways for Interviews

1. **No single best strategy** - choice depends on model size, hardware, and constraints
2. **Communication is often the bottleneck** at scale
3. **Overlap computation with communication** to hide latency
4. **Hybrid approaches** are standard in production (e.g., DP + TP + ZeRO)
5. **Memory accounting** is critical - know the 16Ψ rule for Adam + FP16
6. **Training ≠ Inference** - different optimization goals require different parallelism strategies

---