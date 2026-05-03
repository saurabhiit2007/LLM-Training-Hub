# Data Parallelism

## 1. Overview

The most widely used baseline parallelism strategy:
- **Full model replicated** across N workers
- Each worker processes a **different shard** of the input batch
- Gradients are **synchronized** after backward pass
- All replicas apply **identical optimizer updates**

---

## 2. How It Works

```
GPU 0: Model Copy | Batch[0:32]   → Forward → Backward → Gradients_0
GPU 1: Model Copy | Batch[32:64]  → Forward → Backward → Gradients_1
GPU 2: Model Copy | Batch[64:96]  → Forward → Backward → Gradients_2
GPU 3: Model Copy | Batch[96:128] → Forward → Backward → Gradients_3

                      ↓ All-Reduce ↓

GPU 0,1,2,3: Averaged Gradients → Optimizer Step (synchronized)
```

**Result**: Effective batch size = N × local batch size

---

## 3. PyTorch DDP Implementation Details

### 1. Gradient Bucketing

**Problem**: Waiting for all gradients before communication wastes time.

**Solution**: Group parameters into ~25MB buckets
- Communication starts as soon as a bucket is ready
- Reduces idle time by avoiding one large sync at the end

```python
# Conceptual example
# Parameters grouped by size:
# Bucket 1: [layer1.weight, layer1.bias] → 23MB
# Bucket 2: [layer2.weight, layer2.bias] → 24MB
# As soon as Bucket 1 gradients ready → All-Reduce starts
```

**Configuration**:
```python
model = DDP(model, bucket_cap_mb=25)  # Default: 25MB
```

---

### 2. Asynchronous Gradient Reduction

**Key Optimization**: Overlap communication with computation

**How it works**:
1. Backward pass computes gradients layer-by-layer (last → first)
2. When a bucket's gradients are ready, DDP launches **async All-Reduce**
3. While communication runs in background, backward continues on earlier layers
4. Ideally, by the time backward finishes, most communication is done

```
Timeline:
────────────────────────────────────────────────
Computation: [Layer N] [Layer N-1] ... [Layer 1]
                 ↓        ↓             ↓
Comm:        [Bucket 1 All-Reduce─────────────]
                     [Bucket 2 All-Reduce────]
                           [Bucket 3 All-Reduce]
────────────────────────────────────────────────
```

> **Common Misconception**: DDP does NOT sync after every layer. Synchronization is event-driven and overlaps with backpropagation.

---

### 3. Synchronization Timing

- Gradients synchronized **once per training iteration** (not per layer)
- Each parameter's gradient reduced **exactly once** after computation
- Optimizer step happens only after **all reductions complete**

---

## 4. Memory Requirements

Each GPU must store:
- **Model parameters**: 2Ψ (FP16) or 4Ψ (FP32)
- **Gradients**: 2Ψ (FP16) or 4Ψ (FP32)
- **Optimizer states (Adam)**: 12Ψ (FP32 master + momentum + variance)
- **Activations**: Depends on batch size and sequence length

**Total**: ~16Ψ bytes per GPU (excluding activations)

---

## 5. Communication Cost

### All-Reduce Complexity

For N GPUs and Ψ parameters:
- **Data transferred**: 2Ψ(N-1)/N per GPU (ring All-Reduce)
- **Latency**: O(log N) for tree-based, O(N) for ring-based
- Grows **linearly** with number of GPUs

---

### Bandwidth Requirements

For 7B model (14GB gradients in FP16):
- 100Gbps network: ~1.1 seconds just for communication
- 400Gbps network: ~0.28 seconds

> Communication overhead becomes significant beyond 64-128 GPUs

---

## 6. Gradient Accumulation with DDP

**Purpose**: Simulate larger batch size without increasing memory

```python
for i, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = outputs.loss / accumulation_steps
    loss.backward()  # Gradients accumulate
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()  # Only sync here
        optimizer.zero_grad()
```

**Benefits**:
- Effective batch = `accumulation_steps × local_batch × num_gpus`
- **Reduces sync frequency** from every step to every K steps
- Saves communication bandwidth

**Trade-off**: Slower convergence per step (but same per sample)

---

## 8. Best Practices

### 1. Bucket Size Tuning
```python
# Smaller buckets: More overlap, but more overhead
# Larger buckets: Less overhead, but less overlap
model = DDP(model, bucket_cap_mb=25)  # Default is usually good
```

### 2. Gradient Clipping
```python
# Must happen BEFORE optimizer step
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### 3. Find Unused Parameters
```python
# If model has conditional branches
model = DDP(model, find_unused_parameters=True)
# Warning: Adds overhead, only use when necessary
```

### 4. Static Graph Optimization
```python
# If model structure doesn't change
model = DDP(model, static_graph=True)  # PyTorch 1.11+
# Reduces overhead by ~10%
```

---

## 9. Debugging Tips

### Issue: "RuntimeError: Expected to have finished reduction in the prior iteration"

**Cause**: Uneven workload across GPUs (e.g., variable sequence lengths)

**Fix**:
```python
# Option 1: Pad to same length
# Option 2: Use find_unused_parameters=True
# Option 3: Manual synchronization
torch.distributed.barrier()
```

### Issue: "Out of Memory with DDP"

**Checklist**:
1. Check if activation memory is the issue (use activation checkpointing)
2. Reduce local batch size
3. Switch to ZeRO-2/FSDP to shard optimizer and gradients

---
