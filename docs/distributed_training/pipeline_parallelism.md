## 1. Core Concept

Pipeline Parallelism splits the model **by layer depth** across devices.

**Key Idea**: Each GPU owns a contiguous block of layers (a "stage").

```
GPU 0: Layers 1-3   ─┐
GPU 1: Layers 4-6    ├─ Forward: 0→1→2→3
GPU 2: Layers 7-9    │  Backward: 3→2→1→0
GPU 3: Layers 10-12 ─┘
```

**When to use**: Very deep models where depth is the bottleneck.

---

---

## 2. The Pipeline Bubble Problem

### Naive Pipeline (No Micro-Batching)

```
Timeline for 4 GPUs processing 1 batch:

GPU 0: [FWD]                    [BWD]
GPU 1:      [FWD]          [BWD]
GPU 2:           [FWD] [BWD]
GPU 3:                [FWD] [BWD]

       ████ Compute    □□□□ Idle (bubble)
```

**Problem**: Only one GPU is active at a time → **75% idle time!**

This idle time is called the **"pipeline bubble"**.

---

---

## 3. Micro-Batching Solution

Split the global batch into **M micro-batches** that flow through the pipeline.

### Example: 4 Micro-Batches

```
Timeline with 4 micro-batches (F=Forward, B=Backward):

GPU 0: F1 F2 F3 F4          B4 B3 B2 B1
GPU 1:    F1 F2 F3 F4       B4 B3 B2 B1
GPU 2:       F1 F2 F3 F4    B4 B3 B2 B1
GPU 3:          F1 F2 F3 F4 B4 B3 B2 B1

       ████ Compute    □ Small bubbles at start/end
```

**Result**: Much better GPU utilization!

---

---

## 4. Pipeline Bubble Calculation

**Bubble fraction** = (Number of pipeline stages - 1) / Number of micro-batches

For P pipeline stages and M micro-batches:

- Bubble time = `(P - 1) / M`

**Example**:
- 4 stages, 1 micro-batch: 3/1 = **75% bubble** ❌
- 4 stages, 8 micro-batches: 3/8 = **37.5% bubble** ⚠️
- 4 stages, 16 micro-batches: 3/16 = **18.75% bubble** ✅

> **Rule of thumb**: Use at least 4× micro-batches as pipeline stages.

---

---

## 5. Pipeline Schedules

### 1. GPipe (Fill-Drain)

Simple schedule: Fill pipeline, then drain it.

```
F=Forward, B=Backward

Stage 0: F1 F2 F3 F4 -- -- B4 B3 B2 B1
Stage 1: -- F1 F2 F3 F4 -- -- B4 B3 B2
Stage 2: -- -- F1 F2 F3 F4 -- -- B4 B3
Stage 3: -- -- -- F1 F2 F3 F4 -- -- B4
```

**Pros**: Simple to implement  
**Cons**: Large bubbles at start and end

---

### 2. PipeDream-Flush (1F1B)

**One-Forward-One-Backward**: Interleave forward and backward.

```
Stage 0: F1 F2 F3 F4 B1 B2 B3 B4
Stage 1: -- F1 F2 F3 B1 B2 B3 F4 B4
Stage 2: -- -- F1 F2 B1 B2 F3 F4 B3 B4
Stage 3: -- -- -- F1 B1 F2 F3 F4 B2 B3 B4
```

**Pros**: Smaller bubbles, earlier gradient computation  
**Cons**: More complex synchronization

---

### 3. Interleaved 1F1B (Virtual Pipeline Stages)

Each GPU handles **multiple non-contiguous stages**.

```
Model split into 8 virtual stages on 4 GPUs:
GPU 0: Stages 1, 5
GPU 1: Stages 2, 6
GPU 2: Stages 3, 7
GPU 3: Stages 4, 8
```

**Pros**: Further reduces bubble  
**Cons**: More complex, higher communication

---

---

## 6. Memory Trade-offs

### Activation Memory

Unlike DP or TP, PP **increases activation memory**.

**Why?**
- Must store activations for all in-flight micro-batches
- Each micro-batch's activations held until backward pass

**Memory**: O(M × activation_size)

Where M = number of micro-batches.

### Memory vs Utilization Trade-off

- **More micro-batches**: Better utilization, **higher memory**
- **Fewer micro-batches**: Lower memory, **worse utilization**

```
Example: GPT-3 175B
- 8 pipeline stages
- 32 micro-batches → Good utilization but ~4GB activations per stage
- 8 micro-batches → Poor utilization but ~1GB activations per stage
```

---

---

## 7. Communication Pattern

### Forward Pass
- Send activations from stage i to stage i+1
- Point-to-point communication (Send/Recv)

### Backward Pass
- Send gradients from stage i+1 to stage i
- Reverse order of forward pass

**Bandwidth**: Lower than TP (only inter-stage, not all-to-all)  
**Latency**: Higher than DP (sequential dependency)

---

---

## 8. Common Interview Questions

### Q1: "Explain the pipeline bubble and how to reduce it."

**Strong Answer**:

**Pipeline bubble** = idle time when GPUs wait for activations/gradients.

**Cause**: With naive pipelining, only one stage is active at a time.

**Solutions**:
1. **Micro-batching**: Split batch into M chunks
   - Bubble = (P-1)/M where P = num stages
   - Use M ≥ 4P for <25% bubble

2. **Better schedules**: 1F1B instead of fill-drain
   - Reduces bubble further
   - Interleaves forward and backward

3. **Virtual pipeline stages**: Each GPU handles multiple non-contiguous stages
   - Minimal bubble
   - Added complexity

---

### Q2: "What's the memory overhead of pipeline parallelism?"

**Answer**:

**Increased memory**:
- **Activation storage**: Must hold activations for all M micro-batches in flight
- Memory grows with M: more micro-batches → more memory
- Trade-off: Utilization ↑, Memory ↑

**Decreased memory**:
- **Parameter storage**: Each GPU only stores P/N layers (N = num GPUs)
- **Gradient storage**: Also P/N
- **Optimizer states**: Also P/N

**Net effect**: Saves parameter memory but increases activation memory.

**Example**: 48-layer model on 4 GPUs
- Each GPU: 12 layers instead of 48
- But: Must store activations for 16 micro-batches

---

### Q3: "Compare Pipeline Parallelism vs Tensor Parallelism."

**Answer**:

| Aspect | Pipeline Parallelism | Tensor Parallelism |
|--------|---------------------|-------------------|
| **Splits** | Layers (depth) | Individual layers (width) |
| **Communication** | Point-to-point (between stages) | All-to-all (All-Gather, All-Reduce) |
| **Frequency** | Once per micro-batch | Every layer in fwd/bwd |
| **Latency** | Higher (sequential) | Lower (parallel) |
| **Memory** | ↑ Activations, ↓ Parameters | ↓ Both |
| **Bubble** | Yes (need micro-batching) | No |
| **Use case** | Very deep models | Very wide layers |

**Often combined**: TP within stages, PP across stages.

---

### Q4: "How do you choose the number of pipeline stages?"

**Answer**:

**Considerations**:

1. **Model depth**: Natural split points (e.g., every 12 transformer layers)

2. **Memory balance**: Each stage should have similar memory footprint

3. **Communication**: More stages → more inter-stage communication

4. **Bubble**: More stages → larger bubble (need more micro-batches)

**Typical choices**:
- GPT-3 175B: 8-16 pipeline stages
- Smaller models (<10B): 2-4 stages if used at all
- Often matches node count (1 stage per node)

**Rule**: Balance memory reduction with bubble overhead. Usually 4-16 stages.

---

### Q5: "Why does Pipeline Parallelism hurt latency-sensitive workloads?"

**Answer**:

**In training**: Not a huge issue because we optimize for throughput
- Large batches amortize bubble overhead
- Micro-batching keeps GPUs busy

**In inference**: Major problem
- Small batch sizes (often batch=1)
- Can't use micro-batching effectively
- Sequential dependency → high latency
- Token generation is sequential by nature

**Result**: PP rarely used for inference. Prefer model replication or TP.

---

---

## 9. Practical Implementation

### GPipe Style (PyTorch)

```python
import torch.distributed as dist

class PipelineStage:
    def __init__(self, module, stage_id, num_stages):
        self.module = module
        self.stage_id = stage_id
        self.num_stages = num_stages
    
    def forward(self, micro_batches):
        outputs = []
        
        # Forward pass for all micro-batches
        for mb in micro_batches:
            if self.stage_id > 0:
                # Receive from previous stage
                mb = recv_from_previous_stage()
            
            out = self.module(mb)
            
            if self.stage_id < self.num_stages - 1:
                # Send to next stage
                send_to_next_stage(out)
            
            outputs.append(out)
        
        # Backward pass (reversed order)
        for out in reversed(outputs):
            out.backward()
        
        return outputs
```

---

---

## 10. Gradient Accumulation vs Micro-batching

### Gradient Accumulation (DP)
```
for micro_batch in batches:
    loss = model(micro_batch)
    loss.backward()  # Accumulate gradients
optimizer.step()  # One update
```
- All micro-batches on same GPU
- No communication until optimizer step

---

### Micro-batching (PP)
```
# Different micro-batches on different stages simultaneously
GPU 0: micro_batch_1 → GPU 1 → GPU 2 → GPU 3
GPU 0: micro_batch_2 → ...
```
- Each micro-batch flows through pipeline
- Communication after each stage
- Enables pipeline parallelism

**Key difference**: PP micro-batches are **in-flight simultaneously** across stages.

---

---

## 11. Debugging Tips

### Issue: Poor Utilization
**Symptoms**: GPUs idle most of the time

**Solutions**:
- Increase number of micro-batches (4× pipeline stages minimum)
- Use 1F1B schedule instead of GPipe
- Profile bubble time

### Issue: Out of Memory
**Symptoms**: OOM during training

**Solutions**:
- Reduce number of micro-batches (less activation memory)
- Use activation checkpointing
- Reduce batch size per micro-batch

### Issue: Slow Communication
**Symptoms**: High time in Send/Recv

**Solutions**:
- Check inter-node bandwidth
- Ensure balanced stage sizes
- Consider hybrid TP+PP (TP within node, PP across nodes)

---

---

## 12. Advanced: Hybrid PP + TP

Most efficient setup for large models:

```
┌─────────────────┐
│  Pipeline Stage 0  │  ← 8-way Tensor Parallel
│  (Layers 1-12)     │
│  [GPU 0-7]         │
└─────────────────┘
        ↓ activation
┌─────────────────┐
│  Pipeline Stage 1  │  ← 8-way Tensor Parallel
│  (Layers 13-24)    │
│  [GPU 8-15]        │
└─────────────────┘
```

**Benefits**:
- TP reduces per-stage memory (use NVLink within node)
- PP reduces total parameter memory (across nodes)
- Minimizes cross-node communication (only inter-stage activations)

---

---

## 13. Key Takeaways

1. **PP splits model by depth** - each GPU owns contiguous layers
2. **Pipeline bubble is unavoidable** - use micro-batching to minimize (aim for <20%)
3. **Memory trade-off**: ↓ Parameters, ↑ Activations
4. **More micro-batches** → better utilization but higher memory
5. **1F1B schedule** is better than naive fill-drain
6. **Not ideal for inference** due to latency overhead
7. **Often combined with TP** - TP within stages, PP across stages
8. **Rule: M ≥ 4P** (micro-batches ≥ 4× pipeline stages)

---