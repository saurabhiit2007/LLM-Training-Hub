# 💾 Memory Considerations for LLM Training and Inference

## 1. GPU and CPU Memory and Transfer Rates

### 1.1 GPU Memory

GPU memory (HBM - High Bandwidth Memory) is high bandwidth memory physically attached to the GPU package, designed to feed thousands of parallel compute cores efficiently.

**Key characteristics:**

- Very high bandwidth
- Low access latency relative to CPU memory
- Limited capacity compared to system RAM

**Typical values for NVIDIA A100:**

- Capacity: 40GB or 80GB HBM2e
- Peak bandwidth: ~1.6 TB/s

**GPU memory stores:**

- Model weights
- Activations
- Gradients
- Optimizer states
- KV cache during inference

---

### 1.2 CPU Memory

CPU memory refers to system RAM (DDR4 or DDR5) located on the motherboard.

**Key characteristics:**

- Much larger capacity (64GB to 1TB+)
- Significantly lower bandwidth (~50 to 100 GB/s per socket)
- Higher latency compared to GPU memory

**CPU memory is used for:**

- Data loading and preprocessing
- Checkpoint storage before transfer
- Offloaded parameters or optimizer states (ZeRO-Offload, Paged Adam)

---

### 1.3 GPU to CPU Transfer Rates

Data movement between GPU and CPU happens over interconnects.

**Approximate peak transfer rates:**

- PCIe Gen4: ~32 GB/s
- PCIe Gen5: ~64 GB/s
- NVLink (A100): ~300 GB/s

**Key insight:** Even with NVLink, transfer bandwidth is far lower than on-device GPU memory bandwidth (1.6 TB/s), making frequent transfers expensive.

---

---

## 2. Bandwidth, Latency, and Compute

### 2.1 Bandwidth vs Latency vs Compute

| Metric | Definition | Typical Values (A100) | Matters For |
|--------|-----------|----------------------|-------------|
| **Bandwidth** | Data transfer rate | ~1.6 TB/s (HBM) | Large tensor operations |
| **Latency** | Time to first byte | ~100-200 ns (HBM) | Small tensor ops, kernel launch |
| **Compute** | Arithmetic capability | ~312 TFLOPs (BF16) | Matrix multiplications |

**In practice:**
- Many LLM workloads are **memory bandwidth bound**, not compute bound
- Compute units often wait on memory due to limited data reuse

---

### 2.2 Compute vs Memory Bound Regimes

**Arithmetic intensity** determines the bottleneck:
$$
\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes transferred}}
$$

**Transformers are often memory bound during:**

- Attention (especially long sequences)
- Layer normalization
- Optimizer updates
- Embedding lookups

**Compute bound operations:**

- Large matrix multiplications (when batch size and dimensions are large)

---

---

## 3. Training Memory for a 7B Model on a Single A100

### 3.1 Memory Components During Training

Training requires four major components in GPU memory:

1. Model weights
2. Gradients
3. Optimizer states
4. Activations

**Quantifying each component:**

Let $P = 7 \times 10^9$ parameters, BF16 for weights/gradients (2 bytes), FP32 for optimizer (4 bytes).

#### 3.1.1 Model Weights

$$
M_{\text{weights}} = P \times 2 \text{ bytes} = 14 \text{ GB}
$$

#### 3.1.2 Gradients

$$
M_{\text{grads}} = P \times 2 \text{ bytes} = 14 \text{ GB}
$$

#### 3.1.3 Optimizer States (Adam/AdamW)

Adam maintains two FP32 states per parameter (momentum and variance):

$$
M_{\text{optimizer}} = P \times 2 \times 4 \text{ bytes} = 56 \text{ GB}
$$

**Running total: 14 + 14 + 56 = 84 GB** (already exceeds A100 80GB!)

#### 3.1.4 Activations

Activation memory depends on layers $L$, batch size $B$, sequence length $S$, hidden dimension $d$:

$$
M_{\text{act}} \approx L \times B \times S \times d \times C \times 2 \text{ bytes}
$$

Where $C$ is a constant (typically 10-20 without gradient checkpointing).

**For 7B model:**
- Without checkpointing: ~32 GB
- With checkpointing: ~2-5 GB

#### 3.1.5 Total Training Memory

$$
M_{\text{total}} = M_{\text{weights}} + M_{\text{grads}} + M_{\text{optimizer}} + M_{\text{act}}
$$

**Without gradient checkpointing:** ~116 GB  
**With gradient checkpointing:** ~89 GB

**This explains why full fine-tuning of a 7B model doesn't fit on a single A100.**

---

### 3.2 Techniques That Enable Feasible Training

**Common strategies:**

| Technique | Memory Savings | Trade-off |
|-----------|---------------|-----------|
| **LoRA** | 100-1000× (trainable params) | Slightly lower quality |
| **Gradient checkpointing** | 80-90% (activations) | +25-33% compute time |
| **8-bit optimizers** | 75% (optimizer states) | Minimal quality loss |
| **Gradient accumulation** | 50-75% (activations) | Longer iteration time |
| **Mixed precision (BF16)** | 50% (weights/grads/acts) | Requires compatible hardware |

**Typical combination for 7B on 80GB A100:**
- BF16 mixed precision
- Gradient checkpointing
- 8-bit Adam
- Batch size 1-2 with gradient accumulation

---

### 3.3 CPU Offloading

**What can be offloaded:**
- Optimizer states (ZeRO-Offload, Paged Adam)
- Parameters (ZeRO-Infinity with NVMe)

**Trade-offs:**

✅ Enables fitting larger models  
❌ Severely reduces training throughput (50-100× slower optimizer step)  
❌ Often impractical for production pretraining

**When to use:** Single-GPU fine-tuning with memory constraints.

---

---

## 4. Memory Differences Between Training and Inference

### 4.1 Training vs Inference Memory Profile

| Component | Training | Inference | Notes |
|-----------|----------|-----------|-------|
| **Weights** | 14 GB | 14 GB | Same |
| **Gradients** | 14 GB | 0 GB | No backward pass |
| **Optimizer** | 56 GB | 0 GB | Not needed |
| **Activations** | 5-32 GB | 1-2 GB | No need to store for backprop |
| **KV cache** | 0 GB | 5-20 GB | Autoregressive decoding |
| **Total** | **89-116 GB** | **20-36 GB** | **3-5× difference** |

**Key insight:** Training requires 3-5× more memory than inference for the same model.

**Why inference fits models that training cannot:**
- No gradients
- No optimizer states
- Minimal activation storage

---

---

## 5. KV Cache Memory Usage

### 5.1 What Is the KV Cache

The KV cache stores key and value tensors from attention for previously processed tokens during **autoregressive decoding**.

**Purpose:** Avoid recomputing attention over the full context for each new token.

**Speed-up:** $O(S^2) \rightarrow O(S)$ for generating $S$ tokens.

---

### 5.2 Which Memory Does KV Cache Use

**KV cache resides in:**

✅ GPU memory (during standard inference)  
❌ CPU memory (only in specialized offloading setups)

**Why GPU?** Attention computation requires fast access; CPU storage would destroy throughput.

---

### 5.3 KV Cache Memory Scaling

**KV cache memory formula:**

$$
M_{\text{KV}} = 2 \times L \times B \times S \times d \times \text{bytes per element}
$$

Where:
- Factor of 2: both K and V
- $L$: number of layers
- $B$: batch size
- $S$: sequence length
- $d$: hidden dimension

**Example (7B LLaMA, BF16):**

For single request ($B=1$), sequence length 4096:
$$
M_{\text{KV}} = 2 \times 32 \times 1 \times 4096 \times 4096 \times 2 = 2.15 \text{ GB}
$$

For batch size 8:
$$
M_{\text{KV}} = 2.15 \times 8 = 17.2 \text{ GB}
$$

**This makes KV cache the dominant memory consumer during long-context inference.**

---

### 5.4 KV Cache Management Techniques

**Problem:** KV cache grows linearly with generated tokens and batch size.

**Solutions:**

| Technique | Memory Reduction | Notes |
|-----------|-----------------|-------|
| **Sliding window** | Caps at fixed size | Only keep last W tokens |
| **PagedAttention (vLLM)** | 10-30% | Better packing, shared prefixes |
| **Grouped-Query Attention (GQA)** | 4-32× | Fewer KV heads (requires model change) |
| **KV cache quantization (INT8)** | 50% | Store K/V in INT8 |

**GQA example (7B model, seq=4096):**

- Multi-Head (32 KV heads): 2.15 GB
- GQA (8 KV heads): 0.54 GB (4× reduction)
- MQA (1 KV head): 0.07 GB (32× reduction)

---

---

## 6. Multi-GPU Memory Considerations

### 6.1 Data Parallelism (DP)

**Memory per GPU:**
- Each GPU holds full model copy
- Activations/gradients split across batch

$$
M_{\text{GPU}} = M_{\text{weights}} + M_{\text{optimizer}} + \frac{M_{\text{activations}}}{N_{\text{GPUs}}}
$$

**Limitation:** Model must fit on single GPU.

---

### 6.2 ZeRO (Zero Redundancy Optimizer)

**ZeRO partitions optimizer states, gradients, and parameters:**

| Stage | Partitions | Memory Savings (vs DP) |
|-------|-----------|----------------------|
| **ZeRO-1** | Optimizer states | 4× |
| **ZeRO-2** | + Gradients | 8× |
| **ZeRO-3** | + Parameters | Up to $N_{\text{GPUs}}$× |

**Example (7B model, 8 GPUs, ZeRO-3):**

Per GPU memory:
$$
\frac{14 + 14 + 56}{8} = 10.5 \text{ GB}
$$

Enables training on GPUs with <16 GB memory.

**Trade-off:** More communication (all-gather parameters each layer) vs Data Parallel.

---

### 6.3 Tensor and Pipeline Parallelism

**Tensor Parallelism (TP):**
- Model weights sharded across GPUs
- Each GPU computes a slice (e.g., split attention heads)

**Pipeline Parallelism (PP):**
- Layers distributed across GPUs
- Sequential processing with pipeline stages

**Both reduce per-GPU memory but require careful implementation.**

---

---

## 7. Interview Questions

### Q1: Why does training a 7B model require ~100GB but inference only ~20GB?

**Answer:**

**Training (7B model):**
- Weights: 14 GB
- Gradients: 14 GB
- Optimizer (Adam): 56 GB
- Activations: 5-32 GB
- **Total: 89-116 GB**

**Inference (7B model):**
- Weights: 14 GB
- Activations: 1-2 GB (minimal)
- KV cache: 5-20 GB
- **Total: 20-36 GB**

**Key differences:**
- No gradients in inference (no backward pass)
- No optimizer states (not training)
- Activations only for current layer (not stored for backprop)

Optimizer states alone (56 GB) exceed entire inference budget.

---

### Q2: Walk through the memory bottleneck when fine-tuning a 13B model on a single 80GB A100.

**Answer:**

**Memory breakdown (13B parameters, BF16):**

| Component | Memory |
|-----------|--------|
| Weights | 26 GB |
| Gradients | 26 GB |
| Adam states | 104 GB ⚠️ |
| Activations | 10-40 GB |
| **Total** | **166-196 GB** ❌ |

**Problem:** Optimizer states alone exceed GPU capacity.

**Solutions (stacked):**

1. **Gradient checkpointing:** 40 GB → 5 GB (activations)
   - New total: ~161 GB (still doesn't fit)

2. **8-bit optimizer:** 104 GB → 26 GB (optimizer)
   - New total: 26 + 26 + 26 + 5 = **83 GB** (still over)

3. **Reduce batch size:** 4 → 1 with gradient accumulation
   - Activations: 5 GB → 2 GB
   - New total: **80 GB** ✅ (barely fits!)

**Alternative: LoRA (rank 32)**
- Trainable params: ~50M (0.4% of 13B)
- Total memory: 26 (weights) + 0.1 (grads) + 0.4 (optimizer) + 2 (act) = **28.5 GB** ✅

**Recommendation:** LoRA for efficiency, or 8-bit Adam + checkpointing + small batch for full fine-tuning.

---

### Q3: Explain why attention becomes memory-bound for long sequences.

**Answer:**

**Attention computation:**
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**The bottleneck:** The intermediate $S \times S$ attention matrix.

For sequence length $S=8192$:
- Attention matrix size: $8192^2 \times 2 \text{ bytes} = 134$ MB per head
- 32 heads: $32 \times 134 = 4.3$ GB just for attention scores
- This must be materialized in memory → **memory bound**

**Arithmetic intensity is low:**

For large $S$, most operations are reading/writing the $S^2$ matrix, not computing.

**Flash Attention solution:**
- Never materializes full $S \times S$ matrix
- Recomputes attention in blocks (tiling)
- Reduces memory: $O(S^2) \rightarrow O(S)$

---

### Q4: You're serving 100 requests with a 7B model. KV cache is 60GB. How can you reduce this?

**Answer:**

**Current state:** 60 GB / 100 requests = 0.6 GB per request

This implies ~1,144 tokens per request average (context + generation).

**Reduction strategies:**

| Strategy | Reduction | Trade-off |
|----------|-----------|-----------|
| **Reduce context to 512 tokens** | 55% | May truncate important context |
| **Use GQA model (8 KV heads vs 32)** | 75% | Requires different model |
| **PagedAttention (vLLM)** | 10-30% | Better packing, no quality loss |
| **KV cache quantization (INT8)** | 50% | Slight quality degradation |
| **Reduce concurrent requests (100→50)** | 50% | Lower throughput |

**Recommended:**
- vLLM with PagedAttention + context limit (1024 tokens)
- Achieves ~70% reduction without quality loss
- Total: ~18-24 GB

---

### Q5: Compare memory patterns of Data Parallel vs ZeRO-3.

**Answer:**

**Setup:** 8 GPUs, 7B model (14 GB parameters)

**Data Parallel (DP):**

Memory per GPU:
- Full model copy: 14 GB
- Full optimizer: 56 GB
- 1/8 of activations/gradients
- **Total: ~84 GB**

Communication:
- All-reduce gradients: 14 GB per GPU
- Simple, efficient

**ZeRO-3:**

Memory per GPU:
- 1/8 of model: 1.75 GB
- 1/8 of optimizer: 7 GB
- 1/8 of gradients: 1.75 GB
- **Total: ~12-15 GB** (5-7× less!)

Communication:
- All-gather parameters before each layer
- Reduce-scatter gradients
- Total: ~28 GB per GPU (2× DP)

**Trade-off:**
- ZeRO-3: Less memory, more communication
- DP: More memory, less communication

**When to use:**
- **DP:** Model fits on single GPU, want max throughput
- **ZeRO-3:** Model doesn't fit, willing to trade speed for capacity

---

### Q6: Design a memory budget for serving a 70B model at 1000 QPS.

**Answer:**

**Model:** 70B parameters (140 GB in BF16)

**Throughput analysis:**
- QPS: 1000 requests/sec
- Average latency: ~6 seconds (input processing + generation)
- Concurrent requests: 1000 × 6 = 6000 ⚠️

**This is unrealistic.** Must optimize:

**1. Use Grouped-Query Attention:**
- Reduce KV heads 80 → 8 (10× reduction)

**2. Realistic concurrent batching:**
- vLLM continuous batching
- Actual concurrent: ~500-1000 (not 6000)

**3. Average active tokens:**
- Not all requests at max length
- Effective: ~300 tokens per request

**KV cache estimate:**
- 1000 concurrent × 300 tokens × 0.26 MB/token = **78 GB**

**System design (8× H100 80GB):**

| Component | Memory per GPU |
|-----------|---------------|
| Model (TP-8) | 140 GB / 8 = 17.5 GB |
| KV cache | 78 GB / 8 = 9.75 GB |
| Workspace | 2 GB |
| **Total** | **29.25 GB** ✅ |

**Architecture:**
- 8× H100 80GB (tensor parallelism)
- vLLM with PagedAttention
- 70B model with GQA
- Expected: 800-1200 QPS sustained

---

## 8. Summary

**Key takeaways:**

### Memory Hierarchy
- GPU HBM: ~1.6 TB/s, 40-80 GB capacity (keep hot data here)
- CPU RAM: ~80 GB/s, 64GB-2TB capacity (offload cold data)
- PCIe: ~32 GB/s (minimize transfers)

### Training vs Inference
- Training: 3-5× more memory (gradients + optimizer states dominate)
- Inference: KV cache dominates for long contexts
- Can infer models that won't fit for training

### Compute vs Memory Bound
- Most Transformer ops are memory-bound (attention, LayerNorm, embeddings)
- Only large matrix multiplications are compute-bound
- Optimize for memory bandwidth, not just FLOPs

### Key Optimization Techniques
- **Gradient checkpointing:** 80-90% activation memory reduction
- **8-bit optimizers:** 75% optimizer state reduction
- **LoRA:** 100-1000× trainable parameter reduction
- **ZeRO-3:** 5-7× memory reduction in multi-GPU training
- **GQA/MQA:** 4-32× KV cache reduction
- **PagedAttention:** 10-30% better KV cache utilization

### Design Principles
1. Keep hot data on GPU (avoid CPU-GPU transfers)
2. Use gradient checkpointing for memory-constrained training
3. Consider LoRA/PEFT before full fine-tuning
4. For serving: use vLLM, GQA models, and batching
5. For multi-GPU: ZeRO-3 if memory-limited, DP if throughput-critical

---

## 9. Further Reading

- **ZeRO:** [Memory Optimizations Toward Training Trillion Parameter Models (2019)](https://arxiv.org/abs/1910.02054)
- **Flash Attention:** [Fast and Memory-Efficient Exact Attention (2022)](https://arxiv.org/abs/2205.14135)
- **PagedAttention:** [Efficient Memory Management for LLM Serving (2023)](https://arxiv.org/abs/2309.06180)
- **GQA:** [Training Generalized Multi-Query Transformer Models (2023)](https://arxiv.org/abs/2305.13245)
