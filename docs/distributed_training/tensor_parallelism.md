# Tensor Parallelism

## 1. Core Concept

Tensor Parallelism splits **individual layers** across devices so that a single layer's computation is distributed.

**When to use**: When individual layers are too large to fit on a single GPU

---

## 2. The Basic Idea

Consider a linear layer: **Y = XW**

Where:
- X: `[batch, hidden_in]`
- W: `[hidden_in, hidden_out]`
- Y: `[batch, hidden_out]`

**Problem**: If `hidden_out` is very large (e.g., 50,000), W doesn't fit on one GPU.

**Solution**: Split W across GPUs and compute partial results.

---

## 3. Column Parallelism (Output Dimension Split)

### How It Works

Split weight matrix **by columns**:

```
W = [W₁ | W₂ | ... | Wₖ]

GPU 0: W₁ with shape [hidden_in, hidden_out/k]
GPU 1: W₂ with shape [hidden_in, hidden_out/k]
...
```

Each GPU computes:
```
Y₁ = X · W₁  (shape: [batch, hidden_out/k])
Y₂ = X · W₂  (shape: [batch, hidden_out/k])
```

**Result**: Each Yᵢ is a **partial output** (different features).

---

### Communication Pattern

**Forward Pass**:
- All-Gather to concatenate [Y₁, Y₂, ...] → Y
- Each GPU gets the full output

**Backward Pass**:
- Gradients w.r.t. X require All-Reduce

```
Forward:  X → [Y₁, Y₂, Y₃] → All-Gather → Y
Backward: Y_grad → [X_grad₁, X_grad₂, X_grad₃] → All-Reduce → X_grad
```

---

### Use Case
- Feed-forward network (FFN) in transformers
- First linear projection in attention (Q, K, V)

---

## 4. Row Parallelism (Input Dimension Split)

### How It Works

Split weight matrix **by rows**:

```
W = [ W₁ ]
    [ W₂ ]
    [ ... ]
    [ Wₖ ]

GPU 0: W₁ with shape [hidden_in/k, hidden_out]
GPU 1: W₂ with shape [hidden_in/k, hidden_out]
```

Input X is also split:
```
X = [X₁ | X₂ | ... | Xₖ]
```

Each GPU computes:
```
Y₁ = X₁ · W₁  (shape: [batch, hidden_out])
Y₂ = X₂ · W₂  (shape: [batch, hidden_out])
```

**Result**: Each Yᵢ is a **partial sum** for the same output features.

---

### Communication Pattern

**Forward Pass**:
- All-Reduce to sum [Y₁, Y₂, ...] → Y
- Each GPU gets the same final output

**Backward Pass**:
- Gradient flow mirrors forward communication

```
Forward:  [X₁, X₂] → [Y₁, Y₂] → All-Reduce → Y
Backward: Y_grad → All-Reduce → [X₁_grad, X₂_grad]
```

---

### Use Case
- Output projection in attention
- Second FFN layer in transformers

---

## 5. Why Both Exist: Complementary Design

Column and row parallelism are **complementary** and minimize total communication:

```
Transformer Block:

Input
  ↓
[Column Parallel] QKV projection
  ↓ All-Gather
Attention
  ↓
[Row Parallel] Output projection
  ↓ All-Reduce
[Column Parallel] FFN Layer 1
  ↓ All-Gather
Activation
  ↓
[Row Parallel] FFN Layer 2
  ↓ All-Reduce
Output
```

**Key Insight**: Alternating column/row avoids redundant communication and balances memory.

---

## 6. Megatron-LM Style TP

Megatron-LM popularized this pattern for transformers:

### Self-Attention
```python
# Column parallel (split attention heads)
Q, K, V = column_parallel_linear(X)  # All-Gather output

# Attention computation (no communication)
attention_output = self_attention(Q, K, V)

# Row parallel (reduce across heads)
output = row_parallel_linear(attention_output)  # All-Reduce output
```

---

### Feed-Forward Network
```python
# Column parallel
hidden = column_parallel_linear(X)  # All-Gather
hidden = gelu(hidden)

# Row parallel
output = row_parallel_linear(hidden)  # All-Reduce
```

---

## 7. Communication Cost

For a single linear layer with Ψ parameters:

**Column Parallelism**:
- Forward: All-Gather → `2Ψ/k` communicated per GPU
- Backward: All-Reduce → `2Ψ/k` communicated per GPU

**Row Parallelism**:
- Forward: All-Reduce → `2Ψ/k` communicated per GPU
- Backward: Similar

**Total per layer**: ~`4Ψ/k` per GPU

> **Critical**: Communication happens **inside forward/backward pass**, making TP very latency-sensitive.

---

## 8. Memory Savings

For model with Ψ parameters across k GPUs:

| Component | Per GPU |
|-----------|---------|
| **Parameters** | Ψ/k |
| **Gradients** | Ψ/k |
| **Optimizer States** | 12Ψ/k |
| **Activations** | Depends (also reduced due to smaller intermediate tensors) |

**Example**: 7B model with 8-way TP
- Parameters: 14GB / 8 = 1.75GB per GPU
- Optimizer: 84GB / 8 = 10.5GB per GPU

---

## 10. Sequence Parallelism Extension

**Problem**: Even with TP, activation memory from long sequences is huge.

**Solution**: Also split the **sequence dimension**.

```
Standard TP:
Each GPU: [batch, full_sequence, hidden/k]

Sequence Parallelism:
Each GPU: [batch, sequence/k, hidden/k]
```

**Communication**:
- All-Gather for operations that need full sequence (attention)
- All-Reduce for operations that can work on shards

**Use case**: Very long context models (32k+, 128k+ tokens)

---

## 11. TP + Other Parallelisms

### TP + DP (Most Common)
```
┌────────────────┐  ┌────────────────┐
│  TP Group 0    │  │  TP Group 1    │  ← Data Parallel
│  [GPU 0-7]     │  │  [GPU 8-15]    │     dimension
└────────────────┘  └────────────────┘
     ↑ 8-way TP         ↑ 8-way TP
```

### TP + PP
- TP within each pipeline stage
- Reduces per-stage memory further

### TP + ZeRO
- TP for model parallelism
- ZeRO for optimizer state sharding
- Best of both worlds

---

## 12. Implementation Tips

### 1. Communication Backend
```python
# Use NCCL for GPU communication
torch.distributed.init_process_group(backend='nccl')
```

### 2. Tensor Parallel Group Setup
```python
# Create TP groups
world_size = 8
tp_size = 4  # 4-way TP, 2-way DP

for i in range(world_size // tp_size):
    ranks = list(range(i * tp_size, (i + 1) * tp_size))
    group = torch.distributed.new_group(ranks)
```

### 3. Column Parallel Linear
```python
class ColumnParallelLinear(nn.Module):
    def forward(self, x):
        # Each GPU has W[:, start:end]
        output_parallel = F.linear(x, self.weight, self.bias)
        # All-Gather to get full output
        output = all_gather(output_parallel, self.tp_group)
        return output
```

---

## 13. Key Takeaways

1. **TP splits individual layers**, not batches (that's DP)
2. **Column/row parallelism are complementary** - alternate to minimize communication
3. **Communication happens inside fwd/bwd** - very latency sensitive
4. **Typically 8-way TP within a node** using NVLink
5. **Megatron-LM pattern** is the standard for transformer TP
6. **Combine with DP** for full cluster scaling
7. **Sequence parallelism** extends TP for very long contexts

---