# ⚙️ Block-wise k-bit Quantization

---

### 1. Overview

**Block-wise k-bit quantization** is a technique that compresses model weights into low-bit representations (e.g., 4-bit, 8-bit) while preserving performance and minimizing quantization error.
Instead of quantizing each value independently, **block-wise quantization** divides the weight matrix into smaller **blocks (chunks)** and performs quantization relative to **local statistics** (like scale and zero-point) of each block.

This local normalization significantly reduces quantization error caused by **outlier values** — a common issue in transformer weights.

---

### 2. Motivation

#### 🧠 Problem: Outliers in Weight Distributions

Weights in large models (especially in attention layers) often follow **heavy-tailed distributions** — a few large values coexist with many small ones.  
In **global quantization**, a single scale \( s_{\text{global}} = \frac{\max(|W|)}{2^{k-1}-1} \) is used for all weights.  
Large outliers force the scale up, making most small weights collapse to zero after quantization.

---

!!! example 
    Consider:
    $$
    W = [0.01, 0.02, -0.03, 0.05, 3.0]
    $$
    
    With **4-bit global quantization**:
    $$
    s_{\text{global}} = \frac{3.0}{7} \approx 0.43
    $$
    
    Quantized weights → `[0, 0, 0, 0, 7]` — almost all small weights vanish due to the single large outlier.

---

#### ✅ Solution: Block-wise Quantization
Split weights into small **blocks** (e.g., 64–256 values each), and compute a separate scale per block:
$$
s_b = \frac{\max(|W_b|)}{2^{k-1}-1}
$$
Each block adapts to its local range, preserving fine details while still compressing efficiently.

By partitioning weights into **blocks** and computing **scale/offset per block**, quantization adapts to local statistics and better preserves precision.

---

### 3. Mathematical Formulation

!!! example "📘 Steps for block quantization"
    
    Let:

    * $W \in \mathbb{R}^{d \times k}$: full-precision weight matrix
    * $B_i \subset W$: the i-th block of size $n_b$
    * $k$: number of bits used for quantization (e.g., 4 or 8)

    #### Step 1: Compute Local Scale and Zero-Point

    For each block $B_i$:

    $$
    s_i = \frac{\max(B_i) - \min(B_i)}{2^k - 1}
    $$
    
    $$
    z_i = \text{round}\left(-\frac{\min(B_i)}{s_i}\right)
    $$

    Where:

    * $s_i$: scale factor for block $i$
    * $z_i$: zero-point (offset)

    ---

    #### Step 2: Quantization

    Quantized integer representation:

    $$
    q_i = \text{clip}\left(\text{round}\left(\frac{B_i}{s_i}\right) + z_i, 0, 2^k - 1\right)
    $$
    
    #### Step 3: Dequantization (Reconstruction)
    
    $$
    \hat{B_i} = s_i \times (q_i - z_i)
    $$

    The final reconstructed weight matrix:

    $$
    \hat{W} = \bigcup_i \hat{B_i}
    $$
    
    ---

### 4. Double Quantization

**Double quantization** is a secondary compression layer designed to reduce the overhead of storing multiple block-wise scales. Instead of storing each block’s scaling factor \( s_j \) as a 16-bit or 32-bit float, **these scale values themselves are quantized** into a lower precision representation (e.g., 8-bit or 4-bit).

!!! example "📘 Details"

    #### 4.1. Concept
    If there are $N$ blocks, each with a scale $s_j$:
    
    $$
    \tilde{s_j} = \text{quantize}(s_j, s_{\text{meta}}, q_{\min}, q_{\max})
    $$
    
    Here, $s_{\text{meta}}$ is a higher-level scale shared across a group of block-scales.
    
    At dequantization:
    
    $$
    s_j = s_{\text{meta}} \cdot \tilde{s_j}
    $$

    $$
    \hat{x_i} = s_j \cdot q_i
    $$
    
    This approach can yield **20–30% memory savings**, especially when using small block sizes where the number of stored scales is large.
    
    #### 4.2. Example
    Consider a model layer with 10,000 blocks of weights.  
    Each block has one scale \( s_i \).
    
    | **Parameter** | **Value** |
    |----------------|-----------|
    | Number of blocks | 10,000 |
    | Scale per block (FP16) | 2 bytes |
    | Memory (without double quantization) | 20 KB |
    | Quantized scale (8-bit) | 1 byte |
    | Memory (with double quantization) | 10 KB |
    
    So double quantization reduces metadata memory by **50%** with negligible degradation (typically < 0.1% accuracy loss).
    
    #### 4.3. Implementation in Bitsandbytes
    
    In **bitsandbytes 0.39+**, both **block-wise quantization** and **double quantization** are implemented jointly:

    - Each weight block is quantized in **NF4 format**.
    - Each block’s **scale value** is quantized using **8-bit quantization**.
    - The quantized scales are stored alongside the 4-bit codes.
    - Dequantization happens transparently during forward passes.
    
    This enables models like LLaMA-2 70B to be fine-tuned on **single 48GB GPUs**.
    
    ---
    #### 4.4 Key Notes
    - Double quantization is orthogonal but **complementary** to block-wise quantization.  
    - It primarily targets **metadata compression**, not model accuracy.  
    - Used in **QLoRA** to compress per-block scales efficiently.
    
    | **Aspect** | **Effect** |
    |-------------|------------|
    | **Memory Efficiency** | Up to 2× reduction in metadata storage |
    | **Accuracy Impact** | Negligible (< 0.1% degradation) |
    | **Computation Overhead** | Minimal (scales dequantized once per block) |
    | **Compatibility** | Fully supported in `bitsandbytes` & `QLoRA` stack |
    
    ---

### 5. Implementation Details (Pseudo-Code)

```python
def blockwise_quantize(weights, block_size=64, num_bits=4):
    q_blocks, scales, zeros = [], [], []
    n = len(weights)
    for i in range(0, n, block_size):
        block = weights[i:i+block_size]
        min_val, max_val = block.min(), block.max()
        scale = (max_val - min_val) / (2 ** num_bits - 1)
        zero_point = -min_val / scale
        q_block = np.round(block / scale + zero_point).clip(0, 2 ** num_bits - 1)
        q_blocks.append(q_block)
        scales.append(scale)
        zeros.append(zero_point)
    return q_blocks, scales, zeros
```

### 6. Example (4-bit Quantization)

!!! example "📘 Working example"

    Consider a block of weights:
    
    $$
    B_i = [-0.9, -0.3, 0.2, 0.5, 1.0]
    $$
    
    For $k = 4$ bits:

    * $\min = -0.9, \max = 1.0$
    * $s_i = (1.0 - (-0.9)) / 15 = 0.1267$
    * $z_i = -(-0.9) / 0.1267 = 7.1 \approx 7$

    Quantized values:

    $$
    q_i = \text{round}(B_i / s_i + z_i) = [0, 5, 9, 11, 15]
    $$
    
    Dequantized:
    
    $$
    \hat{B_i} = s_i \times (q_i - 7) = [-0.9, -0.26, 0.32, 0.51, 1.01]
    $$
    
    The reconstruction closely approximates the original block.
    
    ---

### 7. Advantages

| **Aspect**    | **Benefit**                                                     |
| ------------- | --------------------------------------------------------------- |
| Local scaling | Reduces sensitivity to outliers                                 |
| Memory        | Lower storage cost (e.g., 4-bit = 8× compression)               |
| Compute       | Enables efficient GPU matrix-multiplication with custom kernels |
| Accuracy      | Closer performance to full precision                            |

---

### 8. Hardware Implementation

* Most modern inference frameworks (e.g., **bitsandbytes**, **TensorRT**) store the **scale and zero-point per block**.
* For 4-bit quantization, typical block sizes: **32, 64, or 128**.
* Scales are stored in FP16 to balance precision and storage.

---

### 9. Visualization

A conceptual diagram of block-wise quantization:

```
┌────────────────────────────┐
│         Weight Matrix      │
│  [w₁, w₂, …, wₙ]          │
└────────────────────────────┘
          ↓ Split into Blocks
┌──────────────┬──────────────┐
│ Block 1      │ Block 2      │ ...
└──────────────┴──────────────┘
     ↓                 ↓
Compute s₁,z₁      Compute s₂,z₂
     ↓                 ↓
Quantize each block separately
     ↓                 ↓
Store q₁,s₁,z₁,...,qₙ,sₙ,zₙ
```

Each block retains its **own quantization scale and offset**, enabling more accurate low-bit representation.

---

### 10. Relationship to QLoRA

QLoRA uses **4-bit NormalFloat (NF4)** quantization with **block-wise statistics**:

* Each block (typically 64 elements) uses local mean and std for normalization.
* NF4 values are quantized into [-1, 1] with learned scales.
* This approach allows fine-tuning large LLMs on a **single GPU** without significant accuracy loss.

---

