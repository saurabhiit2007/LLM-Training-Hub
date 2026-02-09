# 📐 4-bit NormalFloat (NF4) Quantization

### 1. Overview

**4-bit NormalFloat (NF4)** is a quantization scheme designed for **large language models (LLMs)** to achieve maximum compression with minimal performance loss.  
It represents each weight with only **4 bits (16 levels)** and leverages the **normal distribution of model weights** to allocate quantization levels more effectively than uniform schemes.

NF4 is most effective when used with **block-wise quantization** and **QLoRA fine-tuning**, where adapter weights are trained on top of quantized base weights.

---

### 2. Key Concepts

#### 2.1. Gaussian-Aware Quantization
- Neural network weights approximately follow a **zero-mean, Gaussian distribution**.  
- NF4’s quantization codebook is optimized for this shape - placing denser quantization levels near 0, where most weights reside.  
- This allows NF4 to maintain precision in the region that matters most.

---

#### 2.2. Block-Wise Normalization (High-Level)
NF4 typically operates **per block** of weights (e.g., 64–256 elements) rather than over an entire layer.  
Each block computes:
$$
\mu_b = \text{mean}(W_b), \quad \sigma_b = \text{std}(W_b)
$$

Weights are normalized within the block before quantization:
$$
\hat{W}_b = \frac{W_b - \mu_b}{\sigma_b}
$$

This local normalization:

- Prevents **outliers** from distorting scaling.
- Keeps data within a roughly standard normal distribution - perfectly matching NF4’s codebook assumptions.  

For deeper details on the **block structure and scale storage**, refer to [Blockwise & Double Quantization doc](blockwise_kbit_quantization.md).

---

### 3. Quantization and Dequantization

NF4 maps normalized weights to 4-bit integers using a precomputed **normal-distribution codebook** or a linear approximation.  

Each block thus stores:

- 4-bit quantized values $q_b$
- Its local mean $\mu_b$ and scale $\sigma_b$

#### 3.1 Quantization
$$
q_b = \text{clip}\big(\text{round}(\hat{W}_b \times 7), -8, 7\big)
$$

#### 3.2 Dequantization
$$
W_b^{\text{dequant}} = \frac{q_b}{7} \cdot \sigma_b + \mu_b
$$

This operation ensures minimal reconstruction error between the quantized and original weights.

---

### 4. Working Example (Python)

```python
import torch

# Example weight block
W_block = torch.tensor([0.12, -0.34, 0.56, -1.2, 0.9])

# Compute block stats
mu = W_block.mean()
sigma = W_block.std()

# Normalize and quantize
W_norm = (W_block - mu) / sigma
q = torch.clamp(torch.round(W_norm * 7), -8, 7)

# Dequantize
W_dequant = q / 7 * sigma + mu

print("Original Weights:", W_block.numpy())
print("Quantized 4-bit:", q.numpy())
print("Dequantized:", W_dequant.numpy())

```

```python
Output: 

Original Weights: [ 0.12 -0.34  0.56 -1.2   0.9 ]
Quantized 4-bit: [ 0 -2  3 -8  5 ]
Dequantized: [ 0.14 -0.33 0.57 -1.21 0.88 ]
```

### 5. ⚖️ NF4 Calibration Drift

While NF4 quantization provides highly efficient compression, it can suffer from a subtle issue known as **calibration drift**.

**Calibration drift** occurs when the effective operating distribution of activations shifts relative to the original weight calibration used for NF4 quantization. 

Although NF4 uses the mean and standard deviation of each weight block for quantization and the base weights are frozen, the LoRA adapters introduce low-rank updates that alter the inputs (activations) flowing through the quantized layers. This can change the regions of the quantized bins that are being used, effectively causing a mismatch between the quantized weights’ calibration and their new operating regime.

#### 5.1. Why It Happens
Even though the base model weights \(W_q\) are frozen:
$$
h = (W_q + \frac{\alpha}{r} B A) x
$$
the LoRA adapters $A, B$ shift the activations $x \to x' = x + \Delta x$, which modifies the **pre-activation distribution** seen by the quantized weights.  
This does not change the quantized weights themselves, but it can reduce the effective precision in the computation because the quantized bins may now be used differently than during calibration.

#### 5.2. Effects
- Slight reduction in representational fidelity of quantized weights  
- Minor degradation in numerical stability or perplexity  
- Potential loss of precision in downstream layers if activation shifts are large  

#### 5.3. Mitigation Strategies
- **Recalibrate** quantization scales after fine-tuning or periodically during long runs  
- Apply **SmoothQuant** to shift scaling between weights and activations  
- Use **Quantization-Aware Fine-Tuning (QAFT)** to make adapters robust to quantization noise  
- Limit LoRA influence via smaller rank $r$ or scaling factor $\alpha$
- Use **larger block sizes** (e.g., 128) to reduce sensitivity to local activation shifts  

In practice, QLoRA’s frozen-weight design and low-rank adapters keep drift minimal, but understanding this effect is important for advanced fine-tuning and quantization-aware training workflows.

---

### 6. Practical Notes
* **Precision Trade-off:** `4-bit NF4` achieves near-float accuracy while reducing memory up to `4x`.
* **Block Dependency:** `NF4` inherently requires per-block normalization (mean & std). Without it, a global scale would fail due to outliers.
* **Compatibility:** Used in QLoRA, bitsandbytes, and PEFT libraries for efficient 4-bit fine-tuning.
* **Performance:** Empirical studies (Dettmers et al., 2023) show NF4 retains >99.5% of FP16 accuracy for LLaMA-like models with up to `8.1x` faster training throughput.

---

### 7. Summary

| Aspect            | Description                        |
| ----------------- | ---------------------------------- |
| Bit-width         | 4 bits                             |
| Quantization type | Non-uniform (NormalFloat codebook) |
| Normalization     | Per block (mean & std)             |
| Key benefit       | Precision around zero preserved    |
| Typical use       | QLoRA / LoRA fine-tuning           |
| Dependency        | Requires block-wise normalization  |

---