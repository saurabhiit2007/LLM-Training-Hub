# QLoRA: Quantized Low-Rank Adaptation

---

### 1. Overview

**QLoRA** (Quantized Low-Rank Adaptation) is a parameter-efficient fine-tuning (PEFT) technique designed to adapt large pre-trained LLMs to downstream tasks without modifying all the model weights.

It achieves this by combining **4-bit quantization** (using [NormalFloat-4, or NF4](../optimization/4bit_normal_float.md)) with [Low-Rank Adaptation (LoRA)](./lora.md), enabling fine-tuning of massive models (e.g., 65B parameters) on a single 48 GB GPU - with performance close to full fine-tuning.

---

### 2. Motivation

Fine-tuning large LLMs poses major computational and memory challenges. QLoRA addresses these by:

* **Reducing Memory Footprint** - 4-bit quantization shrinks model memory up to **75%**, enabling single-GPU fine-tuning.
* **Preserving Accuracy** - NF4 quantization minimizes quantization error by modeling real weight distributions.
* **Parameter Efficiency** - Only a small number of low-rank matrices (LoRA adapters) are trained.
* **Ease of Integration** - Built atop Hugging Face PEFT, it fits easily into existing LLM fine-tuning workflows.

---

### 3. Core Concepts

#### 3.1 Quantization in QLoRA

The base model’s parameters are quantized into **4-bit NormalFloat (NF4)** values and kept frozen during fine-tuning.
NF4 uses a **normal-distribution-aware quantization** scheme that minimizes the quantization error between original FP16 weights and 4-bit representations.

> 🔗 *Detailed explanation:* [NF4 Quantization: Principles and Implementation](../optimization/4bit_normal_float.md)

In addition, QLoRA leverages **block quantization** and **double quantization** to optimize memory even further:

* **Block Quantization:** Weights are quantized in small blocks (e.g., 64 values per block) with block-specific scaling factors, balancing compression and precision.
  This reduces quantization noise compared to uniform quantization.

* **Double Quantization:** Instead of storing the full-scale values for each block, these scale values are themselves quantized (typically to 8 bits).
  This reduces memory overhead by ~0.37 bits per parameter on average.

> 🔗 *Detailed explanation:* [Block & Double Quantization in QLoRA](blockwise_kbit_quantization.md)

---

#### 3.2 Low-Rank Adaptation (LoRA)

LoRA introduces **trainable low-rank matrices** (A) and (B) into each transformer layer, approximating weight updates as:

$$
\Delta W = B A
$$

where $A \in \mathbb{R}^{r \times d}$, $B \in \mathbb{R}^{d \times r}$, and $r$ is the rank (e.g., 8–16).
The base weights $W_0$ are frozen, and only $A, B$ are trained.

The adapted output is:

$$
h = W_0 x + \frac{\alpha}{r} B (A x)
$$

where $\alpha$ is a scaling factor controlling the LoRA contribution.

---

### 4. Integrating Quantization and LoRA

QLoRA’s key innovation is the **combination of 4-bit quantization with LoRA fine-tuning**, enabling efficient adaptation without unfreezing or copying large models.

!!! example "Step-by-Step Process"

    #### Step-1. **Quantize Base Model:**
        The pretrained model weights $W_0$ are quantized once into NF4 format using `bitsandbytes`:
        $$
        W_0^{(q)} = Q_{\text{NF4}}(W_0)
        $$
    
        * Quantization uses per-block scaling and optional double quantization.
        * These quantized weights are **frozen** during training.

    #### Step-2. **Dynamic Dequantization During Forward Pass:**
        During each forward pass, QLoRA dequantizes small blocks of weights on-the-fly:
    
        * The `bnb.nn.Linear4bit` layer from `bitsandbytes` automatically dequantizes just-in-time for computation.
        * After the matrix multiplication, the dequantized block is discarded to minimize GPU memory usage.

    #### Step-3. **Trainable LoRA Adapters (Full Precision):**
    
        The LoRA adapter matrices (A) and (B) are added to target modules (e.g., query/key/value projections) and are trained in **FP16 or BF16 precision**:
        $$
        \Delta W = B A
        $$
        These are **not quantized**, since:
    
        * They constitute <1% of total parameters.
        * Quantization would harm convergence and stability.
        * Keeping them in higher precision stabilizes gradient updates.

    #### Step-4. **Combined Forward Pass:**
    
        $$
        h = (W_0^{(dq)} + \frac{\alpha}{r} B A) x
        $$
    
        * $W_0^{(dq)}$: dynamically dequantized base weights.
        * $\frac{\alpha}{r} B A$: LoRA correction term in FP16/BF16.
        * Gradients flow only through LoRA parameters.

    #### Step-5. **Backward Pass & Updates:**
    
        * Only LoRA parameters are updated during training.
        * Quantized base weights remain frozen and untouched.
        * Gradients and optimizer states are maintained in FP16/BF16 for efficiency.

    ### 🧠 Inference with QLoRA
    
    During inference, QLoRA continues to leverage **4-bit quantization** to ensure efficiency while maintaining accuracy:
    
    * The **base model weights remain quantized (NF4)**, allowing the model to run efficiently on limited GPU memory.
    * The **LoRA adapter weights** are applied in higher precision (typically **fp16** or **bf16**) to preserve the fine-tuned adaptations.
    * During the forward pass, the quantized base weights are **temporarily dequantized** for computation and combined with the adapter outputs:
    
    $$
    h = W_{q}x + \frac{\alpha}{r}B(Ax)
    $$

    where $W_q$ represents the **quantized** base model weights.

    * The **majority of computation** is performed on the quantized backbone, while the LoRA adapter adds a small high-precision correction.
    * This hybrid setup provides a balance between **memory efficiency** (from quantization) and **model fidelity** (from LoRA adapters), enabling **low-cost, high-performance inference** even on large LLMs.

---

### 5. Quantization Mechanics Summary

| **Feature**                  | **Description**                                                                        | **Benefit**                     |
| ---------------------------- | -------------------------------------------------------------------------------------- | ------------------------------- |
| **NF4**                      | NormalFloat-4 data type; 4-bit quantization optimized for normally distributed weights | Preserves accuracy              |
| **Block Quantization**       | Quantizes weights in fixed-size blocks with shared scaling                             | Reduces quantization error      |
| **Double Quantization**      | Second quantization of scale parameters                                                | Saves additional memory         |
| **Mixed Precision Training** | Adapters in fp16/bf16; base model in NF4                                               | Optimal compute/memory tradeoff |

---

### 6. Precision Summary

| **Component**            | **Quantized?**    | **Precision**          | **Trainable?** | **Notes**                         |
| ------------------------ | ----------------- | ---------------------- | -------------- | --------------------------------- |
| Base model weights (W_0) | ✅ Yes (NF4 4-bit) | Dequantized on-the-fly | ❌ No           | Frozen, quantized by bitsandbytes |
| LoRA adapters (A, B)     | ❌ No              | FP16/BF16              | ✅ Yes          | Trained normally                  |
| Gradients                | ❌ No              | FP16/BF16              | ✅ Yes          | Only for adapters                 |
| Optimizer state          | ❌ No              | FP16/BF16              | ✅ Yes          | Small memory footprint            |

---

### 7. Implementation Details

* QLoRA uses `bnb.nn.Linear4bit` to wrap quantized linear layers.
* PEFT integrates LoRA adapters directly on top of quantized layers.
* Both components are fused during forward passes.
* During inference, the quantized base and LoRA adapters can be merged for efficient deployment.

--- 

### 8. Troubleshooting Guide

| **Issue**             | **Cause**                      | **Mitigation**                                |
| --------------------- |--------------------------------| --------------------------------------------- |
| OOM / CUDA errors     | Batch too large / rank too high | Lower `r`, enable offload/checkpointing       |
| Training instability  | LR too high, quant noise       | Lower LR or α, use LoRA dropout               |
| Underfitting          | Too low rank                   | Increase `r` or apply adapters to more layers |
| Overfitting           | Too high capacity              | Reduce epochs or use dropout                  |
| Quantization mismatch | [NF4 calibration drift](4bit_normal_float.md) | Re-quantize base model, validate small batch  |

---

### 9. Comparison: LoRA vs QLoRA

| **Method**           | **Quantized Base** | **Trainable Params** | **Memory Use** | **Performance** |
| -------------------- | ------------------ | -------------------- | -------------- | --------------- |
| **Full Fine-tuning** | ❌ No               | 100%                 | 🔴 High        | ✅ High          |
| **LoRA**             | ❌ No               | < 1%                 | 🟠 Low         | ✅ High          |
| **QLoRA**            | ✅ 4-bit (NF4)      | < 1%                 | 🟢 Very Low    | ✅ Comparable    |

---

### 10. Limitations & Challenges

* Requires accurate NF4 quantization calibration.
* Sensitive to optimizer precision and scaling.
* Not ideal for large domain shifts (may need full finetuning).
* Adapter stacking requires version management.

---