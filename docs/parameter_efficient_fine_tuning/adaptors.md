## 1. Overview

**Parameter-Efficient Fine-Tuning (PEFT)** is a family of techniques that adapt a pre-trained model to new tasks by training only a **small subset of parameters** rather than the full model.

### Why PEFT?

Full fine-tuning of a 7B model requires:
- Weights: 14 GB
- Gradients: 14 GB
- Optimizer states: 56 GB
- **Total: ~84 GB** (before activations)

PEFT reduces trainable parameters from billions to millions while preserving most of the pre-trained model's quality.

### Core Idea

```
Pre-trained model weights (frozen)
         +
Small trainable adapter modules
         =
Task-specific model
```

---

---

## 2. PEFT Techniques Taxonomy

```
PEFT Methods
│
├── Adapter-based
│   ├── Series Adapters (Houlsby et al.)
│   ├── Parallel Adapters
│   └── AdapterFusion
│
├── Low-Rank Decomposition
│   ├── LoRA
│   ├── QLoRA
│   ├── LoRA+ / LoRA-FA
│   └── DoRA
│
├── Prefix / Prompt-based
│   ├── Prefix Tuning
│   ├── Prompt Tuning
│   └── P-Tuning v2
│
└── Soft Masking / Selective
    ├── BitFit
    └── (IA)³
```

---

---

## 3. Adapters

### 3.1 What Are Adapters?

**Adapters** are small neural network modules **inserted between layers** of a pre-trained model. Only adapter parameters are trained; the base model is frozen.

**Original Adapter architecture (Houlsby et al., 2019):**

```
Input
  │
  ▼
Self-Attention (frozen)
  │
  ▼
[Adapter] ←── trainable
  │
  ▼
LayerNorm (frozen)
  │
  ▼
Feed-Forward (frozen)
  │
  ▼
[Adapter] ←── trainable
  │
  ▼
LayerNorm (frozen)
  │
  ▼
Output
```

---

### 3.2 Adapter Architecture

Each adapter module contains:
1. **Down-projection:** $d \rightarrow r$ (compress to bottleneck)
2. **Non-linearity:** ReLU or GELU
3. **Up-projection:** $r \rightarrow d$ (expand back)
4. **Residual connection:** Add input to output

$$
\text{Adapter}(h) = h + W_{\text{up}} \cdot \sigma(W_{\text{down}} \cdot h)
$$

Where:
- $h \in \mathbb{R}^d$: input hidden state
- $W_{\text{down}} \in \mathbb{R}^{r \times d}$: down-projection
- $W_{\text{up}} \in \mathbb{R}^{d \times r}$: up-projection
- $r \ll d$: bottleneck dimension (e.g., $r = 64$, $d = 4096$)
- $\sigma$: non-linearity

**Initialization:** $W_{\text{up}}$ initialized to zero → adapter starts as identity (no perturbation).

---

### 3.3 Parameter Count

For a Transformer with $L$ layers, hidden dim $d$, bottleneck $r$:

$$
N_{\text{adapter}} = 2 \times L \times 2 \times (d \times r + r \times d) = 8Ldr
$$

Factor of 2: two adapters per layer (after attention and after FFN).

**Example (7B LLaMA-2, $L=32$, $d=4096$, $r=64$):**

$$
N_{\text{adapter}} = 8 \times 32 \times 4096 \times 64 = 67M \text{ params}
$$

**67M / 7B = ~1%** of total parameters.

---

### 3.4 Adapter Variants

| Variant | Where Inserted | Key Difference |
|---------|---------------|----------------|
| **Houlsby** | After attention + after FFN | Original, 2 adapters per layer |
| **Pfeiffer** | After FFN only | 1 adapter per layer, similar performance |
| **Parallel** | Alongside layers (not series) | Faster inference, no latency |
| **AdapterFusion** | Between adapters | Combines multiple task adapters |

**Parallel Adapter:**

```
Input
  ├──→ Self-Attention (frozen) ──→ +──→
  └──→ [Adapter] (trainable)   ──→ ↑
                                    Output
```

**Advantage:** No additional latency in theory (computed in parallel with attention).

---

---

## 4. LoRA vs Adapters

| Aspect | Adapters | LoRA |
|--------|----------|------|
| **Where** | Inserted between layers | Parallel to existing weights |
| **Inference latency** | Added (sequential) | None (can merge weights) |
| **Architecture change** | Yes (new layers) | No (same structure) |
| **Parameters** | ~1-2% | ~0.1-1% |
| **Typical use** | NLP tasks | LLM fine-tuning |

**LoRA's key advantage: Zero inference overhead**

After training, merge $\Delta W$ into original weights:

$$
W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A
$$

The merged model has the **same architecture and speed** as the original.

---

---

## 5. Prefix Tuning and Prompt Tuning

### 5.1 Prefix Tuning

**Concept:** Prepend trainable "prefix" tokens to the **keys and values** of every attention layer.

```
Standard attention:
  Attend over: [token₁, token₂, ..., tokenₙ]

Prefix attention:
  Attend over: [prefix₁, ..., prefixₖ, token₁, ..., tokenₙ]
  (prefix tokens are trainable, input tokens are frozen)
```

**Memory:** Prefixes stored per layer → $L \times k \times 2d$ parameters.

**Characteristics:**
- No architecture changes to base model
- Different from LoRA: influences attention differently
- Works well for generation tasks (summarization, translation)
- Less popular today (LoRA outperforms in most benchmarks)

---

### 5.2 Prompt Tuning

**Concept:** Prepend trainable soft tokens to the **input embedding** only (not every layer).

```
Standard input: [token₁, ..., tokenₙ]
Prompted input: [soft₁, ..., softₖ, token₁, ..., tokenₙ]
```

**Key difference from prefix tuning:** Only input layer modified (not all layers).

**Characteristics:**
- Very few parameters ($k \times d$ total)
- Only competitive with full fine-tuning at 10B+ scale
- Simple to implement
- Lower performance than LoRA for smaller models

---

### 5.3 BitFit

**Concept:** Only train **bias terms** of the model.

- Trainable params: ~0.1% of model
- No architecture changes
- Surprisingly effective for classification tasks
- Not competitive for generation/instruction following

---

---

## 6. (IA)³

**Concept:** Learn to **rescale** activations with learned vectors.

For attention:
$$
\text{Attention} = \text{softmax}\left(\frac{(l_k \odot K)(l_q \odot Q)^T}{\sqrt{d_k}}\right)(l_v \odot V)
$$

Where $l_k, l_q, l_v$ are **learned scaling vectors**.

**Characteristics:**
- ~0.01% trainable params (10× less than LoRA)
- Good for few-shot and continual learning scenarios
- Less capacity than LoRA for complex tasks
- Fast training (very few parameters)

---

---

## 7. Choosing the Right PEFT Method

### 7.1 Decision Matrix

| Scenario | Recommended Method | Why |
|----------|-------------------|-----|
| LLM instruction tuning | **LoRA (r=16-64)** | Best quality/cost balance |
| Consumer GPU (<24GB) | **QLoRA** | Memory efficiency |
| Multiple task adapters | **AdapterFusion** | Combine task knowledge |
| Minimal params | **(IA)³** | Fewest parameters |
| Translation/summarization | **Prefix Tuning** | Strong for seq2seq |
| Zero inference overhead needed | **LoRA** (merge) | Can merge into base |
| Production deployment | **LoRA** (merged) | Same speed as base model |
| Research / exploration | **DoRA or LoRA+** | State-of-the-art quality |

---

### 7.2 Quality vs Efficiency Trade-off

```
Quality
  │
  │  Full Fine-tuning ●
  │             DoRA ●
  │            LoRA+ ●
  │             LoRA ●
  │        Adapters ●
  │   Prefix Tuning ●
  │   Prompt Tuning ●
  │          BitFit ●
  │           (IA)³ ●
  └─────────────────────── Memory / Compute
     Low                     High
```

---

### 7.3 Rank Selection Guide

```
Task complexity → Rank choice:

Classification (few classes)         r = 4
Named entity recognition             r = 8
Sentiment, summarization             r = 8-16
Instruction following (general)      r = 16-32
Complex reasoning, coding            r = 32-64
Full model behavior change           r = 64-128
```

---

---

## 8. PEFT in Production

### 8.1 Serving Multiple Adapters

**Problem:** Serving 100 different LoRA adapters would require 100 model copies.

**Solution:** Base model + hot-swap adapters.

```python
# Load one base model
model = AutoModelForCausalLM.from_pretrained("base_model")

# Load multiple adapters
model.load_adapter("./adapter_customer_service", adapter_name="cs")
model.load_adapter("./adapter_code", adapter_name="code")
model.load_adapter("./adapter_medical", adapter_name="med")

# Switch adapters at inference time
model.set_adapter("cs")   # Customer service mode
output = model.generate(...)

model.set_adapter("code") # Code mode
output = model.generate(...)
```

**Memory:** Base model (14 GB) + N adapters (50 MB each) vs N full models (N × 14 GB).

---

### 8.2 LoRA Merging Strategies

**Merge into base model (zero overhead):**

```python
merged = model.merge_and_unload()  # Single adapter, no overhead
```

**TIES merging (multiple adapters):**

Combine multiple LoRA adapters by:
1. Trim small values (sparsify)
2. Elect sign based on majority vote
3. Disjoint merge (no conflicts)

```python
# Merge customer service + code adapters
from peft import PeftModel

merged = PeftModel.from_pretrained(base_model, "adapter_cs")
merged = merged.merge_and_unload()
# Apply second adapter...
```

---

### 8.3 Multi-task Learning with Adapters

**Adapter for each task, shared base:**

```
Base Model (frozen)
      │
  ┌───┴───┬───────┬───────┐
  ↓       ↓       ↓       ↓
[NER]  [QA]  [Summ] [Class]
Adapter Adapter Adapter Adapter
```

**AdapterFusion:** Learn to combine multiple task adapters:

```python
from adapters import AdapterConfig, AdapterFusionConfig

# Load trained task adapters
model.load_adapter("./ner_adapter", config="pfeiffer")
model.load_adapter("./qa_adapter", config="pfeiffer")

# Learn fusion weights
fusion_config = AdapterFusionConfig.load("dynamic")
model.add_adapter_fusion(["ner", "qa"], fusion_config)
```

---

---

## 9. Interview Questions

### Q1: What is the fundamental difference between LoRA and traditional adapters?

**Answer:**

Both train small modules while freezing the base model, but differ in key ways:

**Traditional Adapters:**
- Inserted **sequentially** between layers
- Add new computational path in the forward pass
- **Inference latency:** Always increased (new layers in path)
- Architecture: `Input → Attention → Adapter → FFN → Adapter → Output`

**LoRA:**
- Applied **parallel** to existing weight matrices
- Modifies the weight update, not the architecture
- **Inference latency:** Zero (weights can be merged post-training)
- Architecture: Same as original model after merging

**Memory comparison (7B model):**
- Adapters: ~67M params (2 per layer, $r=64$)
- LoRA: ~4M params (q, v projections, $r=16$)

**When to prefer adapters:**
- Multi-task serving (swap without merging)
- When inference latency is tolerable
- AdapterFusion scenarios

**When to prefer LoRA:**
- Production deployment (merge = no overhead)
- Lower parameter count needed
- Most LLM fine-tuning scenarios today

---

### Q2: Explain the math behind why LoRA works. Why does low rank make sense?

**Answer:**

**Empirical observation:** The weight update matrix $\Delta W$ during fine-tuning has **low intrinsic rank**.

**What this means:**

$\Delta W \in \mathbb{R}^{d \times k}$ has rank $r^* \ll \min(d, k)$.

This means the information gain from fine-tuning lives in a low-dimensional subspace.

**Evidence from the original LoRA paper:**

- GPT-3 fine-tuned on NLU tasks
- The "intrinsic dimension" of adaptation is very low
- $r = 1$ or $r = 2$ captures most of the update for some tasks
- $r = 4$ or $r = 8$ works well for most tasks
- Increasing $r$ beyond 64 rarely helps

**Mathematical justification:**

If the true update $\Delta W = UV^T$ where $U \in \mathbb{R}^{d \times r}$, $V \in \mathbb{R}^{k \times r}$:

- Standard fine-tuning: store $d \times k$ values
- LoRA: store $d \times r + k \times r = (d+k) \times r$ values

**Compression ratio:**
$$
\frac{(d+k) \times r}{d \times k} = r \left(\frac{1}{k} + \frac{1}{d}\right) \approx \frac{2r}{d} \text{ (when } d = k\text{)}
$$

For $d = 4096$, $r = 16$: ratio = $\frac{32}{4096} = 0.78\%$

**Why randomly initialized $A$ + zero $B$ works:**

- $A$ initialized with Gaussian noise: creates a random low-rank projection
- $B$ initialized to zero: $\Delta W = BA = 0$ at start (no perturbation)
- As training proceeds, $B$ learns the structure of the update
- $\frac{\alpha}{r}$ scaling controls the effective learning rate of the LoRA path

**Follow-up:** How does rank affect the expressiveness?  
**Answer:** Higher rank → more expressiveness (can represent more complex updates) but more parameters. It's a hyperparameter controlling the capacity of the adaptation.

---

### Q3: You need to fine-tune a 13B model on a single 24GB GPU. What PEFT strategy would you use?

**Answer:**

**Memory budget:** 24 GB

**Option 1: LoRA on FP16 model**

| Component | Memory |
|-----------|--------|
| Base model (FP16) | 26 GB ❌ |

Doesn't fit.

**Option 2: QLoRA (4-bit base + LoRA)**

| Component | Memory |
|-----------|--------|
| Base model (4-bit NF4) | 6.5 GB |
| LoRA adapters (r=32, BF16) | 0.5 GB |
| Gradients (LoRA only) | 0.5 GB |
| Optimizer (8-bit Adam) | 1.0 GB |
| Activations (checkpointed) | 2.0 GB |
| **Total** | **~10.5 GB** ✅ |

**Configuration:**

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    quantization_config=bnb_config,
)

# LoRA config
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    use_dora=True,  # DoRA for better quality (2024)
)

model = get_peft_model(model, lora_config)

# Optimizer
import bitsandbytes as bnb
optimizer = bnb.optim.PagedAdamW8bit(
    model.parameters(), lr=2e-4
)
```

**Why this works:**

- 4-bit base model: 8× memory reduction for weights
- LoRA: only ~1% of params are trained → tiny gradients/optimizer states
- Gradient checkpointing: reduces activation memory
- 8-bit optimizer: further reduces optimizer state memory

**Follow-up:** What quality loss to expect?  
**Answer:** QLoRA on 13B model typically achieves 95-98% of full fine-tuning quality, depending on task complexity and LoRA rank. For most instruction-following tasks, the difference is not noticeable.

---

### Q4: Why doesn't LoRA add inference overhead, but adapters do?

**Answer:**

**Adapters add sequential computation:**

```
Forward pass with adapter:
h → Attention → h' → W_down → r → σ → W_up → Δh → h' + Δh → ...
                     (bottleneck)
```

The adapter is **in the computation path** — you can't skip it. Every token goes through the extra layers. Latency increases proportionally to adapter size.

**LoRA can be merged:**

During training:
$$
h = W_0 x + \underbrace{\frac{\alpha}{r} B A x}_{\text{LoRA path}}
$$

After training, merge:
$$
W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A
$$

This is just **one matrix multiply** — same as the original operation. LoRA effectively modifies the weights offline, leaving no extra computation at inference time.

**The merge is mathematically exact:**

```python
# Before merge (two matrix multiplies):
h = x @ W0.T + x @ (A.T @ B.T) * (alpha / r)

# After merge (one matrix multiply, same result):
W_merged = W0 + (B @ A) * (alpha / r)
h = x @ W_merged.T
```

**When you can't merge:**
- Multi-adapter serving (need to hot-swap)
- Continual learning (adapters change over time)
- In these cases, adapters may be preferable


### Q5: How does AdapterFusion combine knowledge from multiple tasks?

**Answer:**

**Problem:** Train adapters for Task A, Task B, Task C separately. Now fine-tune on Task D that could benefit from all three.

**Naive approach:** Fine-tune new adapter. Loses Task A, B, C knowledge.

**AdapterFusion:** Learn to combine trained adapters dynamically.

---

**Architecture:**

At each Transformer layer:

```
Input h
  │
  ├──→ Adapter_A(h) = a_A
  ├──→ Adapter_B(h) = a_B
  └──→ Adapter_C(h) = a_C

Fusion layer:
  scores = softmax(Q(h) · K([a_A, a_B, a_C])^T)
  output = scores · V([a_A, a_B, a_C])
```

Trained: Fusion attention weights (Q, K, V)  
Frozen: Adapter_A, Adapter_B, Adapter_C, Base model

**Training stages:**

1. **Stage 1:** Train task-specific adapters independently
2. **Stage 2:** Freeze all adapters, train only fusion attention weights on target task

**Memory at serving:**

```
Base model: 14 GB (frozen)
Adapter_A: 50 MB (frozen)
Adapter_B: 50 MB (frozen)
Adapter_C: 50 MB (frozen)
Fusion weights: 10 MB (trained)
Total: ~14.16 GB vs 4 × 14 = 56 GB for separate full models
```

**Limitation:** Only works with same base model. Can't fuse adapters from different base models.

---
