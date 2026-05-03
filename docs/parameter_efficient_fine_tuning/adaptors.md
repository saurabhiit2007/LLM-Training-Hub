# Adapters and PEFT Methods

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