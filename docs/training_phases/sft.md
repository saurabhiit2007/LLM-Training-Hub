Supervised Fine-Tuning (SFT) is the stage where a **pre-trained base model** is transformed into a **useful assistant** that follows instructions, respects formats, and exhibits desired interaction behavior. While pre-training builds a broad world model, SFT shapes *how* that knowledge is expressed.

From an interview perspective, SFT is best understood as **behavioral alignment via supervised learning**.

---

## 1. What SFT Optimizes

Map broad knowledge into a consistent, controllable interface.

**Conceptual shift**

- Pre-training: “Continue the text”
- SFT: “Respond appropriately to a user instruction”

This is achieved without reinforcement learning. The training signal is fully supervised.

Within SFT:

- **Instruction Tuning** defines what behavior you are teaching
- **Task Formatting** defines how that behavior is presented to the model

---

---

## 2. Instruction Tuning and Task Formatting

### 2.1 Instruction Tuning

Instruction tuning teaches the model to condition its output on an explicit instruction rather than implicit continuation.

**Training objective**

Standard cross-entropy loss with **selective loss masking**.

If:

- $x$ = prompt tokens (system + user)
- $y$ = assistant response tokens

Then SFT minimizes:

$$
\mathcal{L}_{\text{SFT}} = -\sum_{t=1}^{|y|} \log P(y_t \mid x, y_{<t})
$$

Tokens belonging to $x$ are excluded from the loss.

**Why masking matters**

- Prevents memorization of prompts
- Ensures gradients only optimize response generation
- Stabilizes alignment behavior

---

### 2.2 Task and Prompt Formatting

Modern SFT relies heavily on **structured role-based templates**, such as ChatML or LLaMA-style formats.

**Example**
```text
<|system|> You are a helpful assistant. <|end_of_text|>
<|user|> Summarize this article. <|end_of_text|>
<|assistant|> The article discusses...
```

**Why Formatting Matters**

- Separates intent, context, and response  
- Enables multi-turn dialogue modeling  
- Reduces ambiguity during inference  

Formatting does more than separate roles.

- **Implicit policy learning**  
  Role tokens and system messages act as soft constraints that the model internalizes as behavioral priors.

- **Gradient routing**  
  Loss masking combined with role tokens ensures gradients primarily shape response behavior rather than prompt reconstruction.

- **Inference controllability**  
  Well-designed templates allow downstream systems to inject safety, tools, or routing instructions without retraining.

- **Multi-turn state compression**  
  Structured formatting helps the model compress dialogue history into latent state representations instead of treating each turn independently.

---

---

## 3. Prompt Diversity as Regularization

Prompt diversity is best understood as a **regularization strategy**, not just data augmentation.

---

### 3.1 Semantic Diversity
Maintains coverage of distinct internal circuits:

- Symbolic reasoning and math
- Program synthesis and execution-style reasoning
- Creative and stylistic generation
- Factual recall under instruction pressure
- Conversational grounding

---

### 3.2 Structural Diversity
Reduces shortcut learning:

- Paraphrased intents prevent lexical memorization
- Variable verbosity avoids length priors
- Explicit vs implicit constraints force instruction parsing

**Key insight**  
Insufficient structural diversity causes the model to learn *response templates* instead of *instruction semantics*.

---

---

## 4. Human vs Synthetic Supervision

### 4.1 Human-Labeled Data

**Strengths**

- High factual accuracy  
- Strong alignment with human preferences  
- Better safety and tone control  

**Limitations**

- Expensive  
- Slow to scale  
- Limited coverage of edge cases  

---

### 4.2 Synthetic Supervision

Modern post-training pipelines rely heavily on synthetic data generation.

#### Self-Instruct

- Start with a small human-curated seed set  
- Use a strong model to generate new instructions and responses  
- Filter for quality  

#### Evol-Instruct as Curriculum Learning

Evol-Instruct implicitly creates a **difficulty curriculum**:

- Base instruction
- Added constraints
- Multi-hop or multi-objective reasoning
- Strict formatting or safety requirements

This improves:

- Instruction decomposition
- Constraint satisfaction
- Planning depth

---

### 4.3 Rejection Sampling (Best-of-N)

**Process**

1. Generate $K$ responses per prompt  
2. Score them using:
     - A reward model, or  
     - A stronger reference model  
3. Select the best response  
4. Fine-tune on the selected outputs  

**Key properties**

- Sharpens instruction adherence without policy gradients
- Reduces variance compared to RLHF
- Biases the model toward high-reward modes

**Advanced risk**

- Over-optimization toward the reward model
- Reduced output diversity
- Reward hacking if the scorer is weak

---

---

## 5. Data Quality over Quantity

### 5.1 The LIMA Hypothesis

**“Less Is More for Alignment”**

**Key insight:**  

~1,000 extremely high-quality examples can outperform tens of thousands of noisy ones  

**Implications:**  

- Careful curation matters more than scale  
- Labeler expertise is critical  
- Reduces overfitting and style bias  

---

### 5.2 Typical SFT Data Mix

A strong SFT dataset often includes:

- Reasoning and step-by-step explanations  
- Creative and stylistic writing  
- Coding and math problems  
- Safety and refusal examples  
- Multi-turn conversations  

---

---

## 6. Training Optimizations and Stability

| Technique       | Purpose         | Explanation                                                                             |
|-----------------|----------------|-----------------------------------------------------------------------------------------|
| Packing         | Throughput     | Concatenates multiple short samples into a single context window to avoid padding waste |
| Loss Masking    | Correct gradients | Computes loss only on assistant tokens                                                  |
| NEFTune         | Generalization | Adds noise to embeddings during SFT to prevent token-level overfitting                  |
| Low learning rate | Stability     | Typical values are $1e^{-6}$ to $5e^{-6}$                                               |
| Dropout         | Regularization | Reduces stylistic memorization                                                          |

---

### 6.1 Packing vs Padding

In Supervised Fine-Tuning, training samples vary widely in length. How these samples are batched has a **direct impact on compute efficiency, gradient quality, and training stability**.

---

#### Padding

All sequences in a batch are padded to the length of the longest sequence using `[PAD]` tokens.

**Example**

- `Seq 1: [x x x x x x]`
- `Seq 2: [x x x PAD PAD PAD]`
- `Seq 3: [x x x x PAD PAD]`


**Why it is inefficient**

- Attention, feedforward layers, and layer norms still execute on padded tokens
- Memory bandwidth and FLOPs are wasted on tokens that contribute no gradient
- Effective tokens per batch can drop sharply when length variance is high

**Impact at scale**

- Lowers tokens processed per second
- Increases training cost
- Reduces gradient signal density

---

#### Packing

Multiple short samples are concatenated into a single long sequence up to the model’s maximum context length. Each sample is separated by an EOS or special boundary token, and loss masking prevents cross-sample leakage.

**Example**

`[Prompt₁ → Response₁ <EOS> Prompt₂ → Response₂ <EOS> Prompt₃ → Response₃]`

**Why it is efficient**

- Nearly every token contributes to loss
- Attention computation is fully utilized
- Higher effective batch token count without increasing memory

**Why Packing Improves GPU Utilization**

Packing increases:

- **Arithmetic intensity** by reducing idle FLOPs
- **Token density per batch**, improving gradient signal-to-noise ratio
- **Throughput**, often by 2x to 3x in instruction-tuning workloads where samples are short

This is especially impactful for:

- Instruction datasets with short prompts
- Chat-style SFT data
- Small to medium batch sizes constrained by memory

---

**Important Implementation Details**

- **Loss masking** must reset at each sample boundary
- **Attention masking** must prevent tokens from attending across examples
- **EOS handling** is critical to avoid information leakage
- **Position indices** may need resetting depending on the architecture

**Incorrect packing can cause:**

- Cross-example contamination
- Training instability
- Spurious memorization

---

**Padding vs Packing Summary**

| Aspect | Padding | Packing |
|------|--------|--------|
| Compute efficiency | Low | High |
| Token utilization | Sparse | Dense |
| Training speed | Slow | Fast |
| Implementation complexity | Simple | Moderate |
| Risk of leakage | None | Requires care |

> **Takeaway:** Packing is not just an optimization. It changes the **effective learning dynamics** by increasing gradient density and stabilizing updates, which is why it is now standard practice in large-scale SFT pipelines.

---

## 6.2 NEFTune: Concept and Recent Insights

**NEFTune** (Noisy Embeddings Fine-Tuning) is a targeted regularization method used during supervised fine-tuning to improve generalization and stability.

### Core Idea

During SFT, small **controlled noise** is injected into the **embedding layer** (or early representation layers). The noise acts as a soft regularizer that prevents the model from over-specializing on specific training tokens or patterns.

Instead of purely minimizing loss on the fine-tuning dataset, NEFTune encourages the model to learn representations that are **robust to small perturbations**, resulting in better performance on unseen prompts and fewer hallucinations.

### Why It Works

- **Prevents token memorization**  
  Noise makes exact token sequences less predictable, forcing the model to rely on deeper semantic features instead of surface patterns.

- **Improves out-of-distribution (OOD) robustness**  
  Tuning with noise helps the model resist over-confidence on narrow fine-tuning distributions.

- **Smooths loss landscape**  
  By blurring precise embedding positions, NEFTune reduces sharp local minima that often cause overfitting.

### How It Is Applied

A typical NEFTune variant:

- Add Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$ to embeddings each forward pass
- The noise scale $\sigma$ is kept small so that semantics are preserved but spurious correlations are suppressed

During backpropagation the noise is **not removed**; it shapes gradient updates continuously.

### Recent Trends (2025–2026)

- NEFTune has emerged as a standard trick in instruction-tuning pipelines for models like LLaMA-derivatives and open-weight assistants.
- Research shows it consistently boosts generalization metrics (e.g., AlpacaEval, Ultrachat) without increasing dataset size.
- Some systems use **layer-wise noise schedules**, adding noise in early layers and reducing it in later layers to balance representation robustness with output precision.
- Combined with packing and rejection sampling, NEFTune significantly improves instruction fidelity on long-context dialogues.

### Practical Tips

- Use a conservative noise magnitude to avoid destabilizing training
- Pair NEFTune with low learning rates and dropout for maximum regularization
- Monitor validation generalization rather than training loss to tune noise hyperparameters

---

---

## 7. Overfitting and Catastrophic Forgetting

### Catastrophic Forgetting

The model loses general reasoning or factual knowledge learned during pre-training.

**Causes**

- Narrow SFT domain  
- High learning rates  
- Full fine-tuning on small datasets  

**Mitigations**

- Mix 5–10% pre-training style data  
- Use PEFT methods such as LoRA  
- Lower learning rates  
- Shorter training schedules  

---

### Overfitting

The model learns labeler-specific style rather than task intent.

**Symptoms**

- Over-politeness  
- Repetitive phrasing  
- Template-like answers  

**Mitigations**

- Prompt diversity  
- Early stopping  
- Noise injection such as NEFTune  

---

---

## 8. LoRA vs Full Fine-Tuning

### LoRA and PEFT

- Low compute cost  
- Preserves base model knowledge  
- Lower risk of catastrophic forgetting  
- Preferred for alignment and style shifts  

### Full Fine-Tuning

- Needed for large domain shifts  
- Higher risk of forgetting  
- Requires careful regularization  

**Rule of thumb:**  

- Behavior change → LoRA  
- Knowledge change → Full fine-tuning  

---

---

## 9. Common Failure Modes After SFT

### Increased Hallucinations

Often caused by **knowledge contradiction**.

If SFT data conflicts with pre-training facts, the model may prioritize format compliance over correctness.

**Mitigations**

- Fact-consistent SFT data  
- Retrieval-augmented generation  
- Post-SFT preference optimization  

---

---

## 10. SFT vs Pre-training Summary

| Aspect         | Pre-training     | Supervised Fine-Tuning |
|----------------|-----------------|-----------------------|
| Objective      | World modeling  | Behavior alignment    |
| Data scale     | Trillions of tokens | 10k to 100k samples |
| Loss           | Full sequence NTP | Masked response NTP |
| Compute        | Massive         | Moderate             |
| Primary risk   | Under-training  | Overfitting and forgetting |

---
