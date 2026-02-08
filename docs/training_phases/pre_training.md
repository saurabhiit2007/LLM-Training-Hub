## 1. Overview

Pre-training defines the raw capability ceiling of a Large Language Model. The architectural choices, data quality, and scaling decisions made during this phase dominate downstream performance far more than post-training alignment or prompting techniques. Most limitations observed in production are traceable to decisions made during pre-training.

---

---

## 2. Training Objectives and Scaling Laws

### Self-Supervised Learning

LLMs are trained using self-supervision, where labels are derived directly from the data. Given a sequence of tokens x = (x₁, x₂, ..., xₜ), the model learns to predict the next token $P(xₜ | x₁, ..., xₜ₋₁)$

This approach requires no human annotation, scales naturally with data size, and supports emergent behaviors like reasoning and in-context learning. Despite its simplicity, this objective implicitly captures syntax, semantics, world knowledge, and procedural patterns.

---

### Negative Log Likelihood (NLL) Objective

Training minimizes the Negative Log Likelihood, which is equivalent to minimizing cross-entropy. The loss function strongly penalizes confident incorrect predictions and encourages model calibration.

**Critical limitation:** NLL optimizes average token prediction accuracy, not task success, reasoning correctness, or truthfulness. This is why post-training alignment is necessary.

---

### Chinchilla Scaling Laws: Training Optimality

The Chinchilla results (Hoffmann et al.) demonstrated that many prior models were over-parameterized and under-trained. The core insight: for a fixed compute budget, model performance is maximized when model size and training tokens are scaled proportionally (roughly 20 tokens per parameter).

This shifted industry practice from "bigger models" to "compute-optimal training" with massive, high-quality corpora.

---

### Inference-Optimal Scaling: The LLaMA Paradigm

#### Why Chinchilla Isn't the Full Story

Chinchilla scaling laws optimize for training compute efficiency. However, they ignore a critical constraint: most LLM costs are paid after training, during inference.

In production, models are trained once but served millions of times. This fundamentally changes the optimization objective.

#### Training Cost vs Inference Cost

| Cost Type | Scales With | Frequency |
|-----------|-------------|-----------|
| Training | Parameters × Tokens | Paid once |
| Inference | Active parameters | Per request |

**Key insight:** Inference cost depends on model size, not training tokens. Chinchilla-optimal models may be cheap to train but expensive to serve.

#### Chinchilla-Optimal vs Inference-Optimal

| Chinchilla-Optimal | Inference-Optimal |
|-------------------|-------------------|
| Large models, fewer tokens | Small models, massive tokens |
| ~20 tokens per parameter | 100+ tokens per parameter |
| High inference cost | Low inference cost |
| Research-optimal | Production-optimal |

Modern open models like LLaMA 3 follow inference-optimal scaling, training smaller models on vastly more data than Chinchilla recommends. This approach maximizes quality for a fixed deployment footprint.

#### Why Over-Training Works

Smaller models are capacity-limited, not data-limited. By exposing them to vastly more data, representations become more robust, rare patterns are reinforced, and generalization improves significantly.

Even though labeled as "over-training" by Chinchilla standards, additional data continues to reduce error, improve reasoning, and make smaller models competitive with larger alternatives.

---

---

## 3. Data Pipeline and Quality

Data quality often matters more than model size. The principle "garbage in, garbage out" applies strictly to LLMs. A well-curated dataset can outperform a larger, noisier one.

### Raw Data Sources

Typical pre-training corpora combine multiple domains:

- **Web crawl data (CommonCrawl):** Provides breadth and common knowledge
- **Books and long-form text:** Improves discourse modeling and coherence
- **Code repositories (GitHub):** Enhances logical consistency and state tracking
- **Mathematical/scientific text (arXiv):** Improves symbolic manipulation and reasoning
- **Structured documents:** Adds format understanding

---

### Data Cleaning and PII Redaction

Critical cleaning steps include:

- **HTML boilerplate removal:** Strips navigation, scripts, ads, and markup so models learn meaningful content
- **Unicode normalization:** Converts equivalent characters to canonical form, reducing vocabulary fragmentation
- **Language-specific normalization:** Applies rules for lowercasing, diacritics, script consistency
- **PII redaction:** Uses regex (emails, SSNs, phone numbers), entity recognition (names → placeholders), and memorization audits to prevent private data leakage

---

### Duplicate Removal

Duplicates distort training distribution and waste compute. Removal techniques:

- **Exact hashing:** SHA-256 matching for identical documents
- **MinHash/LSH:** Locality-sensitive hashing for near-duplicates
- **Embedding similarity:** Catches semantic duplicates with different wording

Duplicates cause inflated frequency biases, artificially low validation loss, and memorization instead of abstraction.

---

### Benchmark Contamination

Leakage sources include public benchmark solutions, GitHub repositories with answers, and overlapping evaluation sets. This leads to inflated scores and misleading claims about capabilities.

Mitigation requires proactive filtering using n-gram matching against test sets and post-hoc auditing of model outputs.

---

### Data Mixing and Annealing

How data sources are combined determines model characteristics:

- **Static mixing:** Pre-assigned weights (e.g., 60% web, 20% code, 10% math)
- **Dynamic selection (DoReMi):** Continuously adjusts sampling weights using proxy model feedback, prioritizing data that improves validation loss
- **Annealing:** Curriculum learning where high-quality data (textbooks, math) is upsampled in final 5-10% of training to polish skills

---

### Synthetic Data: Opportunities and Risks

Synthetic data fills gaps and emphasizes rare skills. However, uncontrolled use risks model collapse—reinforcement of errors leading to distribution narrowing and reduced creativity.

Synthetic feedback loops can permanently damage model quality if not carefully managed.

---

---

## 4. Architecture Choices

### Decoder-Only Transformers

Most LLMs use decoder-only Transformers for causal attention alignment, simpler deployment with KV caching, and strong scaling behavior.

---

### Mixture of Experts (MoE)

MoE scales model capacity while keeping inference costs manageable. The FFN layer is split into multiple experts, with a learnable gate selecting top-k (usually 2) experts per token.

Result: huge total parameters but low active parameters per token. Trade-offs include training instability (load balancing) and high memory bandwidth requirements.

---

### Modern Component Standards

Standard Transformers (2017) are rarely used today. Modern defaults:

- **RMSNorm:** Replaces LayerNorm with simpler computation (no mean centering) and better numerical stability
- **Pre-Norm:** Normalization before attention/FFN layers improves gradient flow
- **SwiGLU:** Activation function replacing ReLU/GeLU, adds gating for better convergence
- **Bias-free layers:** Removing bias terms improves stability

---

### Tokenization

Common approaches include BPE (Byte Pair Encoding) and SentencePiece Unigram. Modern nuances:

- **Byte-fallback:** Falls back to raw bytes for unknown characters, eliminating `<UNK>` tokens (crucial for code)
- **Digit splitting:** Splitting numbers (2025 → 2, 0, 2, 5) improves arithmetic reasoning
- **Trade-off:** Larger vocabularies compress text better (faster inference) but increase embedding size and training difficulty

---

### Positional Embeddings

Modern models use Rotary Positional Embeddings (RoPE) for implicit relative positioning, better extrapolation to longer contexts, and compatibility with FlashAttention.

---

### Attention Variants

To reduce memory and compute:

- **Multi-Query Attention (MQA):** All heads share one KV head
- **Grouped-Query Attention (GQA):** Groups of heads share a KV head (standard in LLaMA), drastically reducing KV cache memory

---

### Context Length Scaling

Increasing context length impacts memory quadratically and causes training stability issues. Common strategies:

- **Long-context fine-tuning:** Pre-train on short context (4k), then anneal on long context (128k)
- **Ring Attention:** For sequences longer than single-GPU memory

---

---

## 5. Key Takeaways

1. **Scaling laws are objective-dependent:** Chinchilla optimizes training compute; inference-optimal scaling optimizes deployment cost
2. **Data quality trumps quantity:** Clean, diverse, deduplicated data outperforms larger noisy datasets
3. **Modern architectures diverge from vanilla Transformers:** RMSNorm, Pre-Norm, SwiGLU, RoPE are now standard
4. **Tokenization matters:** Byte-fallback and digit splitting significantly impact performance
5. **Pre-training decisions are hard to fix:** Most limitations observed later trace back to architecture, data, or scaling choices made here

---

---

## 6. Common Interview Questions

### Conceptual Understanding

**Q1: What is the difference between Chinchilla-optimal and inference-optimal scaling?**

Chinchilla-optimal minimizes training compute by scaling parameters and tokens proportionally (~20 tokens per parameter). Inference-optimal minimizes deployment cost by training smaller models on far more tokens (100+), accepting higher training costs for lower recurring inference costs. The key insight is that most LLM costs are paid during inference, not training.

---
**Q2: Why can't we just fine-tune a poorly pre-trained model to fix its limitations?**

Pre-training defines the capability ceiling. If a model hasn't learned fundamental patterns during pre-training (e.g., syntax, world knowledge, reasoning structures), fine-tuning cannot inject these capabilities—it can only specialize existing ones. You can't teach calculus through fine-tuning if the model never learned arithmetic during pre-training.

---

**Q3: What is model collapse and how does synthetic data cause it?**

Model collapse occurs when models are trained on data generated by other models, creating a feedback loop that amplifies errors and reduces diversity. Each generation reinforces biases and mistakes, leading to distribution narrowing and degraded outputs. It's particularly dangerous in uncontrolled synthetic data pipelines.

---

**Q4: Why do we remove duplicates from training data?**

Duplicates distort the training distribution, causing the model to overfit to repeated patterns and waste compute on redundant examples. They lead to inflated frequency biases, artificially low validation loss (the model memorizes instead of generalizing), and poor performance on novel inputs.

---

### Technical Deep Dives

**Q5: How does Mixture of Experts (MoE) reduce inference cost while maintaining capacity?**

MoE splits the FFN layer into multiple expert networks. A learned router selects top-k experts (typically k=2) for each token, activating only a fraction of total parameters. For example, a model might have 8×7B total parameters but only activate 2×7B=14B per token. This maintains high capacity (many experts) while keeping compute proportional to active parameters.

---

**Q6: What is the difference between Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)?**

MQA uses a single shared KV head for all query heads, maximizing KV cache reduction but potentially limiting expressiveness. GQA is a compromise where groups of query heads share a KV head (e.g., 8 query heads might share 2 KV heads). GQA balances memory efficiency with model quality and is the standard in models like LLaMA.

---

**Q7: Why do modern tokenizers split numbers digit-by-digit instead of keeping them as single tokens?**

Digit splitting (2025 → 2, 0, 2, 5) dramatically improves arithmetic reasoning because it allows the model to learn place-value operations compositionally. When numbers are single tokens, the model must memorize arithmetic facts for every possible number. With digit splitting, it learns generalizable operations on individual digits.

---

**Q8: What is data annealing and why is it effective?**

Data annealing is a form of curriculum learning where high-quality data (textbooks, mathematical proofs, structured reasoning) is heavily upsampled in the final 5-10% of training. This "polishes" the model by reinforcing desirable patterns after it has learned general capabilities from diverse data. It's effective because the model can leverage its broad knowledge to better absorb refined, high-signal data.

---

### System Design & Trade-offs

**Q9: You have a fixed compute budget. How do you decide between training a 70B parameter model for 1 epoch vs a 7B parameter model for 10 epochs?**

This depends on your deployment constraints. If you need low-latency inference and can't afford to serve 70B parameters, choose the 7B model trained longer (inference-optimal). If training is one-time and you can afford the inference cost, the 70B model may achieve better final performance (Chinchilla-optimal). Modern production systems heavily favor smaller models trained longer due to recurring inference costs dominating total expenditure.

---

**Q10: How would you detect and mitigate benchmark contamination in your pre-training dataset?**

Proactive approach: Use n-gram overlap detection (e.g., 13-gram matching) against all evaluation benchmarks before training, and filter out documents with high overlap. Reactive approach: After training, perform memorization audits by checking if the model can complete exact benchmark sequences. Additionally, use diverse private evaluation sets that are never released publicly to get uncontaminated performance estimates.

---

**Q11: Your model's validation loss stopped decreasing but training loss continues to drop. What's happening and what should you do?**

This is classic overfitting—the model is memorizing training data instead of learning generalizable patterns. Solutions: increase data diversity, add stronger deduplication, reduce model capacity, implement early stopping, or add regularization. In modern LLM pre-training with massive datasets, this often indicates data quality issues (too many duplicates, narrow domain coverage) rather than insufficient model size.

---

**Q12: Why do we use causal (autoregressive) masking instead of bidirectional attention like BERT during pre-training?**

Causal masking aligns with the natural generation task: predicting the next token given prior context. This creates a model that can generate coherent text. Bidirectional models like BERT use masked language modeling (predicting masked tokens with full context) which excels at understanding but cannot generate autoregressively. For general-purpose LLMs that need both understanding and generation, causal masking is preferred. BERT-style models require separate decoder components for generation tasks.

---