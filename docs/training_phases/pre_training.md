# Pre-Training

## 1. Overview

Pre-training defines the raw capability ceiling of a Large Language Model. The architectural choices, data quality, and scaling decisions made during this phase dominate downstream performance far more than post-training alignment or prompting techniques. Most limitations observed in production are traceable to decisions made during pre-training.

---

## 2. Training Objective

### Autoregressive Language Modeling

Most LLMs use an **autoregressive (causal language modeling)** objective. The joint probability of a token sequence is factorized as a product of conditional probabilities:

$$P(x_1, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})$$

This formulation requires no human annotation, scales naturally with data size, and implicitly captures syntax, semantics, world knowledge, and procedural reasoning from text alone.

---

### Negative Log-Likelihood Loss

Training minimizes the **Negative Log-Likelihood (NLL)**, equivalent to cross-entropy:

$$\mathcal{L}(\theta) = -\mathbb{E}_{x \sim D} \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})$$

Minimizing NLL is equivalent to minimizing the **KL divergence** between the data distribution $P_{data}$ and model distribution $P_\theta$:

$$H(P_{data}, P_\theta) = H(P_{data}) + D_{KL}(P_{data} \| P_\theta)$$

Since $H(P_{data})$ is fixed, training directly minimizes the divergence — connecting LLM training to information theory and lossless compression.

**Critical limitation:** NLL optimizes average token prediction accuracy, not task success, reasoning correctness, or truthfulness. This is why post-training alignment is necessary.

---

### Why Next-Token Prediction Works

To compress diverse text well, the model must internalize structure across domains:

- **Compression implies abstraction:** Predicting well across syntax, semantics, facts, and procedures requires reusable internal representations that later appear as reasoning or coding ability.
- **Softmax competition:** Increasing probability for the correct token necessarily decreases it for all alternatives, encouraging fine-grained distinctions.

---

### Teacher Forcing and Exposure Bias

During training the model is conditioned on **ground-truth previous tokens**, not its own predictions. This enables full parallelization of the loss computation across all positions in a single forward pass with stable gradients.

**Exposure bias:** At inference, the model must condition on its own (potentially wrong) predictions. Errors compound forward. This train-inference mismatch is a key motivation for post-training techniques like SFT and RLHF.

---

### Perplexity

$$\text{PPL} = \exp(\mathcal{L})$$

Perplexity converts average NLL into an interpretable scale: the model's **effective branching factor** — roughly how many tokens it is uniformly choosing between at each step. PPL = 10 means ~10 equally plausible candidates.

Perplexity measures distributional fit to the training data, not reasoning ability, factual correctness, or instruction following. Downstream task performance requires separate evaluation.

---

## 3. Scaling Laws

### Kaplan and Chinchilla

Early results (Kaplan et al., 2020) showed loss scales as a power-law in parameter count, with data treated as effectively unlimited. Later work (Hoffmann et al., 2022 — "Chinchilla") showed most prior models were under-trained relative to their size.

**Chinchilla finding:** For a fixed compute budget, roughly **20 training tokens per parameter** is compute-optimal. Smaller models trained on more data can outperform larger, under-trained ones.

This shifted practice from "bigger models" toward compute-optimal training on massive, high-quality corpora.

---

### Inference-Optimal Scaling: The LLaMA Paradigm

Chinchilla optimizes training compute. But in production, models are trained once and served millions of times — inference cost depends on model size, not training tokens.

| | Chinchilla-Optimal | Inference-Optimal |
|---|---|---|
| **Model size** | Large | Small |
| **Training tokens** | ~20× parameters | 100×+ parameters |
| **Inference cost** | High | Low |
| **Use case** | Research | Production |

Modern open models (LLaMA 3, Mistral) follow inference-optimal scaling: smaller models trained far beyond Chinchilla's recommendation. "Over-training" a smaller model makes it competitive with larger alternatives at a fraction of serving cost.

---

### Where Scaling Breaks

- **Data scarcity:** High-quality human text is finite; synthetic data risks diversity collapse and self-training loops can cause model collapse.
- **Capability saturation:** Reasoning and planning emerge discontinuously — small loss improvements can hide large behavioral differences.
- **Inference cost:** Larger models increase memory, latency, and cost, motivating efficient architectures (MoE, GQA) rather than raw scale.
- **Test-time scaling:** Recent systems (o1, DeepSeek-R1) scale inference compute rather than parameters, shifting the optimization axis from training to inference. See [Reasoning & Thinking Models](../training_techniques/thinking_llms.md).

---

## 4. Data Pipeline and Quality

Data quality often matters more than model size. A well-curated dataset can outperform a larger, noisier one.

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

Leakage sources include public benchmark solutions, GitHub repositories with answers, and overlapping evaluation sets, leading to inflated scores and misleading capability claims.

Mitigation requires proactive filtering using n-gram matching against test sets and post-hoc memorization audits.

---

### Data Mixing and Annealing

How data sources are combined determines model characteristics:

- **Static mixing:** Pre-assigned weights (e.g., 60% web, 20% code, 10% math)
- **Dynamic selection (DoReMi):** Continuously adjusts sampling weights using proxy model feedback, prioritizing data that improves validation loss
- **Annealing:** Curriculum learning where high-quality data (textbooks, math) is upsampled in the final 5–10% of training to polish skills

---

### Synthetic Data: Opportunities and Risks

Synthetic data fills gaps and emphasizes rare skills. However, uncontrolled use risks model collapse — reinforcement of errors leading to distribution narrowing and reduced creativity. Synthetic feedback loops can permanently damage model quality if not carefully managed.

---

## 5. Architecture Choices

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

## 6. Key Takeaways

1. **Scaling laws are objective-dependent:** Chinchilla optimizes training compute; inference-optimal scaling optimizes deployment cost
2. **NLL has a ceiling:** The training objective optimizes token prediction, not task success — post-training alignment is always required
3. **Data quality trumps quantity:** Clean, diverse, deduplicated data outperforms larger noisy datasets
4. **Modern architectures diverge from vanilla Transformers:** RMSNorm, Pre-Norm, SwiGLU, RoPE are now standard
5. **Tokenization matters:** Byte-fallback and digit splitting significantly impact performance
6. **Pre-training decisions are hard to fix:** Most limitations observed later trace back to architecture, data, or scaling choices made here

---
