# Mid-Training

## 1. Overview

**Mid-training**, also known as **Continued Pre-Training** or the **Annealing Phase**, is a critical intermediate stage in modern LLM development that bridges general pre-training and task-specific alignment (SFT/RLHF).

**Core Purpose:** Transform a general-purpose foundation model into a high-reasoning, domain-specialized system while minimizing catastrophic forgetting.

**Key Benefits:**

- Domain specialization at scale
- Enhanced reasoning capabilities
- Reduced downstream alignment complexity
- 2-5× reduction in SFT data requirements

---

## 2. Strategic Value

Mid-training has become essential for frontier models, serving three primary functions:

### 1. Domain Deep-Diving
Injection of large-scale, curated domain corpora (law, biomedical, mathematics, code) that would dilute general pre-training but enable deep specialization.

### 2. Reasoning Architecture
Transition from pattern matching to structured reasoning through exposure to:

- Synthetic reasoning trajectories
- Mathematical proofs
- Multi-step problem-solving traces

### 3. Capability Extensions

- Long-context reasoning
- Tool usage priors
- Agentic behaviors

---

## 3. Resource Requirements

| Resource | Requirement | Notes |
|----------|-------------|-------|
| **Compute** | 5-15% of pre-training FLOPs | H100/B200 GPU clusters required |
| **Optimizer States** | Full Adam/AdamW moments | Critical for stability |
| **Data Quality** | >95% signal-to-noise ratio | Aggressive filtering needed |
| **Replay Buffer** | 10-20% general data | Prevents catastrophic forgetting |

---

## 4. Training Configuration

### Data Mixture (Typical 2026 Recipe)

- **40%** Specialist corpora (textbooks, technical manuals, papers)
- **30%** Synthetic reasoning (teacher model trajectories)
- **20%** High-quality web data (e.g., FineWeb-Edu)
- **10%** Long-form documents (books, full codebases)

---

### Learning Rate Schedule

Mid-training uses a **Cool-down/Annealing** strategy:

1. **Re-warmup:** Short stabilization on new data distribution
2. **Cosine Decay:** Deep decay toward zero for weight settling

**Critical:** Full optimizer state restoration is mandatory—partial resets often cause irrecoverable divergence.

---

### Training Objectives

- **Primary:** Causal language modeling (same as pre-training)
- **Auxiliary (optional, lightly weighted):**
  - Contrastive losses for retrieval
  - Outcome-conditioned losses for tool use
  - Self-consistency rewards for reasoning

---

### Context Extension (RoPE Scaling)

To enable long-context understanding, modify Rotary Positional Embeddings:

$$f(q, i, \theta) = R_{\Theta, i}^d q$$

Increase base frequency θ (base scaling) to extend interpretable token distances.

---

## Parameter Efficiency Strategies

### Selective Training Options

- **Frozen components:** Token embeddings, early layers, normalization stats
- **Progressive unfreezing:** Gradually activate higher layers
- **Adapter-based:** Train low-rank adapters, merge later

**Benefits:** Reduced forgetting, improved stability, lower compute cost

---

## 5. Common Failure Modes

| Failure Mode | Cause | Detection |
|--------------|-------|-----------|
| **Loss Spikes** | Optimizer state mismatch | Monitor loss curves post-resume |
| **Reasoning Overfitting** | Excessive synthetic data | Check output diversity, verbosity |
| **Semantic Drift** | Insufficient replay buffer | Test general knowledge perplexity |
| **Context Illusions** | RoPE issues without true understanding | Needle-in-haystack tasks |

---

## 6. Evaluation Strategy

### Online Metrics (During Training)

- Replay buffer perplexity
- Long-context loss by position
- Reasoning trace self-consistency

### Offline Probes (Checkpoints)

- Needle-in-haystack retrieval
- Math/code reasoning benchmarks
- Tool-use simulation accuracy
- General knowledge retention tests

**Why it matters:** High rollback costs make early regression detection critical.

---

## 7. Impact on Downstream Stages

A well-executed mid-training phase:

- ✅ Reduces SFT data needs by 2-5×
- ✅ Improves RLHF stability
- ✅ Lowers reward hacking risk
- ✅ Pre-internalizes reasoning norms

---

## 8. Recent Advances (2025-2026)

### Agentic Synthesis
Including tool-use action logs during mid-training improves agentic performance more effectively than SFT alone.

### Internalized RL
Applying RL during mid-training to reward correct reasoning paths in math and code domains.

### Dynamic Curriculum Mixing
Using a "proctor" model to adjust data mixture in real-time based on primary model's loss patterns.

---

## 9. Training Phase Comparison

| Aspect | Pre-Training | Mid-Training | Post-Training (SFT) |
|--------|--------------|--------------|---------------------|
| **Token Count** | 5T - 15T | 100B - 500B | 10M - 50M |
| **Focus** | Breadth | Depth + Reasoning | Behavior + Safety |
| **Data Source** | Raw web scrapes | Curated + Synthetic | Human demonstrations |
| **Compute** | ~100% | ~5-15% | <1% |
| **Forgetting Risk** | N/A | High (needs replay) | Moderate |

---
