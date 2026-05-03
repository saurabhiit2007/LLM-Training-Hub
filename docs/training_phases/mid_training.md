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

### Technical Implementation

**Q4: Why is full optimizer state restoration critical in mid-training?**

**A:** Optimizer states (Adam/AdamW momentum and variance) encode the training trajectory and landscape geometry. Resetting them causes:

- Loss spikes and divergence
- Inefficient exploration of the new data distribution
- Potential inability to recover model quality

Full state restoration ensures smooth continuation from pre-training.

---

**Q5: Explain the learning rate schedule used in mid-training and why it differs from pre-training.**

**A:** Mid-training uses a **re-warmup + cosine decay** schedule:

1. **Re-warmup:** Briefly increases LR to help the model adapt to the new data distribution
2. **Cosine decay to near-zero:** Ensures weights settle deeply into specialized loss landscape valleys

This differs from pre-training's longer, more gradual decay because mid-training starts from an already-trained model.

---

**Q6: How does RoPE scaling enable long-context understanding?**

**A:** RoPE (Rotary Positional Embeddings) encodes position through rotation in embedding space. By increasing the base frequency θ (base scaling), we:

- Compress the positional encoding frequency
- Allow the model to interpret larger token distances
- Extend the effective context window beyond pre-training limits

The model then learns to reason over these extended contexts during mid-training.

---

### Design Decisions

**Q7: What are the trade-offs between full fine-tuning vs. adapter-based mid-training?**

**A:**

| Approach | Pros | Cons |
|----------|------|------|
| **Full Fine-tuning** | Maximum specialization, no architectural constraints | Higher compute, more forgetting risk |
| **Adapter-based** | Lower compute, less forgetting, modular | Potential capacity bottleneck, merge complexity |

Choice depends on available compute, forgetting tolerance, and desired specialization depth.

---

**Q8: Why include synthetic reasoning trajectories in the data mixture?**

**A:** Synthetic data from larger "teacher" models provides:

- Explicit reasoning chains (step-by-step logic)
- Higher density of correct reasoning patterns than web data
- Controlled difficulty and coverage of reasoning types
- Scalability beyond human-annotated examples

This has proven more effective for reasoning than raw data alone.

---

### Debugging & Evaluation

**Q9: Your mid-training run shows increasing perplexity on general benchmarks despite improving on domain tasks. What's happening and how do you fix it?**

**A:** This indicates **catastrophic forgetting**. Solutions:

1. Increase replay buffer proportion (from 10% to 20-30%)
2. Reduce learning rate more aggressively
3. Freeze more early layers
4. Add explicit general knowledge checkpointing
5. Mix in more diverse general data

Monitor the ratio of domain improvement to general degradation.

---

**Q10: How do you detect "context length illusions" where the model appears to handle long contexts but isn't actually reasoning over them?**

**A:** Use **needle-in-haystack** tests:

- Place specific information at various positions in long contexts
- Ask questions requiring that information
- Verify the model retrieves from all positions (especially middle)
- Check if removal of key information changes answers

Also analyze loss by token position—true long-context models show stable loss across positions.

---

### Advanced Topics

**Q11: Describe the concept of "Internalized RL" during mid-training.**

**A:** Instead of applying RL only during post-training, some recent approaches apply lightweight RL during mid-training to:

- Reward correct reasoning paths in real-time
- Shape the model's reasoning prior before behavioral alignment
- Reduce need for heavy RLHF later

This creates models that inherently prefer correct reasoning structures.

---

**Q12: You have limited compute budget. Should you extend pre-training or invest in mid-training?**

**A:** **Generally mid-training** if:

- You need domain specialization or reasoning
- Pre-training already achieved good general capabilities
- You have high-quality curated/synthetic data
- Downstream efficiency (less SFT data) matters

**Extend pre-training** if:

- General knowledge gaps remain
- You have abundant diverse web data
- Target domains are broad and varied

Mid-training offers better ROI for specialization at 5-15% of pre-training cost.

---

## Quick Reference Checklist

**Before Mid-Training:**

- [ ] Full optimizer state saved from pre-training
- [ ] Data mixture designed and validated (>95% quality)
- [ ] Replay buffer prepared (10-20% general data)
- [ ] Evaluation suite ready (general + domain tests)
- [ ] RoPE scaling strategy defined (if extending context)

**During Mid-Training:**

- [ ] Monitor replay buffer perplexity
- [ ] Track reasoning consistency metrics
- [ ] Check loss by token position
- [ ] Validate context length understanding
- [ ] Test tool-use capabilities (if applicable)

**After Mid-Training:**

- [ ] Run full eval suite (general + specialized)
- [ ] Conduct needle-in-haystack tests
- [ ] Measure SFT data efficiency gains
- [ ] Document mixture recipe and outcomes
- [ ] Save final checkpoint + optimizer state

---
