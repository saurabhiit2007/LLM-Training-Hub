# Supervised Fine-Tuning (SFT)

Supervised Fine-Tuning (SFT) transforms a **pre-trained base model** into a **useful assistant** that follows instructions, respects formats, and exhibits desired interaction behavior. While pre-training builds a broad world model, SFT shapes *how* that knowledge is expressed.

**Core concept:** Behavioral alignment via supervised learning.

---

## 1. Conceptual Foundation

### 1.1 What SFT Optimizes

SFT maps broad knowledge into a consistent, controllable interface.

**Conceptual shift:**
- **Pre-training:** "Continue this text"
- **SFT:** "Respond appropriately to a user instruction"

This transformation happens through supervised learning, not reinforcement learning.

### 1.2 Core Components

- **Instruction Tuning:** Defines *what* behavior you teach
- **Task Formatting:** Defines *how* that behavior is presented

---

## 2. Training Mechanics

### 2.1 Training Objective

Standard cross-entropy loss with **selective loss masking**.

Given:
- $x$ = prompt tokens (system + user)
- $y$ = assistant response tokens

SFT minimizes:

$$
\mathcal{L}_{\text{SFT}} = -\sum_{t=1}^{|y|} \log P(y_t \mid x, y_{<t})
$$

**Key:** Tokens in $x$ are excluded from the loss.

**Why masking matters:**
- Prevents prompt memorization
- Ensures gradients only optimize response generation
- Stabilizes alignment behavior

---

### 2.2 Task Formatting

Modern SFT uses **structured role-based templates** (ChatML, LLaMA-style):

```text
<|system|> You are a helpful assistant. <|end_of_text|>
<|user|> Summarize this article. <|end_of_text|>
<|assistant|> The article discusses...
```

**Why formatting is critical:**

1. **Implicit policy learning:** Role tokens act as soft behavioral constraints
2. **Gradient routing:** Loss masking + role tokens shape response behavior
3. **Inference controllability:** Enables injection of safety/tools without retraining
4. **Multi-turn state compression:** Helps model compress dialogue history into latent states

---

## 3. Data Strategy

### 3.1 Prompt Diversity as Regularization

Diversity prevents shortcut learning and maintains circuit coverage.

**Semantic diversity:**
- Math and symbolic reasoning
- Code synthesis
- Creative generation
- Factual recall
- Conversational grounding

**Structural diversity:**
- Paraphrased intents → prevents lexical memorization
- Variable verbosity → avoids length priors
- Explicit vs implicit constraints → forces instruction parsing

**Key insight:** Insufficient diversity causes models to learn *response templates* instead of *instruction semantics*.

---

### 3.2 The LIMA Hypothesis

**"Less Is More for Alignment"**

~1,000 high-quality examples can outperform tens of thousands of noisy ones.

**Implications:**
- Curation > scale
- Labeler expertise is critical
- Reduces overfitting and style bias

---

### 3.3 Typical SFT Data Mix

- Reasoning and step-by-step explanations
- Creative and stylistic writing
- Coding and math problems
- Safety and refusal examples
- Multi-turn conversations

---

## 4. Synthetic Data Generation

### 4.1 Self-Instruct

1. Start with small human-curated seed set
2. Use strong model to generate new instructions/responses
3. Filter for quality

---

### 4.2 Evol-Instruct as Curriculum Learning

Creates implicit difficulty progression:
- Base instruction
- Added constraints
- Multi-hop reasoning
- Strict formatting/safety requirements

**Improves:**
- Instruction decomposition
- Constraint satisfaction
- Planning depth

---

### 4.3 Rejection Sampling (Best-of-N)

**Process:**
1. Generate $K$ responses per prompt
2. Score using reward model or stronger reference model
3. Select best response
4. Fine-tune on selected outputs

**Benefits:**
- Sharpens instruction adherence without RL
- Reduces variance vs RLHF
- Biases toward high-reward modes

**Risks:**
- Over-optimization toward reward model
- Reduced output diversity
- Reward hacking with weak scorers

---

## 5. Training Optimizations

| Technique | Purpose | Explanation |
|-----------|---------|-------------|
| **Packing** | Throughput | Concatenates short samples into single context to avoid padding waste |
| **Loss Masking** | Correct gradients | Computes loss only on assistant tokens |
| **NEFTune** | Generalization | Adds noise to embeddings to prevent token-level overfitting |
| **Low LR** | Stability | Typical: $10^{-6}$ to $5 \times 10^{-6}$ |
| **Dropout** | Regularization | Reduces stylistic memorization |

### 5.1 Packing vs Padding

**Padding:** All sequences padded to longest length
- Wastes compute on `[PAD]` tokens
- Low token utilization
- Simple implementation

**Packing:** Multiple samples concatenated into one sequence
```
[Prompt₁ → Response₁ <EOS> Prompt₂ → Response₂ <EOS> ...]
```

**Benefits:**
- 2-3x throughput improvement
- Higher gradient signal density
- Better GPU utilization

**Implementation requirements:**
- Loss masking at sample boundaries
- Attention masking to prevent cross-example leakage
- Proper EOS handling

---

### 5.2 NEFTune (Noisy Embeddings Fine-Tuning)

**Core idea:** Inject controlled noise into embeddings during training.

**Why it works:**
- Prevents token memorization
- Improves OOD robustness
- Smooths loss landscape

**How:**
- Add Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$ to embeddings
- Keep $\sigma$ small to preserve semantics
- Noise shapes gradient updates continuously

**Recent trends (2025-2026):**
- Standard in instruction-tuning pipelines
- Layer-wise noise schedules common
- Combined with packing for long-context dialogues

---

## 6. Common Challenges

### 6.1 Catastrophic Forgetting

Model loses pre-training knowledge.

**Causes:**
- Narrow SFT domain
- High learning rates
- Full fine-tuning on small datasets

**Mitigations:**
- Mix 5-10% pre-training data
- Use LoRA/PEFT methods
- Lower learning rates

---

### 6.2 Overfitting

Model learns labeler style, not task intent.

**Symptoms:**
- Over-politeness
- Repetitive phrasing
- Template-like answers

**Mitigations:**
- Prompt diversity
- Early stopping
- NEFTune noise injection

---

### 6.3 Increased Hallucinations

Caused by knowledge contradiction between SFT and pre-training data.

**Mitigations:**
- Fact-consistent SFT data
- Retrieval-augmented generation
- Post-SFT preference optimization

---

## 7. LoRA vs Full Fine-Tuning

### LoRA (PEFT)
- ✅ Low compute cost
- ✅ Preserves base model knowledge
- ✅ Lower forgetting risk
- **Use for:** Behavior/style changes

### Full Fine-Tuning
- ✅ Handles large domain shifts
- ⚠️ Higher forgetting risk
- ⚠️ Requires careful regularization
- **Use for:** Knowledge changes

**Rule of thumb:** Behavior change → LoRA | Knowledge change → Full fine-tuning

---

## 8. SFT vs Pre-training

| Aspect | Pre-training | Supervised Fine-Tuning |
|--------|--------------|------------------------|
| **Objective** | World modeling | Behavior alignment |
| **Data scale** | Trillions of tokens | 10k-100k samples |
| **Loss** | Full sequence NTP | Masked response NTP |
| **Compute** | Massive | Moderate |
| **Primary risk** | Under-training | Overfitting, forgetting |

---

### Technical Questions

**Q4: How does packing improve SFT efficiency, and what are the implementation risks?**

Packing concatenates multiple short samples into single sequences up to max context length, improving throughput 2-3x by eliminating padding waste. Implementation must ensure: (1) loss masking resets at sample boundaries, (2) attention masks prevent cross-example leakage, (3) proper EOS token handling. Without these, you get cross-contamination and training instability.

---

**Q5: What is NEFTune and why does it improve generalization?**

NEFTune adds small Gaussian noise to embeddings during training. This prevents token-level overfitting by forcing the model to learn robust semantic features rather than memorizing exact token patterns. It smooths the loss landscape, improves OOD performance, and is now standard in modern SFT pipelines.

---

**Q6: How does rejection sampling (Best-of-N) improve SFT without using RL?**

Generate K responses per prompt, score them with a reward model, keep only the best. This creates a higher-quality dataset without policy gradients. It sharpens instruction adherence and reduces variance compared to RLHF, but risks over-optimization if the reward model is weak.

---

**Q7: Explain catastrophic forgetting in SFT and how to prevent it.**

Catastrophic forgetting occurs when the model loses pre-training knowledge while learning instruction-following behavior. This happens with narrow SFT domains, high learning rates, or small datasets. Mitigate by: (1) mixing 5-10% pre-training data, (2) using LoRA instead of full fine-tuning, (3) lowering learning rates, (4) shorter training schedules.

---

### Design Questions

**Q8: You notice your SFT model is overly verbose and repetitive. What's likely wrong and how do you fix it?**

**Diagnosis:** The model is overfitting to labeler style rather than learning instruction semantics. This happens with insufficient prompt diversity or training too long.

**Solutions:**
- Increase structural diversity (paraphrasing, variable verbosity)
- Apply early stopping based on validation metrics
- Use NEFTune to inject noise
- Check if dataset has repetitive labeler patterns

---

**Q9: When would you choose LoRA over full fine-tuning for SFT?**

**Choose LoRA when:**
- Making behavioral/style changes (tone, format, refusals)
- Working with limited compute
- Want to preserve pre-training knowledge
- Need to deploy multiple adapters

**Choose full fine-tuning when:**
- Large domain shift (e.g., medical → legal)
- Need to inject significant new knowledge
- Have compute budget and careful regularization strategy

---

**Q10: Your SFT model hallucinates more than the base model. Why and how do you debug?**

**Likely causes:**
1. SFT data contradicts pre-training facts
2. Model prioritizes format compliance over correctness
3. Overfitting to confident but wrong labeler responses

**Debug steps:**
1. Sample hallucination cases and trace to training data
2. Check for fact contradictions between SFT and base model knowledge
3. Evaluate on factual QA benchmarks
4. Consider adding retrieval augmentation or fact-checking in SFT data
5. Try preference optimization (DPO) post-SFT to penalize hallucinations

---

### System Design

**Q11: Design an SFT data pipeline for a new domain (e.g., medical assistant).**

**Pipeline:**

1. **Seed data creation** (1-2k examples)
   - Domain experts write high-quality examples
   - Cover: factual QA, reasoning, safety, refusals

2. **Synthetic expansion**
   - Self-Instruct with GPT-4 for diversity
   - Evol-Instruct for difficulty progression
   - Rejection sampling for quality

3. **Quality filtering**
   - Expert review of synthetic data
   - Automated checks (factuality, safety, format)
   - Remove duplicates and low-quality samples

4. **Regularization mix**
   - 80% domain-specific
   - 10% general conversation
   - 10% pre-training continuation

5. **Iteration**
   - Train checkpoint every 500 steps
   - Evaluate on held-out domain tasks
   - Adjust data mix based on failure modes

---

**Q12: How would you diagnose and fix a model that refuses benign requests after SFT?**

**Diagnosis:**
- Over-indexed on safety/refusal examples
- Poor prompt diversity in refusal data
- Model learned conservative heuristics

**Fixes:**
1. Audit refusal training data for false positives
2. Add positive examples for borderline benign cases
3. Use contrastive pairs (safe vs unsafe versions of similar prompts)
4. Consider preference optimization to fine-tune refusal boundary
5. Check system prompt isn't triggering false refusals

---

## 10. Key Takeaways

1. **SFT is about behavior, not knowledge** – most capabilities come from pre-training
2. **Quality > Quantity** – LIMA showed 1k great examples beats 10k mediocre ones
3. **Diversity is regularization** – prevents template learning and shortcut heuristics
4. **Implementation details matter** – packing, masking, and NEFTune significantly impact results
5. **Balance is critical** – avoid forgetting pre-training knowledge while learning new behavior
6. **Synthetic data is powerful** – but requires careful quality control and diversity
7. **Monitor for alignment taxes** – hallucinations, over-refusal, and repetitive style

---
