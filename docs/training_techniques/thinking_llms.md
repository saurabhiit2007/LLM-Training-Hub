# LLM Reasoning and Thinking Models

## 1. Overview: The System 1 vs. System 2 Shift

In 2026, the AI landscape is defined by the transition from "Next-Token Predictors" to "Reasoning Agents." This is often explained using Daniel Kahneman's framework:

* **Generic LLMs (System 1):** Fast, intuitive, and associative. They respond instantly based on high-probability patterns.
* **Thinking LLMs (System 2):** Slow, deliberate, and logical. They use "Inference-Time Compute" to verify their own work before displaying it.

---

## 2. Technical Comparison Table

| Feature | Generic LLMs (GPT-4o, Llama 3) | Thinking LLMs (o1, R1, Claude 3.7) |
| :--- | :--- | :--- |
| **Primary Mechanism** | Pattern Recognition | Reinforcement Learning (RL) + Search |
| **Computation** | Constant (Fixed cost per token) | Variable (More time = More intelligence) |
| **Internal Process** | Direct Output | Hidden "Chain of Thought" (CoT) |
| **Self-Correction** | Rare (usually hallucinations) | Native (backtracks on errors) |
| **Best For** | Chat, Summaries, Creative Writing | Coding, Math, Strategic Planning |

---

## 3. Core Mechanism: Test-Time Compute Scaling

### 3.1 Inference-Time Scaling Laws

Reasoning performance scales with compute allocated at inference, not just with parameter count. This manifests as:

- More internal tokens devoted to reasoning
- Sampling and evaluating multiple candidate solutions
- Backtracking and correction loops

Empirically, reasoning benchmarks show near-log-linear gains with additional inference steps until saturation.

3.2 Compute as a Control Knob

Thinking models expose inference compute as a tunable parameter:

- Low compute approximates System 1 behavior
- High compute activates deeper reasoning loops
- This decouples model intelligence from latency constraints, enabling dynamic deployment strategies.

---

## 4. Internal Reasoning Representations

### 4.1 Hidden Chain-of-Thought

Thinking LLMs generate intermediate reasoning traces that are:

- Hidden from the user
- Used for internal verification and search
- Discarded or summarized before final output

This avoids leaking brittle or unsafe reasoning while retaining cognitive benefits.

### 4.2 Search Over Reasoning Paths

Rather than committing to a single chain, System 2 models often:

- Generate multiple reasoning candidates
- Score them using reward models
- Select or refine the best trajectory

This reframes inference as a planning problem rather than a single forward pass.

---

## 5. Reinforcement Learning for Reasoning

> RL alignment techniques (RLHF, PPO, DPO, GRPO) are covered in depth in the [LLM Alignment & Reasoning](https://github.com/saurabhiit2007/LLM-Alignment-Reasoning) repo. This section covers only the aspects specific to training thinking/reasoning models.

### 5.1 Outcome Reward Models vs Process Reward Models

 | Reward Type | What It Rewards | Limitation |
 | :--- | :--- | :--- |
 | Outcome Reward Model (ORM) | Final answer correctness | No signal for reasoning quality |
 | Process Reward Model (PRM) | Each intermediate reasoning step | Higher training complexity |

PRMs dramatically reduce multi-step hallucinations by aligning rewards with valid intermediate logic.

### 5.2 GRPO: Group Relative Policy Optimization

Introduced prominently in DeepSeek-R1, GRPO:

- Samples multiple reasoning trajectories from the same model
- Ranks them relative to each other
- Reinforces the most efficient and correct paths

This removes reliance on large teacher models and enables reasoning capability to emerge from self-comparison.

---

## 6. Distillation of Reasoning

Reasoning capabilities are **distillable**, even when learned via reinforcement learning.

### Distillation Pipeline
1. Run a large System 2 model with high inference compute
2. Collect validated reasoning traces
3. Use them as supervised fine-tuning data for smaller models

This has enabled 7B–14B models to exhibit structured reasoning previously limited to frontier-scale systems.

---

## 7. Failure Modes and Trade-offs

### 7.1 Reward Hacking
Models may learn that:
- Longer reasoning chains yield higher rewards
- Redundancy masquerades as correctness

Mitigations include length penalties, diversity rewards, and consistency checks.

### 7.2 Overthinking
- Excessive inference compute yields diminishing returns
- Latency increases without accuracy gains

Adaptive early stopping is commonly used to terminate reasoning once confidence thresholds are met.

---

## 8. Evaluation and Benchmarks

Reasoning models require benchmarks that stress multi-step cognition.

| Benchmark | Capability Tested |
| :--- | :--- |
| AIME | Olympiad-level mathematics |
| GPQA | Graduate-level scientific reasoning |
| HumanEval | Algorithmic and coding logic |
| Codeforces | Long-horizon program synthesis |

Evaluation must consider **accuracy versus compute trade-offs**, not accuracy alone.

---

## 9. Deployment Decision Framework

Choosing between System 1 and System 2 depends on:
- Latency tolerance
- Cost constraints
- Error sensitivity

**System 1** models are preferred for high-throughput, low-risk tasks.

**System 2** models are preferred when correctness dominates, such as legal analysis, scientific reasoning, and complex debugging.

Hybrid systems increasingly route queries dynamically based on estimated difficulty.

---


## 10. Notable Models of 2025–2026

* **OpenAI o1 & o3:** The first to commercialize "Reasoning Tokens." o3 is currently the gold standard for competitive programming (Codeforces) and high-level mathematics.
* **DeepSeek-R1:** A massive breakthrough for open-source AI. It proved that RL can "distill" reasoning capabilities into smaller models (like 7B or 14B parameters).
* **Claude 3.7 Sonnet:** Introduced "Hybrid Reasoning," allowing the user to toggle "Extended Thinking." This solves the latency issue by letting users choose when they need the model to "think."
* **Gemini 2.5 Pro (Reasoning):** Uses Google’s massive context window to reason across hours of video or thousands of lines of code simultaneously.

---

## 11. Relationship to Training Phases

Thinking behavior emerges from the interaction of all training stages:

- Pre-training provides linguistic and world knowledge
- Mid-training injects structured reasoning and long-context priors
- RL-based post-training activates inference-time computation strategies

System 2 capability is therefore **not purely a post-training artifact**.

---