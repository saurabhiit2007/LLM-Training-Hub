This documentation tracks the transition from "System 1" (probabilistic next-token prediction) to "System 2" (deliberate, slow thinking) in Large Language Models.

---

## 🏗️ The Three Pillars of AI Reasoning

### 1. The Training Objective: Learning to Think
Reasoning is not just a prompting trick; it is a learned capability baked into model weights during post-training.

* **CoT Training (STaR):** Moving beyond simple "think step-by-step" prompts. We now fine-tune models on massive datasets of reasoning traces where the model has successfully "self-corrected" to reach a solution.
* **RLVR (Reinforcement Learning from Verifiable Rewards):** Unlike RLHF, which is subjective, RLVR uses objective ground truths.
    * **Math & Code:** These serve as "Gold Standard" domains because the reward signal is binary (the code compiles or it doesn't; the equation balances or it doesn't).
* **PRM vs. ORM:**
    * **ORM (Outcome Reward Model):** Rewards only the final answer. High risk of "reward hacking" (getting the right answer for the wrong reason).
    * **PRM (Process Reward Model):** Rewards the model for every intermediate logical step. This is the primary driver of performance in models like OpenAI o1.

---

---

### 2. The Inference Architecture: Scaling Test-Time Compute (TTC)
The "Inference Scaling Law" states that model performance scales with the amount of compute dedicated to "thinking" during the generation phase.

* **Search Strategies:**
    * **Best-of-N:** Sampling $N$ different completions and using a verifier to pick the best one.
    * **MCTS (Monte Carlo Tree Search):** Navigating a tree of possible reasoning steps, simulating outcomes, and backtracking when a path fails.


* **Compute-Optimal Allocation:** Modern systems use a "router" to decide how much TTC to spend.
    * *Low Difficulty:* 1 forward pass (System 1).
    * *High Difficulty:* Thousands of search iterations (System 2).


* **Speculative Reasoning:** Using a small, fast model to draft reasoning steps and a large model to verify them in batches, reducing latency.

---

---

### 3. Behavioral Dynamics: Self-Correction & Traces

How a model manages its internal logic and interacts with the user.

* **Hidden vs. Visible Traces:**
    * **Visible:** High interpretability, but prone to user-influence and "sycophancy."
    * **Hidden:** Used by o1/R1 to prevent "Chain-of-Thought hacking." It allows for **Neuralese**—an internal, highly efficient symbolic language the model develops to reason faster than human language allows.


* **Reflection Loops:** The model is trained to recognize "dead ends." If a logic path contradicts a prior step, the model is penalized and forced to backtrack.


* **Steganography Risk:** A safety concern where models might hide "unsafe" reasoning or planning within hidden thought traces to bypass monitoring.

---

## 🚀 Interview Prep: High-Signal Questions

| Question | Key Insight |
| :--- | :--- |
| **Why use RLVR over RLHF for reasoning?** | RLHF is limited by human checking speed and bias; RLVR uses automated verifiers (compilers/solvers) to scale exponentially. |
| **What is the "Pause" token?** | A latent mechanism (or specific token) that allows a model to perform more internal computation before committing to a visible output. |
| **How does DeepSeek's GRPO change scaling?** | It removes the need for a separate "Critic" model during RL, significantly reducing the VRAM required to train reasoning capabilities. |

---

## 🛠️ Project Ideas for this Repo
1.  **Verifier Implementation:** Build a Python script that uses a Process Reward Model to score a math reasoning trace.
2.  **TTC Benchmark:** Compare the accuracy of a 7B model using "Best-of-10" search against a 70B model using a single pass.
3.  **Trace Visualization:** Create a tool to visualize MCTS tree exploration in a logical puzzle.