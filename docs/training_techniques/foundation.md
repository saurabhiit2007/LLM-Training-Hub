# LLM Training Foundations

## 1 What is an LLM Optimizing?

At its core, a Large Language Model (LLM) is a probabilistic system designed to model the distribution of natural language. Despite emergent reasoning and planning behaviors, the training objective itself is simple: **reduce uncertainty about the next token given prior context**.

### 1.1 The Autoregressive Objective

Most LLMs are trained using an **Autoregressive (AR)** or **Causal Language Modeling (CLM)** objective. The joint probability of a token sequence is factorized as a product of conditional probabilities:

$$
P(x_1, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})
$$

This formulation assumes:

- Tokens are conditionally independent given their history
- Knowledge is captured implicitly through long context dependencies
- Tokenization defines the atomic units of prediction

### 1.2 The Loss Function: Negative Log-Likelihood

Training minimizes the **Negative Log-Likelihood (NLL)** of the observed data, which is equivalent to **Cross-Entropy Loss**:

$$
\mathcal{L}(\theta) = - \mathbb{E}_{(x_1,\dots,x_T) \sim D} \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})
$$

Where:
- $\theta$ are the model parameters
- $x_{<t}$ is the context window
- $P_\theta$ is the predicted probability distribution produced by a Softmax layer

### 1.3 Cross-Entropy, KL Divergence, and Learning

Minimizing cross-entropy implicitly minimizes the **KL divergence** between the true data distribution \(P_{data}\) and the model distribution \(P_\theta\):

$$
H(P_{data}, P_\theta) = H(P_{data}) + D_{KL}(P_{data} \| P_\theta)
$$

Since $H(P_{data})$ is fixed, training pushes the model distribution closer to the real language distribution. This connects LLM training directly to information theory and compression.

### 1.4 Why Next-Token Prediction Works (The Compression Hypothesis)

Predicting the next token forces the model to internalize structure across many domains.

- **Compression implies abstraction:** To compress diverse text, the model must learn syntax, semantics, facts, and procedures.
- **Softmax competition:** Increasing probability mass for the correct token necessarily decreases mass for alternatives, encouraging fine-grained representations.
- **Generalization pressure:** Predicting well across domains requires reusable internal features, which later appear as reasoning, translation, or coding skills.

---

## 2 Language Modeling and Likelihood Minimization

### 2.1 Maximum Likelihood Estimation (MLE)

LLM training is an instance of **Maximum Likelihood Estimation**, where we seek parameters \(\theta^*\) that maximize the likelihood of observed text:

$$
\theta^* = \arg\max_\theta \mathbb{E}_{x \sim D} [\log P_\theta(x)]
$$

MLE provides a stable and scalable objective but does not encode preferences such as helpfulness, safety, or instruction following.

### 2.2 Teacher Forcing

Teacher forcing is a training strategy where the model is conditioned on the **ground-truth previous token** rather than its own prediction when computing the next-token loss. For a sequence \(x_1, \dots, x_T\), the model always receives \(x_{t-1}\) as input when predicting \(x_t\), even if it would have predicted a different token at step \(t-1\).

**Small Practical Example**

**Task:** Predict the sentence  
> *“The cat sat on the mat”*

#### Step 1: Training data
Tokens:  
`[The, cat, sat, on, the, mat]`

#### Step 2: Training with teacher forcing
At each step, the model is conditioned on the **ground-truth previous token**.

1. **Input:** `The`  
   **Target:** `cat`

2. **Input:** `The cat`  
   **Target:** `sat`

3. **Input:** `The cat sat`  
   **Target:** `on`

4. **Input:** `The cat sat on`  
   **Target:** `the`

5. **Input:** `The cat sat on the`  
   **Target:** `mat`

Even if the model predicts an incorrect token at any step, the **next input still uses the true token** from the dataset.

#### Step 3: Inference time behavior
At inference, the model must condition on its **own predictions**.

1. **Input:** `The`  
   **Prediction:** `dog`

2. **Next input:** `The dog`  
   Errors now propagate forward

This train–test mismatch is known as **exposure bias** and is a key limitation of teacher forcing.

**Advantages:**

- **Full parallelization** of loss computation across all positions in a Transformer
- **Stable gradients**, since errors do not compound during training

**Limitation: Exposure Bias**

- Introduces **exposure bias**: during inference, the model must condition on its own past predictions, a distribution shift that can lead to error accumulation. This gap motivates post-training techniques such as supervised fine-tuning and reinforcement learning based alignment.

### 2.3 Perplexity as an Evaluation Metric

**Perplexity** measures how uncertain a language model is, on average, when predicting the next token in a sequence.

Formally, if a model is trained using cross-entropy loss $\mathcal{L}$, then perplexity is defined as:

$$
\text{PPL} = \exp(\mathcal{L})
$$

where $\mathcal{L}$ is the **average negative log-probability** assigned to the correct next token.

#### Intuition

A language model repeatedly answers the question:

> How many plausible choices do I believe the next token could be?

Perplexity converts log probabilities back into an interpretable scale representing the **effective number of choices**.

- **PPL = 1**  
  The model is fully confident and assigns probability 1 to the correct token.

- **PPL = 10**  
  The model behaves as if it is choosing uniformly among about 10 tokens.

- **PPL = 100**  
  The model is highly uncertain, with roughly 100 plausible next tokens.

This is why perplexity is often described as the model’s **average branching factor**.

#### Why Lower Perplexity Is Better

- Lower PPL means the model assigns higher probability to the correct next token.
- This corresponds to lower uncertainty and better next-token prediction.
- Minimizing cross-entropy during training directly minimizes perplexity.

#### Important Clarification

Perplexity is **not** a direct measure of:
- Reasoning ability
- Factual correctness
- Alignment or instruction following

It only measures how well the model predicts the data distribution. While lower perplexity often correlates with better text quality, it does not guarantee better downstream performance.

---

## 3. Why Scaling Works and Where it Breaks

### 3.1 Empirical Scaling Laws

Performance improves predictably as a power-law function of:

- Model parameters
- Training tokens
- Compute budget

This empirical behavior explains the rapid gains from larger models.

#### Kaplan Scaling Laws (2020)

Early results suggested scaling model size was the dominant factor.

- Loss scales roughly as a power-law in parameter count
- Data was treated as effectively unlimited

#### Chinchilla Scaling Laws (2022)

Later work showed most large models were undertrained.

Key findings:

- Optimal performance requires balancing parameters and tokens
- Roughly **20 training tokens per parameter** is compute optimal
- Smaller models trained on more data can outperform larger undertrained models

This led to data-centric model design such as LLaMA and Mistral.

---

### 3.2 Where Scaling Breaks

#### 1. Data Scarcity and Synthetic Feedback Loops

- High-quality human text is limited
- Synthetic data risks reducing diversity
- Repeated self-training can lead to model collapse

#### 2. Capability Saturation

- Loss improves smoothly, but abilities emerge discontinuously
- Reasoning, planning, and tool use do not scale linearly with perplexity
- Small loss gains can hide large behavioral differences

#### 3. Inference Cost and Latency

- Larger models increase memory, latency, and cost
- This motivates inference-efficient designs

#### 4. Test-Time Scaling

- Recent systems scale inference compute rather than parameters
- Models generate longer internal reasoning traces
- This shifts scaling from training time to inference time

Examples include OpenAI o1 and DeepSeek-R1.

---