# 📊 Optimizers and Learning Rate Schedules for LLM Training

## 1. Overview

Training large language models requires careful selection of **optimization algorithms** and **learning rate schedules**. The wrong choice can lead to:
- Divergence (loss → NaN)
- Slow convergence (wasted compute)
- Suboptimal final performance

This guide covers the optimizers and schedules used in modern LLM training, with a focus on **why they work** and **when to use them**.

---

## 2. Optimizers for LLM Training

### 2.1 Why Adam Dominates LLM Training

**AdamW** (Adam with decoupled Weight decay) is the de facto standard for training Transformers.

**Adam update rule:**
$$
\begin{align}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \\
\theta_t &= \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \lambda \theta_{t-1}
\end{align}
$$

Where:
- $m_t$: First moment (momentum)
- $v_t$: Second moment (variance)
- $\beta_1, \beta_2$: Decay rates
- $\alpha$: Learning rate
- $\lambda$: Weight decay (decoupled in AdamW)

**Why Adam works for Transformers:**
1. **Adaptive learning rates**: Different layers need different learning rates
2. **Momentum**: Smooths noisy gradients
3. **Sparse gradients**: Many parameters updated infrequently (embeddings)
4. **Scale invariance**: Normalizes by gradient variance

### 2.2 Standard Hyperparameters

**Default values (used in most LLM papers):**

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,              # Learning rate (often decayed)
    betas=(0.9, 0.999),   # (β₁, β₂) for momentum and variance
    eps=1e-8,             # Numerical stability
    weight_decay=0.1,     # L2 regularization
)
```

**Typical ranges:**
- Learning rate: `1e-5` to `1e-3` (varies with model size)
- β₁: `0.9` (rarely changed)
- β₂: `0.95` to `0.999` (higher for larger models)
- Weight decay: `0.01` to `0.1`

### 2.3 AdamW vs Adam: The Weight Decay Difference

**Standard Adam (L2 regularization):**
$$
g_t = \nabla_\theta L(\theta_t) + \lambda \theta_t
$$
Weight decay is added to gradients before adaptive scaling.

**AdamW (decoupled weight decay):**
$$
\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \alpha \lambda \theta_{t-1}
$$
Weight decay is applied directly to parameters.

**Why AdamW is better:**
- Weight decay effectiveness doesn't depend on gradient magnitude
- More consistent regularization across layers
- Better generalization in practice

**Empirical result:** Most modern LLMs use AdamW exclusively.

### 2.4 Alternative Optimizers

#### **SGD with Momentum**

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4,
)
```

**Pros:**
- Simple, well-understood
- Lower memory (1× state vs Adam's 2×)

**Cons:**
- Requires careful learning rate tuning
- Slower convergence for Transformers
- Not adaptive (all parameters share same LR)

**Usage:** Rarely used for LLMs; more common in computer vision.

#### **Adafactor**

```python
from transformers import Adafactor

optimizer = Adafactor(
    model.parameters(),
    lr=1e-3,
    scale_parameter=True,
    relative_step=False,
    warmup_init=False,
)
```

**Key difference:** Factorized second moment (see memory-efficient optimizers doc)

**Pros:**
- 50% memory vs Adam
- Designed for T5/encoder-decoder models

**Cons:**
- Different convergence behavior
- Requires tuning for new architectures

**Usage:** T5, Flan-T5, some encoder-decoder models

#### **LION (EvoLved Sign Momentum)**

```python
from lion_pytorch import LION

optimizer = LION(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.99),
    weight_decay=0.1,
)
```

**Key idea:** Only use sign of momentum, not magnitude

**Pros:**
- 50% memory vs Adam
- Competitive performance on some tasks
- Simpler algorithm

**Cons:**
- Less proven at scale
- Requires different LR tuning

**Usage:** Experimental; gaining traction for vision-language models

---

## 3. Learning Rate Schedules

### 3.1 Why Schedules Matter

A **fixed learning rate** is rarely optimal:
- **Too high initially**: Divergence or instability
- **Too high later**: Oscillation around optimum
- **Too low throughout**: Slow convergence

**Learning rate schedules** adapt the LR during training to balance exploration and convergence.

### 3.2 Warmup: The Critical First Phase

**Problem without warmup:**
- Early in training, gradients are large and noisy
- Adam's second moment estimate $v_t$ is inaccurate (based on few samples)
- High LR + inaccurate estimates → divergence

**Solution: Linear warmup**
$$
\alpha_t = \alpha_{\text{max}} \cdot \min\left(1, \frac{t}{T_{\text{warmup}}}\right)
$$

```python
from torch.optim.lr_scheduler import LambdaLR

def get_warmup_schedule(optimizer, num_warmup_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0
    
    return LambdaLR(optimizer, lr_lambda)
```

**Typical warmup durations:**
- Small models (<1B): 500-2000 steps
- Medium models (1-10B): 2000-10000 steps
- Large models (>10B): 10000-50000 steps

**Rule of thumb:** Warmup for ~0.5-2% of total training steps.

### 3.3 Common Schedules

#### **Linear Decay with Warmup**

Most common for fine-tuning.

$$
\alpha_t = \begin{cases}
\alpha_{\text{max}} \cdot \frac{t}{T_{\text{warmup}}} & \text{if } t \leq T_{\text{warmup}} \\
\alpha_{\text{max}} \cdot \frac{T_{\text{total}} - t}{T_{\text{total}} - T_{\text{warmup}}} & \text{if } t > T_{\text{warmup}}
\end{cases}
$$

```python
from transformers import get_linear_schedule_with_warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=10000,
)

# Training loop
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch).loss
    loss.backward()
    optimizer.step()
    scheduler.step()  # Update LR
```

**Visualization:**
```
LR
│     
│    ╱╲
│   ╱  ╲
│  ╱    ╲___
│ ╱         ╲___
│╱              ╲___
└──────────────────── Steps
  Warmup   Training
```

**Pros:**
- Simple, predictable
- Works well for fine-tuning

**Cons:**
- LR drops to zero at end (may hurt final performance)

#### **Cosine Decay with Warmup**

Most common for pretraining.

$$
\alpha_t = \begin{cases}
\alpha_{\text{max}} \cdot \frac{t}{T_{\text{warmup}}} & \text{if } t \leq T_{\text{warmup}} \\
\alpha_{\text{min}} + \frac{1}{2}(\alpha_{\text{max}} - \alpha_{\text{min}})\left(1 + \cos\left(\pi \cdot \frac{t - T_{\text{warmup}}}{T_{\text{total}} - T_{\text{warmup}}}\right)\right) & \text{if } t > T_{\text{warmup}}
\end{cases}
$$

```python
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=2000,
    num_training_steps=100000,
    num_cycles=0.5,  # 0.5 = standard cosine
)
```

**Visualization:**
```
LR
│     
│    ╱╲
│   ╱  ╲___
│  ╱       ╲___
│ ╱            ╲___
│╱                 ╲___
└──────────────────────── Steps
  Warmup    Cosine Decay
```

**Pros:**
- Smooth decay (no sharp drops)
- Better final performance than linear
- Used in GPT-3, LLaMA, most modern LLMs

**Cons:**
- Requires knowing total training steps upfront

**Used in:** GPT-3, LLaMA, PaLM, Chinchilla

#### **Cosine with Restarts (SGDR)**

Multiple cosine cycles with restarts.

```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=5000,      # Steps in first cycle
    T_mult=2,      # Multiply cycle length by 2 each restart
    eta_min=1e-6,  # Minimum LR
)
```

**Visualization:**
```
LR
│  ╱╲  ╱╲    ╱╲        ╱╲
│ ╱  ╲╱  ╲  ╱  ╲      ╱  ╲
│╱        ╲╱    ╲    ╱    ╲
│              ╲  ╲  ╱      ╲
└───────────────╲─╲╱────────╲─── Steps
```

**Pros:**
- Escape local minima
- Good for long training runs

**Cons:**
- Spikes in LR can destabilize training
- Less common for LLMs

**Usage:** Experimental; occasionally used for continued pretraining.

#### **Inverse Square Root Decay**

Used in original Transformer paper.

$$
\alpha_t = \alpha_{\text{max}} \cdot \min\left(\frac{1}{\sqrt{t}}, \frac{t}{T_{\text{warmup}}^{1.5}}\right)
$$

```python
def get_inverse_sqrt_schedule(optimizer, num_warmup_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return (num_warmup_steps / current_step) ** 0.5
    
    return LambdaLR(optimizer, lr_lambda)
```

**Visualization:**
```
LR
│    ╱
│   ╱
│  ╱
│ ╱___________
│╱
└──────────────── Steps
```

**Pros:**
- Decays slowly (good for long training)
- Simple formula

**Cons:**
- Never reaches zero
- Less popular than cosine

**Usage:** Original Transformer, some translation models

#### **Constant with Warmup**

Simple: warmup then constant LR.

```python
from transformers import get_constant_schedule_with_warmup

scheduler = get_constant_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
)
```

**Pros:**
- Simplest schedule
- Good for short fine-tuning

**Cons:**
- No decay may hurt final performance

**Usage:** Quick fine-tuning, LoRA, experimentation

---

## 4. Choosing Learning Rates

### 4.1 Scaling Laws

Learning rate should scale with:
1. **Model size**: Larger models need smaller LR
2. **Batch size**: Larger batches need larger LR (approximately linear)
3. **Sequence length**: Longer sequences may need smaller LR

**Empirical scaling rule (Chinchilla paper):**
$$
\alpha_{\text{max}} \approx \frac{0.003}{\sqrt{N_{\text{params}}}} \times \sqrt{\frac{B_{\text{effective}}}{256}}
$$

Where $N_{\text{params}}$ is in billions, $B_{\text{effective}}$ is effective batch size.

**Examples:**

| Model Size | Batch Size | Suggested Max LR |
|------------|-----------|------------------|
| 1B | 512 | 6e-4 |
| 7B | 1024 | 3e-4 |
| 13B | 2048 | 2e-4 |
| 70B | 4096 | 1e-4 |

### 4.2 Learning Rate Finder

**Method:** Gradually increase LR and plot loss.

```python
def find_lr(model, dataloader, optimizer, init_lr=1e-7, final_lr=1.0, num_steps=100):
    """
    Learning rate range test.
    Returns: (lrs, losses)
    """
    model.train()
    lr_mult = (final_lr / init_lr) ** (1 / num_steps)
    
    lrs, losses = [], []
    lr = init_lr
    
    for i, batch in enumerate(dataloader):
        if i >= num_steps:
            break
        
        # Set learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Forward/backward
        optimizer.zero_grad()
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        
        # Record
        lrs.append(lr)
        losses.append(loss.item())
        
        # Increase LR
        lr *= lr_mult
    
    return lrs, losses

# Usage
lrs, losses = find_lr(model, train_loader, optimizer)

# Plot
import matplotlib.pyplot as plt
plt.plot(lrs, losses)
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.show()

# Choose LR where loss decreases fastest (steepest descent)
```

**Interpretation:**
```
Loss
│
│       ╱
│      ╱
│     ╱
│    ╱
│   ╱________
│  ╱
│ ╱
│╱
└──────────── LR (log scale)
   ^
   Pick this LR
   (steepest descent)
```

**Recommended max LR:** Where loss decreases fastest, divided by 3-10.

### 4.3 Per-Layer Learning Rates (Layer-wise LR Decay)

**Observation:** Lower layers (embeddings) change more slowly than higher layers.

**Solution:** Use smaller LR for lower layers.

```python
def get_layer_lrs(model, base_lr=1e-4, decay_rate=0.9):
    """
    Assign decreasing LR to lower layers.
    """
    no_decay = ["bias", "LayerNorm.weight"]
    layer_lrs = []
    
    num_layers = len(model.transformer.h)  # Assuming GPT-like model
    
    for name, param in model.named_parameters():
        # Determine layer number
        if "transformer.h" in name:
            layer_num = int(name.split(".h.")[1].split(".")[0])
            lr = base_lr * (decay_rate ** (num_layers - layer_num - 1))
        else:
            lr = base_lr  # Embeddings, head, etc.
        
        # Weight decay
        if any(nd in name for nd in no_decay):
            layer_lrs.append({"params": [param], "lr": lr, "weight_decay": 0.0})
        else:
            layer_lrs.append({"params": [param], "lr": lr, "weight_decay": 0.1})
    
    return layer_lrs

# Usage
optimizer = torch.optim.AdamW(
    get_layer_lrs(model, base_lr=1e-4, decay_rate=0.9),
)
```

**Used in:** ELECTRA, DeBERTa, some fine-tuning strategies

---

## 5. Gradient Clipping

### 5.1 Why Clip Gradients?

Large gradients can:
- Cause exploding gradients (loss → NaN)
- Destabilize optimizer state
- Waste training steps

**Gradient clipping** bounds gradient magnitude to prevent instability.

### 5.2 Global Norm Clipping (Most Common)

Clip based on total gradient norm across all parameters:

$$
\text{if } \|\mathbf{g}\| > \tau: \quad \mathbf{g} \leftarrow \tau \cdot \frac{\mathbf{g}}{\|\mathbf{g}\|}
$$

```python
import torch.nn.utils

# After backward, before optimizer.step()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

**Typical values:**
- Pretraining: `max_norm=1.0`
- Fine-tuning: `max_norm=0.3` to `1.0`
- Unstable training: `max_norm=0.1` to `0.5`

### 5.3 Per-Parameter Clipping

Clip each parameter's gradient individually:

```python
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

**Less common:** Global norm clipping is preferred for Transformers.

### 5.4 Adaptive Gradient Clipping (AGC)

Clip relative to parameter magnitude:

$$
\lambda = \frac{\|\theta\|}{\|\mathbf{g}\| + \epsilon}
$$

If $\lambda < \tau$, scale $\mathbf{g} \leftarrow \lambda \cdot \mathbf{g}$.

```python
def adaptive_clip_grad(parameters, clip_factor=0.01, eps=1e-3):
    """
    Clip gradients based on parameter norm.
    """
    for p in parameters:
        if p.grad is None:
            continue
        
        p_norm = p.data.norm()
        g_norm = p.grad.data.norm()
        
        max_norm = max(p_norm, eps) * clip_factor
        
        if g_norm > max_norm:
            p.grad.data.mul_(max_norm / (g_norm + 1e-6))
```

**Used in:** NFNets, some vision models; experimental for LLMs

---

## 6. Advanced Topics

### 6.1 Gradient Accumulation with Schedules

When using gradient accumulation, ensure scheduler steps correctly:

```python
accumulation_steps = 4
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=10000 // accumulation_steps,  # Adjust for accumulation
)

for i, batch in enumerate(dataloader):
    loss = model(**batch).loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()  # Step after accumulated update
        optimizer.zero_grad()
```

### 6.2 Learning Rate Scaling for Large Batches

**Linear scaling rule** (Goyal et al.):
$$
\alpha_{\text{new}} = \alpha_{\text{base}} \times \frac{B_{\text{new}}}{B_{\text{base}}}
$$

**Example:**
- Base LR: `1e-4` with batch size `256`
- New batch size: `2048` (8× larger)
- New LR: `8e-4`

**Caveat:** Breaks down for very large batches (>8k). Use square root scaling:
$$
\alpha_{\text{new}} = \alpha_{\text{base}} \times \sqrt{\frac{B_{\text{new}}}{B_{\text{base}}}}
$$

### 6.3 Learning Rate Re-warming (Continued Pretraining)

When resuming training or continuing pretraining:

```python
# Initial training
scheduler1 = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=2000,
    num_training_steps=100000,
)

# Continued training (re-warm)
scheduler2 = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,  # Shorter warmup
    num_training_steps=120000,  # Total steps including previous
)
```

**Best practice:** Use shorter warmup (20-50% of original) when resuming.

---

## 7. Interview Questions

### Q1: Why does Adam work better than SGD for Transformers?

**Answer:**

**Three key advantages:**

1. **Adaptive learning rates:**
   - Transformers have parameters with very different gradient scales
   - Embeddings: sparse, infrequent updates
   - Attention: dense, frequent updates
   - SGD uses same LR for all → poor for sparse parameters
   - Adam normalizes by $\sqrt{v_t}$ → each parameter gets effective LR

2. **Momentum helps with noise:**
   - Transformer gradients are noisy (attention is stochastic)
   - First moment $m_t$ smooths gradients
   - Prevents oscillation

3. **Scale invariance:**
   - Layer norm rescales activations
   - Adam is invariant to gradient scaling (division by $\sqrt{v_t}$)
   - SGD is not → requires careful LR tuning per layer

**Empirical evidence:**
- GPT-2/3, BERT, T5: all use Adam/AdamW
- SGD rarely used except vision models (ResNets, etc.)

**Follow-up:** Why not use SGD with per-layer LRs?  
**Answer:** Possible, but requires manual tuning. Adam automates this adaptation.

---

### Q2: Explain the purpose of learning rate warmup. Why is it critical?

**Answer:**

**Without warmup:**

Early in training (step 1-1000):
- Parameters initialized randomly (e.g., Xavier/Kaiming)
- Gradients are large and noisy
- Adam's second moment $v_t$ is inaccurate (based on few samples)

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

When $t$ is small, $v_t$ hasn't converged to true gradient variance.

**With high LR early:**
$$
\theta_t = \theta_{t-1} - \alpha \frac{m_t}{\sqrt{v_t} + \epsilon}
$$

If $v_t$ is too small (underestimate), step size is huge → **divergence**.

**Warmup solution:**
- Start with very small LR (e.g., 0)
- Gradually increase to max LR over $T_{\text{warmup}}$ steps
- Gives $v_t$ time to stabilize

**Visualization:**
```
Gradient variance estimate
│
│         ╱─────── True variance
│        ╱
│       ╱
│   ___╱
│  ╱
│ ╱
│╱
└──────────── Steps
  ↑
  Warmup period
```

**Empirical rule:** Warmup for 0.5-2% of total training steps.

**Follow-up:** What happens if warmup is too short?  
**Answer:** Training becomes unstable, loss may spike or diverge. If too long, wastes training time (slow initial progress).

---

### Q3: Compare cosine vs linear decay schedules. When should you use each?

**Answer:**

**Linear Decay:**
$$
\alpha_t = \alpha_{\text{max}} \left(1 - \frac{t - T_{\text{warmup}}}{T_{\text{total}} - T_{\text{warmup}}}\right)
$$

**Cosine Decay:**
$$
\alpha_t = \alpha_{\text{min}} + \frac{1}{2}(\alpha_{\text{max}} - \alpha_{\text{min}})\left(1 + \cos\left(\pi \cdot \frac{t - T_{\text{warmup}}}{T_{\text{total}} - T_{\text{warmup}}}\right)\right)
$$

**Comparison:**

| Aspect | Linear | Cosine |
|--------|--------|--------|
| **Decay rate** | Constant | Slow initially, fast at end |
| **Final LR** | 0 | Small but non-zero |
| **Smoothness** | Sharp drop | Smooth curve |
| **Final performance** | Good | **Better** (empirically) |
| **Typical use** | Fine-tuning | Pretraining |

**Why cosine is better for pretraining:**

1. **Gradual decay:** Model doesn't suddenly lose learning capacity
2. **Non-zero final LR:** Continues learning until the end
3. **Smooth transitions:** Less likely to get stuck in poor local minima

**Visualization:**
```
Linear:                Cosine:
LR                     LR
│ ╲                    │ ╲___
│  ╲                   │     ╲__
│   ╲                  │        ╲__
│    ╲                 │           ╲__
│     ╲                │              ╲__
└──────                └─────────────────
```

**Decision guide:**
- **Pretraining (100k+ steps):** Cosine (GPT-3, LLaMA use this)
- **Fine-tuning (1k-10k steps):** Linear or cosine (similar performance)
- **Quick experiments:** Constant with warmup

**Follow-up:** What if you don't know total training steps?  
**Answer:** Use inverse square root decay or implement dynamic scheduling that adjusts based on validation loss.

---

### Q4: You're training a 7B model and loss diverges at step 5000. Debug the optimizer/schedule.

**Answer:**

**Systematic debugging approach:**

**1. Check gradient norms:**
```python
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5

print(f"Gradient norm: {total_norm}")
```

**If norm > 100:** Exploding gradients
- **Fix:** Add/reduce gradient clipping
  ```python
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  ```

**2. Check learning rate at divergence:**
```python
current_lr = optimizer.param_groups[0]['lr']
print(f"LR at step 5000: {current_lr}")
```

**If LR > 5e-4 for 7B model:** Too high
- **Fix:** Reduce max LR to 3e-4, increase warmup
  ```python
  scheduler = get_cosine_schedule_with_warmup(
      optimizer,
      num_warmup_steps=10000,  # Longer warmup
      num_training_steps=100000,
  )
  optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
  ```

**3. Check for NaN/Inf in activations:**
```python
def check_nan_hook(module, input, output):
    if torch.isnan(output).any():
        print(f"NaN detected in {module.__class__.__name__}")

for module in model.modules():
    module.register_forward_hook(check_nan_hook)
```

**If NaN detected:** Numerical instability
- **Fix:** Use BF16 instead of FP16, reduce LR, check for division by zero

**4. Inspect optimizer state:**
```python
# Check second moment (variance)
for name, param in model.named_parameters():
    if param in optimizer.state:
        state = optimizer.state[param]
        if 'exp_avg_sq' in state:
            v = state['exp_avg_sq']
            print(f"{name}: mean v = {v.mean():.2e}, max v = {v.max():.2e}")
```

**If v values are tiny (< 1e-10):** Underflow in variance estimate
- **Fix:** Increase epsilon in Adam
  ```python
  optimizer = torch.optim.AdamW(model.parameters(), eps=1e-6)  # Larger eps
  ```

**5. Common fixes ranked by likelihood:**
1. **Reduce max LR** (most common cause)
2. **Add/tighten gradient clipping**
3. **Increase warmup steps**
4. **Switch FP16 → BF16** (if using mixed precision)
5. **Check data quality** (corrupted batches)

**Example fix:**
```python
# Original (diverged)
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)
scheduler = get_cosine_schedule_with_warmup(optimizer, 2000, 100000)

# Fixed
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, eps=1e-6)
scheduler = get_cosine_schedule_with_warmup(optimizer, 5000, 100000)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

### Q5: Explain AdamW vs Adam. Why is weight decay decoupling important?

**Answer:**

**Standard Adam with L2 regularization:**
$$
g_t = \nabla_\theta L(\theta_t) + \lambda \theta_t
$$

Then apply Adam update:
$$
\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

**Problem:** Weight decay is added to gradient *before* adaptive scaling.

**Effect on parameter $\theta_i$ with small gradient:**
- Gradient: $g_i = 0.01 + \lambda \theta_i$
- Variance: $v_i$ is large (from history)
- Effective step: $\frac{0.01 + \lambda \theta_i}{\sqrt{v_i}}$ is tiny

**Effect on parameter $\theta_j$ with large gradient:**
- Gradient: $g_j = 10.0 + \lambda \theta_j$
- Variance: $v_j$ is large
- Effective step: $\frac{10.0 + \lambda \theta_j}{\sqrt{v_j}}$ is moderate

**Result:** Weight decay is **adaptive** (scaled by gradient magnitude), which is not the intent of regularization.

**AdamW (decoupled weight decay):**
$$
\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \alpha \lambda \theta_{t-1}
$$

Weight decay is applied directly to parameters, *after* adaptive scaling.

**Benefit:**
- Weight decay magnitude is **constant** across all parameters
- More effective regularization
- Better generalization

**Empirical evidence (from AdamW paper):**

| Model | Adam | AdamW |
|-------|------|-------|
| BERT | 84.2% | **85.1%** |
| GPT-2 | - | (all models use AdamW) |

**Follow-up:** When does the difference matter most?  
**Answer:** Large models with diverse parameter scales (embeddings vs attention). For small models or when not using weight decay, Adam ≈ AdamW.

---

### Q6: How would you implement a custom learning rate schedule for cyclical learning?

**Answer:**

**Cyclical learning rate:** Periodically increase/decrease LR to escape local minima.

```python
import torch
from torch.optim.lr_scheduler import LambdaLR
import math

class CyclicalScheduleWithWarmup:
    """
    Cyclical cosine schedule with warmup.
    
    Schedule:
    1. Linear warmup to max_lr
    2. Cosine cycle between max_lr and min_lr
    3. Repeat cycle every cycle_length steps
    """
    def __init__(
        self,
        optimizer,
        num_warmup_steps,
        cycle_length,
        max_lr=1e-4,
        min_lr=1e-6,
        num_cycles=None,
    ):
        self.num_warmup_steps = num_warmup_steps
        self.cycle_length = cycle_length
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.num_cycles = num_cycles
        
        self.scheduler = LambdaLR(optimizer, self.lr_lambda)
    
    def lr_lambda(self, current_step):
        # Warmup phase
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        
        # Check if we've exceeded max cycles
        if self.num_cycles is not None:
            cycles_completed = (current_step - self.num_warmup_steps) // self.cycle_length
            if cycles_completed >= self.num_cycles:
                return self.min_lr / self.max_lr  # Stay at min
        
        # Cosine cycle
        progress = (current_step - self.num_warmup_steps) % self.cycle_length
        progress = progress / self.cycle_length  # Normalize to [0, 1]
        
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        lr_range = self.max_lr - self.min_lr
        lr = self.min_lr + lr_range * cosine_decay
        
        return lr / self.max_lr  # Return multiplier
    
    def step(self):
        self.scheduler.step()

# Usage
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = CyclicalScheduleWithWarmup(
    optimizer,
    num_warmup_steps=1000,
    cycle_length=5000,
    max_lr=1e-4,
    min_lr=1e-6,
    num_cycles=10,
)

for epoch in range(num_epochs):
    for batch in dataloader:
        loss = model(batch).loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

**Visualization:**
```
LR
│    ╱╲    ╱╲    ╱╲
│   ╱  ╲  ╱  ╲  ╱  ╲
│  ╱    ╲╱    ╲╱    ╲
│ ╱
│╱
└─────────────────────── Steps
 Warmup   Cycle 1  Cycle 2
```

**When to use:**
- Long pretraining (100k+ steps)
- When validation loss plateaus
- Experimental (less common than standard cosine)

**Key implementation details:**
1. Modulo operation for cycle position
2. Cosine formula for smooth transitions
3. Optional max cycles to prevent infinite cycling

---

### Q7: Explain the relationship between batch size, learning rate, and gradient noise.

**Answer:**

**Gradient estimation:**

True gradient (expectation over full dataset):
$$
\nabla_\theta L = \mathbb{E}_{x \sim \mathcal{D}}[\nabla_\theta \ell(x, \theta)]
$$

SGD approximation (batch of size $B$):
$$
\hat{\nabla}_\theta L = \frac{1}{B} \sum_{i=1}^B \nabla_\theta \ell(x_i, \theta)
$$

**Gradient noise (variance):**
$$
\text{Var}(\hat{\nabla}_\theta L) \propto \frac{\sigma^2}{B}
$$

Where $\sigma^2$ is intrinsic gradient variance.

**Key insight:** Larger batch → lower gradient noise.

**Implications for learning rate:**

1. **Small batches (B = 32):**
   - High gradient noise
   - Need small LR to avoid oscillation
   - LR ≈ 1e-5 to 1e-4

2. **Large batches (B = 4096):**
   - Low gradient noise
   - Can use large LR for faster progress
   - LR ≈ 1e-3 to 5e-3

**Linear scaling rule:**
$$
\alpha_{\text{new}} = \alpha_{\text{base}} \times \frac{B_{\text{new}}}{B_{\text{base}}}
$$

**Example:**
- Baseline: LR = 1e-4, batch size = 256
- Increase batch to 2048 (8× larger)
- New LR: $1e-4 \times 8 = 8e-4$

**Why this works:**

With batch size $B$ and LR $\alpha$:
$$
\theta_{t+1} = \theta_t - \alpha \cdot \frac{1}{B} \sum_{i=1}^B \nabla \ell(x_i, \theta_t)
$$

Effective step size: $\frac{\alpha}{B} \sum \nabla \ell$

To keep effective step constant when increasing $B$, increase $\alpha$ proportionally.

**Limitations:**

Linear scaling breaks down for very large batches (>8k):
- Generalization gap (large batch → worse test performance)
- Sharp minima (fast convergence to poor local minimum)

**Solution for mega-batches:**
- **Square root scaling:** $\alpha \propto \sqrt{B}$
- **Adaptive batch size:** Increase batch size during training
- **Better optimizers:** LAMB (Layer-wise Adaptive Moments)

**Practical example:**

```python
# Baseline
batch_size = 256
lr = 1e-4

# Scale to 2048 (8×)
batch_size_new = 2048
lr_new = lr * (batch_size_new / batch_size)  # 8e-4

# But also increase warmup proportionally
warmup_steps_new = warmup_steps * (batch_size_new / batch_size)
```

**Follow-up:** Why does generalization degrade with large batches?  
**Answer:** Large batches converge to sharp minima (narrow valleys) which generalize poorly. Small batches add noise that helps find flat minima (wide valleys) with better generalization. This is an active research area ("generalization gap").

---

## 8. Recent Updates (2024-2025)

### 8.1 LION Optimizer Adoption

**LION** gaining traction for vision-language models:

```python
from lion_pytorch import LION

optimizer = LION(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.99),
    weight_decay=0.1,
)
```

**Performance:** Matches or beats AdamW on some benchmarks (CLIP, image generation)

**Status:** Still experimental for pure LLMs; promising for multimodal models

### 8.2 Schedule-Free Optimizers

**New paradigm:** No learning rate schedule needed.

```python
from schedulefree import AdamWScheduleFree

optimizer = AdamWScheduleFree(
    model.parameters(),
    lr=1e-3,  # Fixed LR, no schedule
)

# Training mode
optimizer.train()
for batch in dataloader:
    loss = model(batch).loss
    loss.backward()
    optimizer.step()

# Evaluation mode (averages weights)
optimizer.eval()
with torch.no_grad():
    eval_loss = model(eval_batch).loss
```

**Idea:** Maintains exponential moving average of weights; evaluation uses averaged weights.

**Benefits:**
- Simpler (no schedule to tune)
- Competitive with cosine schedules

**Paper:** "The Road Less Scheduled" (Defazio et al., 2024)

### 8.3 Sophia Optimizer (Second-Order)

**Sophia** (Second-order Clipped Stochastic Optimization):

```python
from sophia import SophiaG

optimizer = SophiaG(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    rho=0.04,  # Curvature EMA
)
```

**Key innovation:** Uses Hessian diagonal estimate for better step sizes.

**Performance:** 2× faster convergence on some LLM pretraining tasks.

**Caveat:** Requires more memory (Hessian estimates), still under evaluation.

**Paper:** "Sophia: A Scalable Stochastic Second-order Optimizer" (Liu et al., 2023)

---

## 9. Quick Reference

### Optimizer Choice

| Scenario | Recommended Optimizer | Settings |
|----------|----------------------|----------|
| **Standard pretraining** | AdamW | lr=3e-4, β₁=0.9, β₂=0.95, wd=0.1 |
| **Fine-tuning** | AdamW | lr=1e-5, β₁=0.9, β₂=0.999, wd=0.01 |
| **T5/encoder-decoder** | Adafactor | lr=1e-3, scale_parameter=True |
| **Memory constrained** | 8-bit AdamW | Same as AdamW |
| **Experimental** | LION | lr=1e-4, β₁=0.9, β₂=0.99 |

### Schedule Choice

| Scenario | Recommended Schedule | Settings |
|----------|---------------------|----------|
| **Pretraining (known steps)** | Cosine with warmup | warmup=2%, min_lr=10% of max |
| **Fine-tuning (<10k steps)** | Linear with warmup | warmup=10% |
| **Quick experiments** | Constant with warmup | warmup=500-1000 steps |
| **Unknown total steps** | Inverse sqrt | warmup=2000 steps |
| **Long training** | Cosine with restarts | T_0=10k, T_mult=2 |

### Gradient Clipping

| Model Size | Recommended max_norm |
|------------|---------------------|
| <1B | 1.0 |
| 1-10B | 1.0 |
| 10-100B | 0.3-1.0 |
| >100B | 0.3 |
| **Unstable training** | **0.1-0.5** |

### Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Loss → NaN early | LR too high, no warmup | Reduce LR, add warmup |
| Loss → NaN late | Exploding gradients | Add/reduce grad clipping |
| Slow convergence | LR too low | Increase LR, check schedule |
| Training unstable | Bad schedule | Use cosine, increase warmup |
| Loss plateaus | LR too low late | Slower decay (higher min_lr) |

---

## 10. Further Reading

- **AdamW Paper:** [Decoupled Weight Decay Regularization (2017)](https://arxiv.org/abs/1711.05101)
- **Learning Rate Schedules:** [SGDR: Stochastic Gradient Descent with Warm Restarts (2016)](https://arxiv.org/abs/1608.03983)
- **Batch Size Scaling:** [Accurate, Large Minibatch SGD (2017)](https://arxiv.org/abs/1706.02677)
- **LION:** [Symbolic Discovery of Optimization Algorithms (2023)](https://arxiv.org/abs/2302.06675)
- **Sophia:** [Sophia: A Scalable Stochastic Second-order Optimizer (2023)](https://arxiv.org/abs/2305.14342)
- **Schedule-Free:** [The Road Less Scheduled (2024)](https://arxiv.org/abs/2405.15682)
- **Chinchilla Scaling Laws:** [Training Compute-Optimal LLMs (2022)](https://arxiv.org/abs/2203.15556)
