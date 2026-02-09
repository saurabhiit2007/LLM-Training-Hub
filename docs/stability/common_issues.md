## 2 Numerical Stability

### 2.4 Loss Spikes and Divergence Diagnosis

#### Common Causes of Loss Spikes

- Learning rate too high
- Insufficient warmup
- Numerical overflow in FP16
- Poor initialization
- Data outliers or corrupted batches

---

#### Diagnosis Checklist

- Check learning rate schedule and warmup length
- Monitor gradient norms
- Enable gradient clipping
- Inspect loss scaling behavior
- Compare FP16 vs BF16 runs
- Verify data preprocessing and labels

---


#### Practical Debugging Tips

- Reduce learning rate and re-run
- Increase warmup steps
- Switch from FP16 to BF16 if possible
- Enable anomaly detection for NaNs and Infs
- Log per-layer gradient norms

---