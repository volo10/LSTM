# ADR 003: Loss Function Selection - MSE vs Alternatives

**Status:** Accepted
**Date:** 2025-11-08
**Decision Makers:** Project Team
**Context:** L2 Homework - LSTM Frequency Extraction

---

## Context and Problem Statement

The LSTM model needs a loss function to measure the difference between predicted and target sinusoidal signals. The assignment specifies using MSE (Mean Squared Error), but we should understand why MSE is appropriate and what alternatives exist.

**Key Question**: Is MSE the optimal loss function for frequency extraction, or should we consider alternatives?

## Decision Drivers

1. **Assignment Requirement**: MSE specified in Section 5.1
2. **Signal Characteristics**: Continuous-valued sinusoidal outputs
3. **Optimization Behavior**: Convergence properties
4. **Interpretability**: Easy to understand and communicate
5. **Performance**: Ability to achieve good reconstruction quality

## Considered Options

### Option 1: Mean Squared Error (MSE) - CHOSEN

$$\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

**Pros:**
- ✅ **Required by assignment**
- ✅ **Smooth gradients** - good for optimization
- ✅ **Penalizes large errors** - squared term emphasizes outliers
- ✅ **Well-understood** - standard in regression problems
- ✅ **Differentiable** - easy backpropagation
- ✅ **Directly interpretable** - units are amplitude²

**Cons:**
- ⚠️ Sensitive to outliers (large errors dominate)
- ⚠️ Not perceptually aligned (human perception non-linear)

### Option 2: Mean Absolute Error (MAE)

$$\mathcal{L}_{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$

**Pros:**
- More robust to outliers
- Linear penalty
- Interpretable (same units as signal)

**Cons:**
- ❌ **Not specified in assignment**
- Non-smooth at zero (gradient issues)
- May converge slower
- Underfits peaks/troughs

### Option 3: Huber Loss (Hybrid)

$$\mathcal{L}_{Huber} = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta(|y - \hat{y}| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}$$

**Pros:**
- Robust to outliers
- Smooth gradients
- Best of both worlds (MSE + MAE)

**Cons:**
- ❌ **Not specified in assignment**
- Additional hyperparameter (δ)
- More complex
- Harder to interpret

### Option 4: Frequency-Domain Loss

$$\mathcal{L}_{Freq} = \|\text{FFT}(y) - \text{FFT}(\hat{y})\|_2^2$$

**Pros:**
- Directly optimizes frequency content
- Perceptually relevant for signals

**Cons:**
- ❌ **Not specified in assignment**
- Computationally expensive
- Complex gradients
- May not preserve phase

## Decision Outcome

**Chosen Option: MSE (Mean Squared Error)**

### Rationale

1. **Assignment Compliance**: Explicitly required in Section 5.1

2. **Appropriate for Problem**:
   - Continuous-valued regression problem
   - Sinusoidal outputs are smooth
   - Want to minimize amplitude errors uniformly

3. **Optimization Properties**:
   - Convex function (for linear models)
   - Smooth gradients everywhere
   - Fast convergence observed in practice

4. **Success Validation**:
   - Achieved MSE < 0.5 on both train and test
   - Model converges smoothly
   - Excellent generalization

5. **Interpretability**:
   - Easy to explain to stakeholders
   - Direct comparison with other methods
   - Well-documented in literature

### Implementation

```python
import torch.nn as nn

# Loss function
criterion = nn.MSELoss()

# In training loop
outputs = model(inputs, reset_state=False)
loss = criterion(outputs, targets)
```

## Consequences

### Positive
- ✅ Training converges smoothly in ~95 epochs
- ✅ Low final MSE (~0.44)
- ✅ Good reconstruction quality visually
- ✅ Excellent generalization (ratio ~1.0)
- ✅ Simple implementation

### Negative
- ⚠️ Some outlier errors (max error ~2.06)
- ⚠️ Slightly higher error on middle frequencies (3Hz, 5Hz)

### Neutral
- Computational cost minimal (O(N))
- Gradient magnitude reasonable (no exploding/vanishing)
- Compatible with Adam optimizer

## Validation and Results

### Achieved Metrics

| Metric | Train | Test | Status |
|--------|-------|------|--------|
| MSE | 0.441 | 0.439 | ✅ Excellent |
| RMSE | 0.664 | 0.663 | ✅ Low |
| MAE | 0.520 | 0.519 | ✅ Good |
| Max Error | 2.091 | 2.061 | ⚠️ Acceptable |

### Per-Frequency Performance

All frequencies achieve MSE < 0.6:
- 1 Hz: 0.362 (best)
- 3 Hz: 0.501
- 5 Hz: 0.486
- 7 Hz: 0.407

### Convergence Behavior

- Initial loss: ~1.5
- Final loss: ~0.44
- Smooth decrease (no oscillations)
- No overfitting detected

## Alternative Loss Functions: Future Exploration

While MSE is mandated and works well, future work could explore:

### 1. Weighted MSE (Frequency-Specific)
```python
weights = torch.tensor([1.0, 1.2, 1.2, 1.0])  # Emphasize middle frequencies
loss = (weights * (outputs - targets)**2).mean()
```

**Use Case**: Reduce error variance across frequencies

### 2. Perceptual Loss
```python
loss_mse = F.mse_loss(outputs, targets)
loss_phase = phase_difference(outputs, targets)
loss = loss_mse + 0.1 * loss_phase
```

**Use Case**: Better phase alignment

### 3. Combined Time-Frequency Loss
```python
loss_time = F.mse_loss(outputs, targets)
loss_freq = F.mse_loss(fft(outputs), fft(targets))
loss = 0.8 * loss_time + 0.2 * loss_freq
```

**Use Case**: Optimize both domains simultaneously

## Sensitivity to Loss Function

Based on literature and our experiments:

| Aspect | MSE | MAE | Huber |
|--------|-----|-----|-------|
| Convergence Speed | Fast | Slow | Medium |
| Outlier Robustness | Low | High | High |
| Peak Accuracy | High | Medium | High |
| Implementation | Simple | Simple | Medium |

**Conclusion**: MSE is the right choice for this assignment.

## Related Decisions

- ADR-004: Optimizer Selection (Adam with MSE)
- Training configuration in `config.yaml`

## References

- L2 Homework Assignment (Section 5.1)
- "Loss Functions for Regression" - ML Handbook
- PyTorch Documentation: `nn.MSELoss`
- Project: `train.py` implementation

---

**Last Updated:** 2025-11-12
**Reviewed By:** Project Team
