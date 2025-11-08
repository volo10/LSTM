# Planning Document: LSTM Frequency Extraction System

## Objective

Develop an LSTM network that receives a **noisy mixed signal S(t)** composed of several sinusoidal components and learns to **extract each clean component frequency** (e.g., 1Hz, 3Hz, 5Hz, 7Hz) as output.

## Problem Statement

Given a mixed signal:
```
S(t) = A₁·sin(2π·f₁·t + φ₁) + A₂·sin(2π·f₂·t + φ₂) + A₃·sin(2π·f₃·t + φ₃) + A₄·sin(2π·f₄·t + φ₄) + noise
```

Where:
- f = [1, 3, 5, 7] Hz (fixed frequencies)
- A_i ~ U(0.8, 1.2) (random amplitudes)
- φ_i ~ U(0, 2π) (random phases)
- Sampling rate: Fs = 1000 Hz
- Duration: 10 seconds

The LSTM should learn to extract individual clean sinusoidal components given a one-hot indicator specifying which frequency to extract.

## Phases

### Phase 1: Dataset Creation
**Goal:** Generate synthetic training and test datasets with proper randomization.

**Key Tasks:**
- Implement signal generation with random amplitudes and phases
- Create mixed signals by summing all components
- Add Gaussian noise to mixed signals
- Generate clean target signals for each frequency
- Split into train/test datasets using different random seeds
- Ensure reproducibility

**Deliverables:**
- `data_generator.py` with signal generation functions
- Train dataset (seed #1)
- Test dataset (seed #2)

### Phase 2: Model Design
**Goal:** Design LSTM architecture suitable for frequency extraction.

**Architecture:**
- **Input:** 5-dimensional vector `[S[t], C1, C2, C3, C4]`
  - S[t]: noisy mixed signal sample at time t
  - C1, C2, C3, C4: one-hot encoded frequency indicator
- **Hidden Layer:** LSTM with configurable hidden size
- **Output:** Single scalar value representing the predicted clean sinusoid at time t
- **State Management:** 
  - Maintain hidden state (h_t) and cell state (c_t) across time steps within the same signal instance
  - Reset state only between different signal instances
  - This allows the LSTM to learn temporal patterns using its memory

**Key Considerations:**
- Sequence length L = 1 (process one time step at a time)
- Internal state is maintained across time steps within the same signal instance
- State is reset only between different signal instances
- This allows LSTM to use its memory to learn temporal and sequential patterns
- Simple architecture to focus on learning frequency-specific patterns

**Deliverables:**
- `model.py` with LSTM class definition

### Phase 3: Training & Evaluation
**Goal:** Train the model and evaluate its performance.

**Training Strategy:**
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam (typical choice for LSTM)
- Batch processing with state resets
- Monitor both training and test MSE
- Save best model checkpoint based on validation performance

**Evaluation Metrics:**
- MSE_train: Performance on training data
- MSE_test: Performance on test data
- Visual comparison: Predicted vs. ground truth signals

**Deliverables:**
- `train.py` for model training
- `evaluate.py` for model evaluation
- Saved model checkpoints

### Phase 4: Visualization & Analysis
**Goal:** Visualize results and verify model performance.

**Visualizations:**
- Predicted vs. target sinusoids for each frequency
- MSE values displayed on plots
- Training curves (loss over epochs)
- Per-frequency performance comparison

**Deliverables:**
- `plot_results.py` with visualization functions
- Generated plots showing reconstruction quality

## Success Metrics

1. **Generalization:** MSE_train ≈ MSE_test
   - Indicates the model generalizes well to unseen data
   - Small gap suggests no overfitting

2. **Visual Quality:** Predicted signals closely match ground truth
   - Sinusoidal shape preserved
   - Correct frequency and phase
   - Minimal noise in predictions

3. **Per-Frequency Performance:** Consistent MSE across all frequencies
   - No bias toward specific frequencies
   - All components extracted equally well

## Technical Considerations

### Reproducibility
- Set random seeds for PyTorch, NumPy, and Python
- Document all hyperparameters
- Version control for code and configurations

### Computational Efficiency
- Batch processing where possible
- GPU acceleration if available
- Efficient data loading

### Code Quality
- Modular design with clear separation of concerns
- Comprehensive documentation
- Type hints for better code clarity
- Unit tests for critical functions

## Milestones

1. **Week 1:** Dataset generation and validation
2. **Week 2:** Model implementation and initial training
3. **Week 3:** Hyperparameter tuning and optimization
4. **Week 4:** Final evaluation and documentation

## Risk Mitigation

**Risk:** Model fails to learn frequency separation
- **Mitigation:** Start with simpler 2-frequency case, gradually increase complexity

**Risk:** Overfitting to training data
- **Mitigation:** Use proper train/test split, monitor both metrics, apply regularization if needed

**Risk:** Numerical instability
- **Mitigation:** Normalize inputs, use gradient clipping, monitor loss values

## Future Extensions

- Support for variable number of frequencies
- Real-time frequency extraction
- Handling of time-varying frequencies
- Noise robustness analysis
- Comparison with traditional signal processing methods (FFT, filters)

