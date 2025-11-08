# Claude Project Definition: LSTM Frequency Extraction

## Role

You are an AI coding assistant inside Cursor. Your goal is to help build an LSTM-based frequency extraction model following the Product Requirements Document (PRD).

## Project Overview

This project implements an LSTM neural network that learns to extract individual frequency components from noisy mixed signals. Given a mixed signal S(t) containing 4 sinusoidal components (1Hz, 3Hz, 5Hz, 7Hz) plus noise, the LSTM should predict the clean sinusoid for a specified frequency.

## Key Specifications

**Input:** [S[t], C1, C2, C3, C4]
- S[t]: noisy mixed signal sample
- C1-C4: one-hot encoded frequency indicator

**Output:** Predicted clean sinusoid value for selected frequency

**Model:** LSTM with sequence length L=1
- State is maintained across time steps within the same signal instance
- State is reset only between different signal instances
- This allows the LSTM to use its memory to learn temporal patterns

**Metrics:** MSE_train and MSE_test (should be approximately equal)

## Development Workflow

Work step by step through the following phases:

### Phase 1: Dataset Creation (`data_generator.py`)

1. **Implement signal generation functions:**
   - `generate_single_sinusoid(frequency, amplitude, phase, sampling_rate, duration)`
   - `generate_noisy_signals(num_samples, frequencies, sampling_rate, duration, noise_std, seed)`

2. **Create dataset class:**
   - Inherit from `torch.utils.data.Dataset`
   - Return input [S[t], C1, C2, C3, C4] and target y[t]
   - Support all 4 frequencies

3. **Generate train/test splits:**
   - Training data: seed #1
   - Test data: seed #2
   - Ensure different random amplitudes and phases

4. **Validation:**
   - Visualize sample signals
   - Verify shapes and dimensions
   - Check frequency content

### Phase 2: Model Implementation (`model.py`)

1. **Define LSTM architecture:**
   ```python
   class FrequencyExtractorLSTM(nn.Module):
       - Input size: 5
       - Hidden size: configurable (e.g., 64)
       - Output size: 1
       - LSTM layers: 1 or more
   ```

2. **Implement state management:**
   - `reset_hidden_state()` method
   - Initialize h_t and c_t to zeros
   - Call only between different signal instances (not between time steps)
   - Maintain state across time steps within the same instance

3. **Forward pass:**
   - Process one time step at a time (L=1)
   - Maintain and update internal states for next time step
   - State persists within the same signal instance
   - Return predicted value

### Phase 3: Training (`train.py`)

1. **Setup training loop:**
   - Iterate through epochs
   - For each batch:
     - Reset LSTM states only when moving to a new signal instance
     - Maintain state across time steps within the same instance
     - Forward pass
     - Compute MSE loss
     - Backward pass and optimize
   - Track training loss

2. **Setup validation loop:**
   - No gradient computation
   - Reset states between samples
   - Compute MSE on test set
   - Track test loss

3. **Configuration:**
   - Loss: MSE (Mean Squared Error)
   - Optimizer: Adam
   - Learning rate: ~0.001 (tune as needed)
   - Batch size: 32-128
   - Epochs: 50-200

4. **Checkpointing:**
   - Save best model based on test MSE
   - Log training progress
   - Save training history

### Phase 4: Evaluation (`evaluate.py`)

1. **Load best model:**
   - Load checkpoint
   - Set to evaluation mode

2. **Compute metrics:**
   - Final MSE_train
   - Final MSE_test
   - Per-frequency MSE

3. **Generate predictions:**
   - Full signal reconstructions
   - All 4 frequency components
   - Sample instances for visualization

### Phase 5: Visualization (`plot_results.py`)

1. **Training curves:**
   - Plot MSE_train vs. epochs
   - Plot MSE_test vs. epochs
   - Show both on same plot

2. **Signal reconstruction:**
   - For each frequency (1, 3, 5, 7 Hz):
     - Plot predicted vs. target
     - Show time series (e.g., first 2 seconds)
     - Display MSE on plot
   - Create 2x2 subplot

3. **Error analysis:**
   - Prediction error over time
   - Error histogram
   - Error statistics

## Implementation Guidelines

### Code Quality
- Use type hints for function signatures
- Add docstrings to all functions and classes
- Follow PEP 8 style guidelines
- Keep functions focused and modular

### Reproducibility
- Set random seeds at the start of each script:
  ```python
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  ```
- Document all hyperparameters
- Save configuration with model checkpoints

### Error Handling
- Validate input dimensions
- Check for NaN/Inf in loss values
- Handle edge cases gracefully

### Documentation
- Clear README with usage examples
- Inline comments for complex logic
- Document assumptions and limitations

## Success Verification

The implementation is successful when:

1. **Training converges:** Loss decreases smoothly
2. **Generalization:** MSE_train ≈ MSE_test (ratio ~0.8-1.2)
3. **Visual quality:** Predicted signals match ground truth
4. **Consistency:** Similar performance across all frequencies
5. **Reproducibility:** Same results with same random seed

## Common Pitfalls to Avoid

1. **State leakage:** Forgetting to reset LSTM states between samples
2. **Data leakage:** Using same random seed for train and test
3. **Dimension mismatch:** Incorrect input/output shapes
4. **Overfitting:** Large gap between train and test MSE
5. **Underfitting:** High MSE on both train and test

## Debugging Checklist

If model doesn't learn:
- [ ] Verify dataset generates correct signals
- [ ] Check input/output dimensions match model
- [ ] Confirm loss is computed correctly
- [ ] Verify gradients are flowing (not zero/NaN)
- [ ] Try simpler case (e.g., 2 frequencies, no noise)
- [ ] Check learning rate (too high/low)
- [ ] Verify state reset is working

## File Structure

```
LSTM2/
├── claude.md              # This file
├── planning.md            # High-level plan
├── tasks.md              # Task breakdown
├── prd.md                # Requirements
├── data_generator.py     # Dataset generation
├── model.py              # LSTM model
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── plot_results.py       # Visualization
├── config.yaml           # Configuration
├── requirements.txt      # Dependencies
├── README.md             # User documentation
├── outputs/              # Results directory
│   ├── models/          # Saved checkpoints
│   ├── plots/           # Generated figures
│   └── logs/            # Training logs
└── L2-homework.pdf       # Original assignment
```

## Next Steps

Start with Phase 1 (Dataset Creation). Once you have working signal generation and can visualize the data, proceed to Phase 2 (Model Implementation), and so on.

Ask clarifying questions if any requirements are unclear. Focus on getting a working end-to-end pipeline first, then optimize and refine.

## Notes

- This is a supervised learning problem with synthetic data
- The LSTM learns to act as a frequency-selective filter
- Sequence length L=1 means we process time steps individually but maintain state
- The one-hot encoding tells the model which frequency to extract
- Success depends on proper state management and sufficient training data

