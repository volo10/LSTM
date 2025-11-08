# Product Requirements Document: LSTM Frequency Extraction System

## 1. Purpose

Train an LSTM neural network that can isolate and extract pure frequency components from noisy mixed signals using supervised learning. The system should demonstrate the ability to learn frequency-specific patterns and generalize to unseen signal instances.

## 2. Background

Traditional signal processing uses Fourier transforms and filters to extract frequency components. This project explores a machine learning approach where an LSTM learns to perform frequency extraction through supervised training on labeled data.

## 3. Inputs

### 3.1 Signal Input
- **Format:** Scalar value S[t] representing the noisy mixed signal at time step t
- **Composition:** Sum of 4 sinusoidal components plus Gaussian noise
- **Sampling Rate:** 1000 Hz
- **Duration:** 10 seconds per signal instance
- **Total Samples per Instance:** 10,000 time steps

### 3.2 Frequency Indicator
- **Format:** One-hot encoded vector [C1, C2, C3, C4]
- **Dimension:** 4 (one for each frequency component)
- **Values:** Binary (0 or 1)
- **Purpose:** Indicates which frequency component to extract

### 3.3 Combined Input
- **Format:** 5-dimensional vector [S[t], C1, C2, C3, C4]
- **S[t]:** Noisy mixed signal value
- **C1-C4:** One-hot frequency indicator

## 4. Outputs

### 4.1 Predicted Signal
- **Format:** Scalar value y_t
- **Meaning:** Predicted clean sinusoid value at time t for the selected frequency
- **Range:** Approximately [-1.2, 1.2] (based on amplitude range)

### 4.2 Full Reconstruction
- **Format:** Time series of predicted values
- **Length:** 10,000 samples (10 seconds at 1000 Hz)
- **Components:** Can generate all 4 frequency components by varying the one-hot indicator

## 5. Signal Specifications

### 5.1 Frequency Components
- **f₁ = 1 Hz** (lowest frequency)
- **f₂ = 3 Hz**
- **f₃ = 5 Hz**
- **f₄ = 7 Hz** (highest frequency)

### 5.2 Signal Parameters
- **Amplitudes:** A_i ~ Uniform(0.8, 1.2) for each component
- **Phases:** φ_i ~ Uniform(0, 2π) for each component
- **Noise:** Gaussian noise with configurable standard deviation
- **Mixed Signal:** S(t) = Σ[A_i · sin(2π·f_i·t + φ_i)] + noise

### 5.3 Clean Targets
- **Format:** Individual sinusoids without noise
- **Formula:** y_i(t) = A_i · sin(2π·f_i·t + φ_i)
- **Purpose:** Ground truth for supervised learning

## 6. Constraints

### 6.1 Model Constraints
- **Sequence Length:** L = 1 (process one time step at a time)
- **State Management:** 
  - Internal LSTM state (h_t, c_t) is maintained across time steps within the same signal instance
  - State is reset only between different signal instances
  - This allows the LSTM to learn temporal patterns using its memory
- **Architecture:** Simple LSTM without complex preprocessing

### 6.2 Data Constraints
- **Training Data:** Generated with random seed #1
- **Test Data:** Generated with random seed #2 (different from training)
- **No Overlap:** Training and test datasets have different random parameters
- **Minimum Samples:** Sufficient for training and evaluation (suggested: 1000 train, 200 test instances)

### 6.3 Computational Constraints
- **Framework:** PyTorch
- **Hardware:** Should run on standard CPU/GPU
- **Training Time:** Reasonable training time (< 1 hour on standard hardware)

## 7. Evaluation Metrics

### 7.1 Primary Metric: Mean Squared Error (MSE)

**MSE_train:**
```
MSE_train = (1/N_train) Σ (y_predicted - y_target)²
```

**MSE_test:**
```
MSE_test = (1/N_test) Σ (y_predicted - y_target)²
```

**Success Criterion:** MSE_train ≈ MSE_test
- Indicates good generalization
- Small gap suggests no overfitting
- Typical acceptable range: ratio between 0.8 and 1.2

### 7.2 Secondary Metrics

**Per-Frequency MSE:**
- Compute MSE separately for each frequency (1, 3, 5, 7 Hz)
- Verify consistent performance across frequencies

**Visual Quality:**
- Qualitative assessment of predicted vs. target signals
- Sinusoidal shape preservation
- Correct frequency and phase alignment

### 7.3 Reporting Requirements
- Report both MSE_train and MSE_test
- Include per-frequency breakdown
- Provide visualization of reconstructed signals
- Document training curves (loss over epochs)

## 8. Deliverables

### 8.1 Code Components

**data_generator.py**
- Functions for signal generation
- Dataset class for PyTorch
- Train/test split generation

**model.py**
- LSTM model class definition
- Forward pass implementation
- State reset functionality

**train.py**
- Training loop implementation
- Validation loop
- Checkpoint saving
- Logging

**evaluate.py**
- Model evaluation on test set
- Metric computation
- Result saving

**plot_results.py**
- Training curve visualization
- Signal reconstruction plots
- Error analysis plots

**config.py or config.yaml**
- Hyperparameter configuration
- Model architecture parameters
- Training parameters

**requirements.txt**
- Python package dependencies
- Version specifications

### 8.2 Documentation

**README.md**
- Project overview
- Installation instructions
- Usage examples
- Results summary

**planning.md**
- High-level approach
- Phases and milestones

**tasks.md**
- Specific actionable tasks
- Task tracking

**claude.md**
- Project definition for AI assistant
- Step-by-step guidance

### 8.3 Results

**Model Checkpoint**
- Saved trained model (best_model.pth)
- Training configuration

**Visualizations**
- Training curves (loss vs. epochs)
- Predicted vs. target signals for each frequency
- Error distribution plots

**Metrics Report**
- Final MSE_train and MSE_test values
- Per-frequency performance
- Training statistics

## 9. Success Criteria

### 9.1 Functional Requirements
✓ Model successfully trains without errors
✓ Loss decreases during training
✓ Model generates predictions for all frequencies
✓ State reset mechanism works correctly

### 9.2 Performance Requirements
✓ MSE_train < 0.1 (target, adjust based on noise level)
✓ MSE_test / MSE_train ratio between 0.8 and 1.2
✓ Consistent performance across all 4 frequencies
✓ Visual inspection shows clear sinusoidal reconstruction

### 9.3 Quality Requirements
✓ Code is modular and well-documented
✓ Results are reproducible (fixed random seeds)
✓ Visualizations are clear and informative
✓ Documentation is comprehensive

## 10. Non-Goals

- Real-time processing (not required)
- Variable number of frequencies (fixed at 4)
- Time-varying frequencies (frequencies are constant)
- Deployment to production (research/educational project)
- Comparison with traditional methods (focus on LSTM approach)
- Hyperparameter optimization (use reasonable defaults)

## 11. Future Enhancements

- Support for variable number of frequency components
- Handling of time-varying frequencies
- Real-time frequency extraction
- Noise robustness analysis across different SNR levels
- Comparison with FFT and traditional filtering methods
- Extension to real-world audio signals
- Multi-channel signal processing

## 12. Dependencies

### 12.1 Required Libraries
- PyTorch (>= 1.9.0)
- NumPy (>= 1.20.0)
- Matplotlib (>= 3.3.0)
- PyYAML (for configuration, optional)

### 12.2 Optional Libraries
- TensorBoard (for advanced logging)
- Seaborn (for enhanced visualizations)
- SciPy (for signal processing utilities)

## 13. Timeline

**Phase 1 (Dataset):** 2-3 days
**Phase 2 (Model):** 1-2 days
**Phase 3 (Training):** 2-3 days
**Phase 4 (Evaluation):** 1-2 days

**Total Estimated Time:** 1-2 weeks

## 14. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Model fails to learn | High | Start with simpler 2-frequency case |
| Overfitting | Medium | Proper train/test split, monitor metrics |
| Numerical instability | Medium | Input normalization, gradient clipping |
| Insufficient training data | Medium | Generate more signal instances |
| Poor generalization | High | Ensure diverse training examples |

## 15. Acceptance Criteria

The project is considered complete when:

1. All code components are implemented and functional
2. Model trains successfully and converges
3. MSE_train ≈ MSE_test (generalization achieved)
4. Visualizations clearly show frequency extraction
5. Documentation is complete and clear
6. Code is reproducible with fixed random seeds
7. All deliverables are present and organized

