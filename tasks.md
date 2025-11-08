# Tasks: LSTM Frequency Extraction System

## Dataset Generation

### Task 1.1: Implement Signal Generation
- [ ] Create `generate_single_sinusoid()` function
  - Parameters: frequency, amplitude, phase, sampling_rate, duration
  - Returns: numpy array of signal samples
  
- [ ] Create `generate_noisy_signals()` function
  - Generate 4 sinusoidal components with f = [1, 3, 5, 7] Hz
  - Random amplitudes: A_i ~ U(0.8, 1.2)
  - Random phases: φ_i ~ U(0, 2π)
  - Sampling rate: Fs = 1000 Hz
  - Duration: 10 seconds
  - Returns: mixed signal S(t) and individual clean components

### Task 1.2: Add Noise
- [ ] Implement Gaussian noise addition
  - Configurable noise level (SNR or standard deviation)
  - Add to mixed signal only, not to clean targets

### Task 1.3: Create Dataset Structure
- [ ] Implement dataset class compatible with PyTorch DataLoader
  - Input: [S[t], C1, C2, C3, C4] where C is one-hot encoded
  - Target: clean sinusoid value at time t for selected frequency
  - Support batching and shuffling

### Task 1.4: Generate Train/Test Splits
- [ ] Generate training dataset with random seed #1
  - Multiple signal instances with different random A_i and φ_i
  - Sufficient samples for training (e.g., 1000 signal instances)
  
- [ ] Generate test dataset with random seed #2
  - Different random parameters than training
  - Sufficient samples for evaluation (e.g., 200 signal instances)

### Task 1.5: Data Validation
- [ ] Verify signal properties (frequency, amplitude range)
- [ ] Check dataset shapes and dimensions
- [ ] Visualize sample signals to confirm correctness

---

## Model Implementation

### Task 2.1: Define LSTM Architecture
- [ ] Create `FrequencyExtractorLSTM` class
  - Input size: 5 (signal + 4 one-hot indicators)
  - Hidden size: configurable (start with 64)
  - Output size: 1 (predicted clean signal value)
  - Number of LSTM layers: configurable (start with 1)

### Task 2.2: Implement State Management
- [ ] Add methods to reset hidden and cell states
  - `reset_hidden_state()` method
  - Call between different signal samples
  
- [ ] Implement forward pass with sequence length L=1
  - Process one time step at a time
  - Update internal states

### Task 2.3: Model Configuration
- [ ] Create configuration file or class for hyperparameters
  - Hidden size
  - Number of layers
  - Dropout (if needed)
  - Learning rate
  - Batch size
  - Number of epochs

---

## Training Pipeline

### Task 3.1: Setup Training Loop
- [ ] Implement `train_epoch()` function
  - Iterate through training data
  - Reset LSTM states between samples
  - Compute loss (MSE)
  - Backpropagation and optimization step
  - Return average training loss

### Task 3.2: Setup Validation Loop
- [ ] Implement `validate_epoch()` function
  - Iterate through test data
  - No gradient computation
  - Reset LSTM states between samples
  - Compute MSE on test set
  - Return average test loss

### Task 3.3: Training Configuration
- [ ] Use MSE loss function
- [ ] Use Adam optimizer
- [ ] Implement learning rate scheduling (optional)
- [ ] Set number of epochs (e.g., 100)

### Task 3.4: Monitoring and Checkpointing
- [ ] Track MSE_train and MSE_test per epoch
- [ ] Save best model checkpoint based on test MSE
- [ ] Log training progress
- [ ] Implement early stopping (optional)

### Task 3.5: Reproducibility
- [ ] Set random seeds for PyTorch, NumPy, Python
- [ ] Document all hyperparameters
- [ ] Save training configuration with model

---

## Evaluation

### Task 4.1: Model Evaluation
- [ ] Load best model checkpoint
- [ ] Compute final MSE_train and MSE_test
- [ ] Generate predictions for visualization

### Task 4.2: Per-Frequency Analysis
- [ ] Compute MSE for each frequency separately
- [ ] Compare performance across frequencies
- [ ] Identify any frequency-specific issues

### Task 4.3: Signal Reconstruction
- [ ] Generate full signal reconstructions for sample instances
- [ ] Extract all 4 frequency components
- [ ] Compare with ground truth signals

---

## Visualization

### Task 5.1: Training Curves
- [ ] Plot MSE_train vs. epochs
- [ ] Plot MSE_test vs. epochs
- [ ] Show both on same plot for comparison

### Task 5.2: Signal Reconstruction Plots
- [ ] For each frequency (1, 3, 5, 7 Hz):
  - Plot predicted vs. target sinusoid
  - Show time series (e.g., first 2 seconds)
  - Display MSE value on plot
  
- [ ] Create 2x2 subplot for all frequencies

### Task 5.3: Error Analysis
- [ ] Plot prediction error over time
- [ ] Histogram of errors
- [ ] Error statistics (mean, std, max)

### Task 5.4: Comparison Visualization
- [ ] Show noisy input signal
- [ ] Show all 4 extracted components
- [ ] Show ground truth components
- [ ] Create comprehensive comparison figure

---

## Documentation

### Task 6.1: Code Documentation
- [ ] Add docstrings to all functions and classes
- [ ] Include type hints
- [ ] Add inline comments for complex logic

### Task 6.2: README
- [ ] Create comprehensive README.md
  - Project overview
  - Installation instructions
  - Usage examples
  - Results summary

### Task 6.3: Results Documentation
- [ ] Document final MSE values
- [ ] Include visualization images
- [ ] Analyze model performance
- [ ] Discuss limitations and future work

---

## Testing and Validation

### Task 7.1: Unit Tests
- [ ] Test signal generation functions
- [ ] Test dataset class
- [ ] Test model forward pass
- [ ] Test state reset functionality

### Task 7.2: Integration Tests
- [ ] Test full training pipeline
- [ ] Test evaluation pipeline
- [ ] Verify reproducibility

### Task 7.3: Sanity Checks
- [ ] Verify model can overfit small dataset (sanity check)
- [ ] Check gradient flow
- [ ] Validate loss decreases during training

---

## Deliverables Checklist

- [ ] `data_generator.py` - Dataset generation
- [ ] `model.py` - LSTM model definition
- [ ] `train.py` - Training script
- [ ] `evaluate.py` - Evaluation script
- [ ] `plot_results.py` - Visualization utilities
- [ ] `config.py` or `config.yaml` - Configuration
- [ ] `requirements.txt` - Python dependencies
- [ ] `README.md` - Project documentation
- [ ] Trained model checkpoint
- [ ] Result plots and figures
- [ ] Training logs

---

## Priority Order

1. **High Priority (Core Functionality)**
   - Tasks 1.1-1.4: Dataset generation
   - Tasks 2.1-2.2: Model implementation
   - Tasks 3.1-3.4: Training pipeline
   - Task 5.2: Basic visualization

2. **Medium Priority (Validation)**
   - Task 1.5: Data validation
   - Task 4.1-4.3: Evaluation
   - Task 5.1: Training curves
   - Task 6.2: README

3. **Low Priority (Enhancement)**
   - Task 5.3-5.4: Advanced visualization
   - Task 7.1-7.3: Testing
   - Task 6.1, 6.3: Detailed documentation

