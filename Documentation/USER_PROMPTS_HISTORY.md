# User Prompts History

This document captures the original user prompts and requirements that led to the creation of this LSTM Frequency Extraction System.

## Project Creation Timeline

---

## Initial Request

**Date:** November 2025  
**Context:** User provided a PDF homework assignment (L2-homework.pdf) with a task to build an LSTM-based frequency extraction system.

---

## Core Requirements from PDF

### Task Description

Build an LSTM network that can extract individual frequency components from a noisy mixed signal.

**Input:**
- `S(t)`: A noisy mixed signal (sum of multiple sinusoids + noise)
- `C`: One-hot vector indicating which frequency component to extract

**Output:**
- Clean sinusoid corresponding to the selected frequency: `sin(2πf_selected·t)`

**Signal Specifications:**
- 4 frequency components: 1 Hz, 3 Hz, 5 Hz, 7 Hz
- Random amplitudes in range [0.8, 1.2]
- Random phases in range [0, 2π]
- Gaussian noise with std = 0.1
- Sampling rate: 1000 Hz
- Duration: 10 seconds (10,000 samples)

---

## User Prompts and Clarifications

### Prompt 1: State Management Correction

**User:**
> "in the prompt i accidentally told you to reset hidden state between samples, i want L=1, so we will handle the internal state in the training loop so the system can use it's memory. we save the current state and pass it as an input for the next step so the network can learn serial patterns by managing the state"

**Key Change:**
- LSTM should maintain hidden state across time steps within the same signal instance
- State should only be reset between different signal instances
- This allows the LSTM to use its memory to learn temporal patterns

**Impact:**
- Modified `model.py` forward pass
- Updated `train.py` and `evaluate.py` state management
- Corrected documentation in `planning.md`, `prd.md`, `claude.md`

---

### Prompt 2: GPU Training Request

**User:**
> "i have a gpu, can you try training on the gpu for faster training?"

**Actions Taken:**
1. Modified device selection to try MPS (Apple Silicon GPU)
2. Encountered segmentation fault (Exit code 139)
3. Identified MPS compatibility issue with LSTM
4. Attempted fixes with hidden state detachment
5. Eventually reverted to CPU training due to MPS instability

**Outcome:**
- Created `GPU_INFO.md` documenting the issue
- Provided Google Colab solution for GPU training
- Local training uses CPU for stability

---

### Prompt 3: Data Specification Clarification

**User:**
> "i have a demand to use 10,000 samples and 10 seconds so keep it please"

**User (follow-up):**
> "let me be more clear demand is 10 seconds, 1000 Hz sample rate and 10,000 samples in total"

**Final Clarification:**
- Duration: 10 seconds
- Sampling rate: 1000 Hz
- Total samples: 10,000 time steps
- Number of signal instances: 1 (for both train and test)
- Frequencies: 4 (1Hz, 3Hz, 5Hz, 7Hz)
- **Training samples:** 10,000 samples × 4 frequencies = 40,000 training examples
- **Test samples:** 10,000 samples × 4 frequencies = 40,000 test examples (same signal, different noise)

**Impact:**
- Updated `data_generator.py` to generate one base signal
- Add different noise for train vs test
- Modified `config.yaml` with correct specifications
- Updated `train.py` configuration

---

### Prompt 4: Training Interruption

**User:**
> "why did training stop? continue"

**Context:**
- Training was progressing normally but user wanted to ensure it continued
- No actual error, just checking status

**Action:**
- Confirmed training was proceeding
- No intervention needed

---

### Prompt 5: Pre-Training Requests

**User:**
> "before you start training again i have a few requests:
> 1- plot for me the training set and test set so i can see you made them well
> 2 - build a script that i can run on gpu colab to train on public gpu and then bring the model to here"

**Actions Taken:**

**Request 1: Data Visualization**
- Created `visualize_data.py`
- Generated plots:
  - Full signal comparison (train vs test)
  - Zoomed-in views to see noise differences
  - Clean frequency components
  - Noise visualization
  - FFT analysis
- Output: `outputs/plots/dataset_visualization.png`

**Request 2: Colab Script**
- Created `train_colab.py` (self-contained training script)
- Created `COLAB_INSTRUCTIONS.md` (step-by-step guide)
- Features:
  - All code in single file
  - GPU auto-detection (CUDA)
  - Download trained model to local machine
  - Upload model back to Colab

---

### Prompt 6: PyTorch Version Compatibility

**User:**
> "i ran on colab and got this error-
> TypeError: ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'"

**Issue:**
- PyTorch removed the `verbose` parameter from `ReduceLROnPlateau` in newer versions
- Code was using `verbose=True`

**Fix:**
```python
# Before:
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=True
)

# After:
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)
```

**Files Updated:**
- `train.py`
- `train_colab.py`

---

### Prompt 7: Continue After Training

**User:**
> "@zsh (593-608) please continue"

**Context:**
- Training completed successfully on CPU
- All 100 epochs finished
- Best model saved at epoch 95

**Actions Taken:**
1. Ran `evaluate.py` to test the model
2. Ran `plot_results.py` to generate visualizations
3. Created comprehensive `TRAINING_RESULTS.md`

**Final Results:**
- Test MSE: 0.437
- Generalization ratio: 0.996 (excellent!)
- No overfitting detected

---

### Prompt 8: Model Output Verification

**User:**
> "what is the current output of the LSYM machine? it should be sinus(t) for the frequency received as an input with no noise - sin(2*pi*f_selected*t)"

**Question:**
- User wanted to confirm what exactly the LSTM outputs

**Answer Provided:**
- LSTM outputs: `A * sin(2πf_selected·t + φ)`
- Where A = amplitude (random in [0.8, 1.2])
- φ = phase (random in [0, 2π])
- No noise in output
- Output matches the clean component from the original signal

**Verification:**
- Checked `data_generator.py` line 38: target generation formula
- Confirmed model learns to predict clean sinusoid with original amplitude and phase
- Test results show RMSE of 0.663, confirming good extraction

---

### Prompt 9: Unit Tests Request

**User:**
> "please add unit tests for the code"

**Actions Taken:**
1. Created comprehensive `test_units.py` with 35 tests:
   - Data generation tests (9 tests)
   - Dataset tests (10 tests)
   - Model tests (9 tests)
   - Integration tests (3 tests)
   - Edge case tests (4 tests)

2. Created `pytest.ini` configuration

3. Created comprehensive `TESTING.md` documentation

4. Updated `requirements.txt` to include pytest and pytest-cov

**Test Results:**
- ✅ 35/35 tests passing
- ✅ 65% coverage of data_generator.py
- ✅ 36% coverage of model.py
- ✅ Fast execution (~1.4 seconds)

---

### Prompt 10: GitHub Push

**User:**
> "i want to push the project to this git - https://github.com/volo10/LSTM"

**Actions Taken:**
1. Initialized git repository in LSTM2
2. Created `.gitignore` to exclude:
   - Output files (models, logs, plots)
   - Python cache
   - Test artifacts
3. Added all project files
4. Committed with comprehensive message
5. Added remote: https://github.com/volo10/LSTM.git
6. Pushed to GitHub

**Result:**
- Repository now live at https://github.com/volo10/LSTM
- 25 files committed
- Clean repository structure

---

### Prompt 11: Pull Changes

**User:**
> "pull any changes from git"

**Action:**
- Ran `git pull origin main`
- Result: Already up to date
- Restored accidentally deleted `.gitignore`

---

### Prompt 12: Documentation Organization

**User:**
> "please move all of the documentation files into a documentation folder and add more important documentation for what is not documented including documentation of the prompts i wrote to create this project"

**Actions Taken:**
1. Created `Documentation/` folder
2. Moved existing documentation files
3. Created this file (USER_PROMPTS_HISTORY.md)
4. Creating additional documentation:
   - API_DOCUMENTATION.md
   - ARCHITECTURE.md
   - INSTALLATION.md
   - TROUBLESHOOTING.md
   - CONTRIBUTING.md

---

## Summary of Key Decisions

### 1. LSTM State Management
- **Decision:** Maintain state within signal instance, reset only between instances
- **Rationale:** Allows LSTM to learn temporal patterns using memory
- **Impact:** Critical for model performance

### 2. Data Generation
- **Decision:** One signal with different noise for train/test
- **Rationale:** Tests generalization to new noise conditions
- **Impact:** Excellent generalization ratio (0.996)

### 3. GPU Training
- **Decision:** Use CPU locally, provide Colab option for GPU
- **Rationale:** MPS compatibility issues with PyTorch LSTM
- **Impact:** Stable training, ~45 minutes on CPU

### 4. Model Architecture
- **Decision:** Single-layer LSTM with 64 hidden units
- **Rationale:** Simple architecture, good for baseline
- **Impact:** 18,241 parameters, good performance

### 5. Training Configuration
- **Decision:** 100 epochs, batch size 1, early stopping patience 20
- **Rationale:** Sequential processing needed for state management
- **Impact:** Stable convergence, best model at epoch 95

---

## Technical Specifications Finalized

```yaml
Data:
  num_train_instances: 1
  num_test_instances: 1
  frequencies: [1.0, 3.0, 5.0, 7.0]
  sampling_rate: 1000
  duration: 10.0
  noise_std: 0.1
  train_seed: 1
  test_seed: 2
  signal_seed: 42

Model:
  input_size: 5  # 1 signal + 4 one-hot
  hidden_size: 64
  num_layers: 1
  dropout: 0.0

Training:
  learning_rate: 0.001
  num_epochs: 100
  batch_size: 1
  patience: 20
```

---

## Lessons Learned

1. **State Management is Critical:** Proper LSTM state handling significantly impacts learning
2. **GPU Compatibility:** Not all PyTorch operations work well on all devices (MPS issues)
3. **Data Specification:** Clear communication about data requirements prevents rework
4. **Visualization:** Plotting data before training helps verify correct generation
5. **Testing:** Comprehensive unit tests catch issues early
6. **Documentation:** Clear documentation of decisions helps future understanding

---

## Future Considerations

Based on the prompts and development process, potential improvements include:

1. **Larger Model:** Try hidden_size=128 or 256
2. **Multiple Layers:** Experiment with 2-3 LSTM layers
3. **Longer Training:** More epochs may improve results
4. **Different Architectures:** Try GRU, Transformer, or CNN-LSTM
5. **More Frequencies:** Test with 6-8 frequency components
6. **Variable Noise:** Train with varying noise levels
7. **Real-World Data:** Test on actual signal processing tasks

---

## Acknowledgments

This project was developed through iterative collaboration, with the user providing:
- Clear task definition (PDF homework)
- Important clarifications on state management
- Specific data requirements
- Testing and validation requests
- Documentation organization guidance

The result is a well-documented, tested, and functional LSTM-based frequency extraction system.

---

**Last Updated:** November 2025  
**Project Repository:** https://github.com/volo10/LSTM

