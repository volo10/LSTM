# Troubleshooting Guide

Common issues and their solutions for the LSTM Frequency Extraction System.

---

## Table of Contents

- [Installation Issues](#installation-issues)
- [Training Issues](#training-issues)
- [GPU/Device Issues](#gpudevice-issues)
- [Data Issues](#data-issues)
- [Model Issues](#model-issues)
- [Testing Issues](#testing-issues)
- [Performance Issues](#performance-issues)
- [Import Errors](#import-errors)
- [Git Issues](#git-issues)

---

## Installation Issues

### Issue: Python not found

**Error:**
```bash
python: command not found
```

**Solution:**
```bash
# Use python3 instead
python3 --version

# Or create alias
alias python=python3
```

---

### Issue: pip install fails

**Error:**
```
ERROR: Could not install packages due to an EnvironmentError
```

**Solutions:**

1. **Use virtual environment:**
   ```bash
   python3 -m venv lstm_env
   source lstm_env/bin/activate
   pip install -r requirements.txt
   ```

2. **Install with --user flag:**
   ```bash
   pip install --user -r requirements.txt
   ```

3. **Upgrade pip:**
   ```bash
   python3 -m pip install --upgrade pip
   ```

---

### Issue: Permission denied during installation

**Error:**
```
Permission denied: '/usr/local/...'
```

**Solutions:**

1. **Use virtual environment (recommended):**
   ```bash
   python3 -m venv lstm_env
   source lstm_env/bin/activate
   pip install -r requirements.txt
   ```

2. **Use --user flag:**
   ```bash
   pip install --user -r requirements.txt
   ```

3. **Use sudo (not recommended):**
   ```bash
   sudo pip install -r requirements.txt
   ```

---

## Training Issues

### Issue: Training crashes immediately

**Error:**
```
Segmentation fault (core dumped)
Exit code: 139
```

**Cause:** Usually MPS (Apple Silicon GPU) compatibility issue with LSTM.

**Solution:**

1. **Force CPU training:**
   ```python
   # In train.py, change device selection:
   device = 'cpu'  # Force CPU
   ```

2. **Use Google Colab for GPU:**
   - See `COLAB_INSTRUCTIONS.md`
   - Uses CUDA instead of MPS

---

### Issue: Training very slow

**Symptoms:**
- Each epoch takes > 1 minute
- CPU usage is low

**Solutions:**

1. **Check batch size:**
   ```python
   # In train.py
   batch_size = 1  # Must be 1 for state management
   ```
   This is expected and correct for this architecture.

2. **Use GPU (Colab):**
   - Training on GPU is 5-10x faster
   - See `COLAB_INSTRUCTIONS.md`

3. **Reduce data for testing:**
   ```python
   # For quick testing only
   duration = 1.0  # Instead of 10.0
   num_epochs = 10  # Instead of 100
   ```

---

### Issue: Loss becomes NaN

**Error:**
```
Train Loss: nan
Valid Loss: nan
```

**Causes & Solutions:**

1. **Exploding gradients:**
   ```python
   # Already implemented in train.py:
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

2. **Learning rate too high:**
   ```python
   # Reduce learning rate
   learning_rate = 0.0001  # Instead of 0.001
   ```

3. **Check data:**
   ```python
   # Verify data has no NaN/Inf
   print(torch.isnan(inputs).any())
   print(torch.isinf(inputs).any())
   ```

---

### Issue: Model not learning (loss plateaus)

**Symptoms:**
- Loss doesn't decrease after initial epochs
- Validation loss stuck at high value

**Solutions:**

1. **Check state management:**
   ```python
   # Ensure state is reset between instances, not between samples
   if prev_instance_id != current_instance_id:
       model.reset_hidden_state(...)
   ```

2. **Increase model capacity:**
   ```python
   hidden_size = 128  # Instead of 64
   num_layers = 2     # Instead of 1
   ```

3. **Adjust learning rate:**
   ```python
   learning_rate = 0.0001  # Try different values
   ```

4. **Check data quality:**
   ```python
   python3 visualize_data.py
   ```

---

### Issue: Early stopping too aggressive

**Symptoms:**
- Training stops after ~20 epochs
- Loss was still decreasing

**Solution:**
```python
# In train.py, increase patience:
patience = 40  # Instead of 20
```

---

## GPU/Device Issues

### Issue: MPS (Apple Silicon) segmentation fault

**Error:**
```
Segmentation fault: 11
Exit code: 139
```

**Cause:** Known PyTorch LSTM + MPS incompatibility.

**Solutions:**

1. **Use CPU (recommended for local):**
   ```python
   device = 'cpu'
   ```

2. **Use Google Colab with CUDA:**
   ```bash
   # See COLAB_INSTRUCTIONS.md
   ```

3. **Wait for PyTorch update:**
   - This is a known issue
   - Check: https://github.com/pytorch/pytorch/issues

**Workaround Attempted (didn't work):**
```python
# These didn't solve the issue:
# - Detaching hidden state
# - Using different LSTM implementations
# - Reducing batch size
```

---

### Issue: CUDA out of memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size:**
   ```python
   batch_size = 1  # Already minimal
   ```

2. **Reduce model size:**
   ```python
   hidden_size = 32  # Instead of 64
   ```

3. **Clear cache:**
   ```python
   torch.cuda.empty_cache()
   ```

4. **Use gradient accumulation:**
   ```python
   # Accumulate gradients over N steps
   if (batch_idx + 1) % accumulation_steps == 0:
       optimizer.step()
       optimizer.zero_grad()
   ```

---

### Issue: CUDA not available

**Error:**
```python
torch.cuda.is_available()  # Returns False
```

**Solutions:**

1. **Check CUDA installation:**
   ```bash
   nvidia-smi
   nvcc --version
   ```

2. **Reinstall PyTorch with CUDA:**
   ```bash
   # For CUDA 11.8:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Verify GPU is recognized:**
   ```python
   import torch
   print(torch.cuda.device_count())
   print(torch.cuda.get_device_name(0))
   ```

---

## Data Issues

### Issue: "Dataset length is 0"

**Cause:** Incorrect data generation parameters.

**Solution:**
```python
# Check parameters in create_datasets:
num_train_instances = 1  # Must be >= 1
duration = 10.0          # Must be > 0
sampling_rate = 1000     # Must be > 0
frequencies = [1.0, 3.0, 5.0, 7.0]  # Must not be empty
```

---

### Issue: "Input size mismatch"

**Error:**
```
RuntimeError: input.size(-1) must be equal to input_size. Expected 5, got 3
```

**Cause:** Model input_size doesn't match dataset.

**Solution:**
```python
# input_size = 1 + number of frequencies
frequencies = [1.0, 3.0, 5.0, 7.0]  # 4 frequencies
input_size = 1 + len(frequencies)    # = 5

model = FrequencyExtractorLSTM(input_size=input_size, ...)
```

---

### Issue: "Signal looks wrong in visualization"

**Symptoms:**
- Training and test signals are identical
- No visible noise difference

**Check:**
```python
# Verify different seeds:
train_seed = 1  # Different
test_seed = 2   # Different
signal_seed = 42  # Same (correct)

# Check noise_std is not zero:
noise_std = 0.1  # Should be > 0
```

---

## Model Issues

### Issue: "Cannot load model"

**Error:**
```
RuntimeError: Error(s) in loading state_dict
```

**Solutions:**

1. **Check model architecture matches:**
   ```python
   # Model architecture must match saved model
   model = FrequencyExtractorLSTM(
       input_size=5,      # Must match
       hidden_size=64,    # Must match
       num_layers=1,      # Must match
   )
   ```

2. **Load with map_location:**
   ```python
   model.load_state_dict(torch.load('model.pth', map_location='cpu'))
   ```

3. **Check file exists:**
   ```python
   import os
   print(os.path.exists('outputs/models/best_model.pth'))
   ```

---

### Issue: "State shape mismatch"

**Error:**
```
RuntimeError: The size of tensor a (1) must match the size of tensor b (4)
```

**Cause:** Hidden state batch size doesn't match input batch size.

**Solution:**
```python
# Always reset state with correct batch size:
model.reset_hidden_state(batch_size=inputs.size(0), device=device)
```

---

## Testing Issues

### Issue: Tests fail on import

**Error:**
```
ModuleNotFoundError: No module named 'data_generator'
```

**Solutions:**

1. **Run from project root:**
   ```bash
   cd /path/to/LSTM2
   pytest test_units.py
   ```

2. **Add to PYTHONPATH:**
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   pytest test_units.py
   ```

3. **Install in editable mode:**
   ```bash
   pip install -e .
   ```

---

### Issue: Test failures after code changes

**Symptoms:**
- Tests were passing, now failing
- Changed model or data code

**Solutions:**

1. **Update tests to match changes:**
   - If you changed input_size, update tests
   - If you changed frequencies, update tests

2. **Check test expectations:**
   ```python
   # Example: If you changed num_frequencies
   assert input_vec.shape == (1 + len(frequencies),)
   ```

3. **Clear pytest cache:**
   ```bash
   rm -rf .pytest_cache
   pytest test_units.py -v
   ```

---

## Performance Issues

### Issue: High memory usage

**Symptoms:**
- System runs out of RAM
- Process killed

**Solutions:**

1. **Check for memory leaks:**
   ```python
   # Make sure to use torch.no_grad() for evaluation:
   with torch.no_grad():
       outputs = model(inputs)
   ```

2. **Clear variables:**
   ```python
   del large_dataset
   import gc
   gc.collect()
   ```

3. **Monitor memory:**
   ```python
   import psutil
   process = psutil.Process()
   print(f"Memory: {process.memory_info().rss / 1024 ** 2:.2f} MB")
   ```

---

### Issue: Disk space full

**Symptoms:**
- "No space left on device"
- Cannot save models

**Solution:**
```bash
# Remove old outputs:
rm -rf outputs/models/checkpoint_*.pth
rm -rf outputs/logs/*.json

# Keep only best model:
ls -lh outputs/models/

# Clear pytest cache:
rm -rf .pytest_cache htmlcov/
```

---

## Import Errors

### Issue: "No module named 'torch'"

**Solution:**
```bash
# Activate virtual environment:
source lstm_env/bin/activate

# Install PyTorch:
pip install torch
```

---

### Issue: "cannot import name 'FrequencyExtractorLSTM'"

**Solution:**
```python
# Make sure file names are correct:
# It's model.py, not models.py
from model import FrequencyExtractorLSTM
```

---

### Issue: "ImportError: DLL load failed" (Windows)

**Solution:**
```bash
# Reinstall PyTorch:
pip uninstall torch
pip install torch

# Or install Visual C++ Redistributable:
# https://aka.ms/vs/16/release/vc_redist.x64.exe
```

---

## Git Issues

### Issue: Merge conflicts

**Solution:**
```bash
# If you modified files that have updates:
git status
git stash  # Save your changes
git pull origin main
git stash pop  # Reapply your changes
# Resolve conflicts manually
```

---

### Issue: Large files in git

**Error:**
```
remote: error: File outputs/models/best_model.pth is 100.00 MB; this exceeds GitHub's file size limit of 100.00 MB
```

**Solution:**
```bash
# These should be in .gitignore already:
cat .gitignore
# If not, add:
echo "outputs/models/*.pth" >> .gitignore

# Remove from git history:
git rm --cached outputs/models/best_model.pth
git commit -m "Remove large model file"
```

---

## Getting Help

If your issue isn't covered here:

1. **Check Documentation:**
   - README.md
   - API_DOCUMENTATION.md
   - ARCHITECTURE.md
   - GPU_INFO.md

2. **Check GitHub Issues:**
   - https://github.com/volo10/LSTM/issues
   - Search for similar problems

3. **Create New Issue:**
   - Go to: https://github.com/volo10/LSTM/issues/new
   - Provide:
     - Error message (full traceback)
     - Python version
     - PyTorch version
     - OS and version
     - Steps to reproduce
     - What you've already tried

4. **Debug Mode:**
   ```python
   # Add to code for more info:
   import logging
   logging.basicConfig(level=logging.DEBUG)
   
   # Print shapes:
   print(f"Input shape: {inputs.shape}")
   print(f"Output shape: {outputs.shape}")
   
   # Check values:
   print(f"Has NaN: {torch.isnan(inputs).any()}")
   print(f"Has Inf: {torch.isinf(inputs).any()}")
   ```

---

## Quick Diagnostic Commands

```bash
# Check Python version
python3 --version

# Check installed packages
pip list

# Check PyTorch
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check CUDA
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check file structure
ls -la

# Check output directories
ls outputs/models/

# Run system test
python3 test_system.py

# Run unit tests
pytest test_units.py -v

# Check git status
git status

# Check disk space
df -h
```

---

## Error Message Reference

| Error | Likely Cause | Section |
|-------|-------------|---------|
| "command not found" | Python/pip not installed | [Installation](#installation-issues) |
| "Segmentation fault" | MPS GPU issue | [GPU Issues](#gpudevice-issues) |
| "RuntimeError: input.size" | Input size mismatch | [Data Issues](#data-issues) |
| "Loss is nan" | Exploding gradients | [Training Issues](#training-issues) |
| "CUDA out of memory" | Batch size too large | [GPU Issues](#gpudevice-issues) |
| "ModuleNotFoundError" | Import path issue | [Import Errors](#import-errors) |
| "Permission denied" | Installation permissions | [Installation](#installation-issues) |

---

**Last Updated:** November 2025  
**Version:** 1.0  
**For additional help:** Create an issue at https://github.com/volo10/LSTM/issues

