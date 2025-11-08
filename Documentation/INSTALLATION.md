# Installation Guide

Complete installation instructions for the LSTM Frequency Extraction System.

---

## Table of Contents

- [Quick Install](#quick-install)
- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [Verification](#verification)
- [Platform-Specific Instructions](#platform-specific-instructions)
- [Troubleshooting Installation](#troubleshooting-installation)

---

## Quick Install

```bash
# 1. Clone the repository
git clone https://github.com/volo10/LSTM.git
cd LSTM

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python3 test_system.py

# 4. Run quick test
pytest test_units.py -v
```

---

## System Requirements

### Minimum Requirements

- **Python:** 3.8 or higher
- **RAM:** 2 GB
- **Storage:** 500 MB
- **OS:** Linux, macOS, or Windows

### Recommended Requirements

- **Python:** 3.10 or higher
- **RAM:** 8 GB or more
- **Storage:** 2 GB
- **GPU:** Optional (CUDA-compatible for faster training)
- **OS:** Linux or macOS

---

## Installation Methods

### Method 1: pip (Recommended)

```bash
# Create and activate virtual environment (recommended)
python3 -m venv lstm_env
source lstm_env/bin/activate  # Linux/macOS
# lstm_env\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### Method 2: Conda

```bash
# Create conda environment
conda create -n lstm_env python=3.10
conda activate lstm_env

# Install PyTorch (choose appropriate version)
# CPU version:
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Or GPU version (CUDA 11.8):
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install numpy matplotlib tqdm pyyaml pytest pytest-cov
```

### Method 3: From Source

```bash
# Clone repository
git clone https://github.com/volo10/LSTM.git
cd LSTM

# Install in development mode
pip install -e .
```

---

## Detailed Installation Steps

### Step 1: Python Installation

#### macOS
```bash
# Using Homebrew
brew install python@3.10

# Verify
python3 --version
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip

# Verify
python3 --version
```

#### Windows
1. Download Python from https://www.python.org/downloads/
2. Run installer
3. ✅ Check "Add Python to PATH"
4. Complete installation
5. Verify in Command Prompt: `python --version`

---

### Step 2: Clone Repository

```bash
# Using HTTPS
git clone https://github.com/volo10/LSTM.git

# Or using SSH
git clone git@github.com:volo10/LSTM.git

# Navigate to project
cd LSTM
```

---

### Step 3: Set Up Virtual Environment

#### Why Use Virtual Environments?
- Isolates project dependencies
- Prevents version conflicts
- Easy to recreate environment

#### Create Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv lstm_env
source lstm_env/bin/activate
```

**Windows:**
```bash
python -m venv lstm_env
lstm_env\Scripts\activate
```

You should see `(lstm_env)` in your terminal prompt.

---

### Step 4: Install Dependencies

#### Core Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### requirements.txt Contents

```txt
torch>=1.9.0
numpy>=1.20.0
matplotlib>=3.3.0
tqdm>=4.60.0
pyyaml>=5.4.0
pytest>=7.0.0
pytest-cov>=3.0.0
```

#### Verify Installation

```python
python3 -c "import torch; print(f'PyTorch {torch.__version__}')"
python3 -c "import numpy; print(f'NumPy {numpy.__version__}')"
```

---

### Step 5: Create Output Directories

```bash
mkdir -p outputs/models
mkdir -p outputs/logs
mkdir -p outputs/results
mkdir -p outputs/plots
```

These directories will store:
- `models/`: Trained model checkpoints
- `logs/`: Training history logs
- `results/`: Evaluation results
- `plots/`: Visualization plots

---

## Verification

### Run System Test

```bash
python3 test_system.py
```

**Expected Output:**
```
======================================================================
LSTM FREQUENCY EXTRACTION - SYSTEM TEST
======================================================================

======================================================================
Testing Data Generation
======================================================================
✓ Training dataset created: 8000 samples
✓ Test dataset created: 4000 samples
...
======================================================================
ALL TESTS PASSED ✓
======================================================================
```

### Run Unit Tests

```bash
pytest test_units.py -v
```

**Expected Output:**
```
============================= test session starts ==============================
...
============================== 35 passed in 1.37s ==============================
```

### Quick Training Test

```bash
# Train for 2 epochs (quick test)
python3 train.py --epochs 2
```

---

## Platform-Specific Instructions

### macOS

#### Apple Silicon (M1/M2/M3)

**Note:** MPS (Metal Performance Shaders) has compatibility issues with LSTM. Use CPU for training.

```bash
# Install dependencies
pip install -r requirements.txt

# Training will automatically use CPU
python3 train.py
```

To use GPU, consider Google Colab (see `COLAB_INSTRUCTIONS.md`).

#### Intel Macs

```bash
# Standard installation
pip install -r requirements.txt
```

---

### Linux

#### Ubuntu/Debian

```bash
# Install system dependencies
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip git

# Install Python packages
pip install -r requirements.txt
```

#### CUDA Setup (for GPU training)

```bash
# Check CUDA version
nvcc --version

# Install PyTorch with CUDA support
# Example for CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU is available
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

### Windows

#### PowerShell/Command Prompt

```bash
# Clone repository
git clone https://github.com/volo10/LSTM.git
cd LSTM

# Create virtual environment
python -m venv lstm_env
lstm_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python test_system.py
```

#### Windows Subsystem for Linux (WSL)

Follow Linux instructions within WSL.

---

## Google Colab Setup

For GPU training without local GPU:

1. Open Google Colab: https://colab.research.google.com/
2. Create new notebook
3. Copy contents of `train_colab.py`
4. Run cells sequentially
5. Download trained model

See `COLAB_INSTRUCTIONS.md` for details.

---

## Docker Installation (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create output directories
RUN mkdir -p outputs/models outputs/logs outputs/results outputs/plots

CMD ["python3", "train.py"]
```

Build and run:

```bash
# Build image
docker build -t lstm-freq-extraction .

# Run container
docker run -v $(pwd)/outputs:/app/outputs lstm-freq-extraction

# Or interactive
docker run -it lstm-freq-extraction bash
```

---

## Troubleshooting Installation

### Problem: "python: command not found"

**Solution:**
```bash
# Try python3 instead
python3 --version

# Or create alias (add to ~/.bashrc or ~/.zshrc)
alias python=python3
```

---

### Problem: "pip: command not found"

**Solution:**
```bash
# macOS/Linux
python3 -m ensurepip --upgrade
python3 -m pip install --upgrade pip

# Or install pip separately
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py
```

---

### Problem: "Permission denied"

**Solution:**
```bash
# Use --user flag
pip install --user -r requirements.txt

# Or fix permissions (not recommended globally)
sudo pip install -r requirements.txt
```

---

### Problem: PyTorch installation fails

**Solution:**
```bash
# Install specific version
pip install torch==2.0.0

# Or use conda
conda install pytorch -c pytorch

# Check compatibility: https://pytorch.org/get-started/locally/
```

---

### Problem: "ModuleNotFoundError: No module named 'torch'"

**Solution:**
```bash
# Ensure virtual environment is activated
source lstm_env/bin/activate  # Linux/macOS
lstm_env\Scripts\activate      # Windows

# Reinstall PyTorch
pip install torch
```

---

### Problem: CUDA not detected

**Solution:**
```bash
# Check CUDA installation
nvidia-smi

# Install correct PyTorch version for your CUDA
# Visit: https://pytorch.org/get-started/locally/

# Example for CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

### Problem: Tests fail on import

**Solution:**
```bash
# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in editable mode
pip install -e .
```

---

## Updating Installation

### Update from Git

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

### Update Dependencies Only

```bash
pip install -r requirements.txt --upgrade
```

### Fresh Install

```bash
# Remove virtual environment
rm -rf lstm_env

# Recreate
python3 -m venv lstm_env
source lstm_env/bin/activate
pip install -r requirements.txt
```

---

## Development Installation

For contributing to the project:

```bash
# Install development dependencies
pip install -r requirements.txt
pip install black flake8 mypy

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install

# Run tests before committing
pytest
```

---

## Uninstallation

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf lstm_env

# Remove project directory
cd ..
rm -rf LSTM

# Or keep project but remove outputs
cd LSTM
rm -rf outputs/
```

---

## Performance Optimization

### For Faster Training

1. **Use GPU:** Follow CUDA setup instructions
2. **Use Colab:** Free GPU access
3. **Reduce epochs:** For quick testing
4. **Increase batch size:** If using multiple signal instances

### For Lower Memory Usage

1. **Reduce hidden size:** In `config.yaml`
2. **Use CPU:** Slight memory advantage
3. **Process shorter sequences:** Reduce duration

---

## Next Steps

After successful installation:

1. **Quick Start:** See `QUICKSTART.md`
2. **Run Training:** `python3 train.py`
3. **Run Tests:** `pytest -v`
4. **Read Documentation:** `Documentation/` folder
5. **Try Visualization:** `python3 visualize_data.py`

---

## Support

If you encounter issues:

1. Check `TROUBLESHOOTING.md`
2. Search existing issues: https://github.com/volo10/LSTM/issues
3. Create new issue with:
   - Error message
   - Python version
   - OS and version
   - Steps to reproduce

---

## Appendix: Version Compatibility

### Tested Configurations

| Python | PyTorch | NumPy | OS | Status |
|--------|---------|-------|----|---------| 
| 3.10 | 2.0.0 | 1.24.0 | macOS 13 | ✅ Works |
| 3.10 | 2.0.0 | 1.24.0 | Ubuntu 22.04 | ✅ Works |
| 3.9 | 1.13.0 | 1.23.0 | Windows 11 | ✅ Works |
| 3.11 | 2.1.0 | 1.25.0 | Ubuntu 22.04 | ✅ Works |
| 3.8 | 1.9.0 | 1.20.0 | macOS 12 | ✅ Works |

### Known Incompatibilities

- ❌ Python < 3.8
- ⚠️ MPS (Apple Silicon) with LSTM (use CPU)
- ⚠️ PyTorch < 1.9.0 (missing features)

---

**Last Updated:** November 2025  
**Version:** 1.0

