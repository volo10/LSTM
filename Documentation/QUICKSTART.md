# Quick Start Guide

Get started with the LSTM Frequency Extraction system in minutes!

## Installation

```bash
# Navigate to project directory
cd /Users/bvolovelsky/Desktop/LLM/LSTM2

# Install dependencies
pip3 install torch numpy matplotlib tqdm pyyaml
```

## Quick Test

Verify everything works:

```bash
python3 test_system.py
```

You should see all tests pass with ✓ marks.

## Full Pipeline

### 1. Train the Model (~10-30 minutes)

```bash
python3 train.py
```

Expected output:
- Progress bars for each epoch
- Training and test loss printed
- Best model saved to `outputs/models/best_model.pth`
- Training history saved to `outputs/logs/`

The training will stop early if the model stops improving (patience=20 epochs).

### 2. Evaluate the Model

```bash
python3 evaluate.py
```

This will:
- Load the best model
- Compute metrics on train and test sets
- Reconstruct sample signals
- Save results to `outputs/results/`

Look for the generalization ratio - it should be between 0.8 and 1.2 for good performance.

### 3. Visualize Results

```bash
python3 plot_results.py
```

This creates 5 plots in `outputs/plots/`:
1. **training_curves.png** - Loss over epochs
2. **signal_reconstruction.png** - Predicted vs. target signals
3. **per_frequency_mse.png** - Performance by frequency
4. **error_distribution.png** - Error histograms
5. **error_over_time.png** - Error time series

## Understanding the Results

### Good Model Performance

✓ **MSE < 0.05** - Low error on both train and test
✓ **Test/Train ratio ≈ 1.0** - Good generalization
✓ **Similar per-frequency MSE** - Consistent across frequencies
✓ **Visual match** - Predicted signals match targets

### If Results Are Poor

❌ **High MSE (>0.1)**: May need longer training or different hyperparameters
❌ **Test >> Train**: Overfitting - reduce model complexity
❌ **Test << Train**: Unusual - check data generation
❌ **One frequency fails**: Check that frequency in data

## Key Concepts

### State Management
The LSTM maintains its internal state (memory) across time steps within the same signal:
- **Within instance**: State is preserved, allowing temporal learning
- **Between instances**: State is reset to prevent information leakage

This is handled automatically in the code - you don't need to manage it manually.

### Architecture
```
Input: [S[t], C1, C2, C3, C4]
  ↓
LSTM (hidden_size=64)
  ↓
Output: predicted clean signal
```

The one-hot encoding [C1, C2, C3, C4] tells the model which frequency to extract.

## Customization

### Change Hyperparameters

Edit `config.yaml`:

```yaml
model:
  hidden_size: 128        # Increase for more capacity
  num_layers: 2           # Add more LSTM layers

training:
  learning_rate: 0.0005   # Adjust learning rate
  num_epochs: 200         # Train longer
```

### Different Frequencies

Edit `config.yaml`:

```yaml
data:
  frequencies: [2.0, 4.0, 6.0]  # Custom frequencies
```

Note: You must also update `model.input_size` to match: `len(frequencies) + 1`

### More Training Data

Edit `config.yaml`:

```yaml
data:
  num_train_instances: 2000   # More signals
  duration: 20.0              # Longer signals
```

## Troubleshooting

### Python not found
Use `python3` instead of `python`:
```bash
python3 train.py
```

### Out of memory
Reduce training instances in `config.yaml`:
```yaml
data:
  num_train_instances: 500
```

### Model not converging
Try:
1. Reduce learning rate: `learning_rate: 0.0005`
2. Increase model size: `hidden_size: 128`
3. Train longer: `num_epochs: 200`

### Import errors
Make sure you're in the correct directory and have installed dependencies:
```bash
cd /Users/bvolovelsky/Desktop/LLM/LSTM2
pip3 install -r requirements.txt
```

## Next Steps

After getting good results:

1. **Read the full README.md** for detailed documentation
2. **Explore planning.md** to understand the approach
3. **Check prd.md** for complete specifications
4. **Experiment** with different hyperparameters
5. **Modify** the code for your own applications

## Getting Help

If something doesn't work:

1. Run `python3 test_system.py` to isolate the issue
2. Check that all files are present (use `ls -la`)
3. Verify Python version: `python3 --version` (should be 3.7+)
4. Check PyTorch installation: `python3 -c "import torch; print(torch.__version__)"`

## Project Files

**Main Scripts:**
- `train.py` - Train the model
- `evaluate.py` - Evaluate performance
- `plot_results.py` - Create visualizations
- `test_system.py` - Quick system test

**Core Modules:**
- `model.py` - LSTM architecture
- `data_generator.py` - Dataset generation

**Documentation:**
- `README.md` - Full documentation
- `QUICKSTART.md` - This file
- `planning.md` - Project plan
- `prd.md` - Requirements
- `claude.md` - AI assistant guide

**Configuration:**
- `config.yaml` - Settings
- `requirements.txt` - Dependencies

Enjoy exploring LSTM-based frequency extraction! 🎵

