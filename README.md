# LSTM Frequency Extraction System

A PyTorch implementation of an LSTM neural network that learns to extract individual frequency components from noisy mixed signals through supervised learning.

---

## 📚 Documentation

**Complete documentation is available in the [`Documentation/`](Documentation/) folder:**

- **[Quick Start](Documentation/QUICKSTART.md)** - Get running in 5 minutes
- **[Installation Guide](Documentation/INSTALLATION.md)** - Detailed setup instructions
- **[Architecture](Documentation/ARCHITECTURE.md)** - System design and components
- **[API Reference](Documentation/API_DOCUMENTATION.md)** - Complete API documentation
- **[Testing Guide](Documentation/TESTING.md)** - How to run and write tests
- **[Troubleshooting](Documentation/TROUBLESHOOTING.md)** - Common issues and solutions
- **[Contributing](Documentation/CONTRIBUTING.md)** - How to contribute
- **[Training Results](Documentation/TRAINING_RESULTS.md)** - Latest training results
- **[User Prompts History](Documentation/USER_PROMPTS_HISTORY.md)** - Project creation history

**→ [Full Documentation Index](Documentation/INDEX.md)**

---

## Overview

This project implements an LSTM-based system that can isolate pure frequency components (1Hz, 3Hz, 5Hz, 7Hz) from noisy mixed signals. The model learns to act as a frequency-selective filter by processing time-series data with a one-hot encoded frequency indicator.

### Key Features

- **Supervised Learning**: Trains on synthetic signals with known ground truth
- **State Management**: LSTM maintains state across time steps within each signal instance, resets only between instances
- **Temporal Learning**: Uses LSTM memory to learn sequential patterns
- **Comprehensive Testing**: 35+ unit tests with >85% coverage
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Reproducible**: Fixed random seeds for consistent results
- **Well-Documented**: Extensive documentation and API reference
- **GPU Support**: Google Colab integration for GPU training

## Project Structure

```
LSTM2/
├── Core Implementation
│   ├── data_generator.py        # Dataset generation
│   ├── model.py                 # LSTM model architecture
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   ├── visualize_data.py        # Data visualization
│   ├── plot_results.py          # Result visualization
│   └── create_assignment_plots.py  # Assignment-specific graphs
│
├── Testing
│   ├── test_units.py            # Unit tests (35 tests)
│   ├── test_system.py           # System integration test
│   └── pytest.ini               # Pytest configuration
│
├── Configuration
│   ├── config.yaml              # Hyperparameters
│   └── requirements.txt         # Dependencies
│
├── Documentation/               # Complete documentation
│   ├── INDEX.md                 # Documentation index
│   ├── QUICKSTART.md
│   ├── INSTALLATION.md
│   ├── ARCHITECTURE.md
│   ├── API_DOCUMENTATION.md
│   ├── TESTING.md
│   ├── TROUBLESHOOTING.md
│   ├── CONTRIBUTING.md
│   ├── TRAINING_RESULTS.md
│   ├── COLAB_INSTRUCTIONS.md
│   ├── GPU_INFO.md
│   ├── USER_PROMPTS_HISTORY.md
│   ├── PROJECT_SUMMARY.md
│   ├── prd.md
│   ├── planning.md
│   ├── tasks.md
│   └── claude.md
│
├── Google Colab
│   └── train_colab.py           # Self-contained Colab script
│
├── Outputs/ (gitignored)
│   ├── models/                  # Trained model checkpoints
│   ├── logs/                    # Training logs
│   ├── results/                 # Evaluation results
│   └── plots/                   # Visualizations
│       └── assignment/          # Assignment-specific graphs
│
└── Other
    ├── README.md                # This file
    ├── L2-homework.pdf          # Original assignment
    ├── .gitignore
    └── LICENSE
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone or download this repository

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install torch numpy matplotlib tqdm pyyaml
```

## Usage

### Quick Start

Run the complete pipeline:

```bash
# 1. Train the model
python train.py

# 2. Evaluate the model
python evaluate.py

# 3. Generate visualizations
python plot_results.py

# 4. Generate assignment-specific graphs (optional)
python create_assignment_plots.py
```

### Step-by-Step Guide

#### 1. Data Generation

The data generator creates synthetic signals automatically during training. To test data generation separately:

```bash
python data_generator.py
```

This will:
- Generate sample training and test datasets
- Display signal properties
- Visualize sample signals

#### 2. Model Training

Train the LSTM model:

```bash
python train.py
```

Training features:
- Automatic dataset generation
- Progress bars for each epoch
- Early stopping with patience
- Learning rate scheduling
- Best model checkpointing
- Training history logging

Expected output:
```
Using device: cpu
Creating datasets...
Training samples: 40,000,000
Test samples: 8,000,000

Creating model...
======================================================================
LSTM Frequency Extractor Model Summary
======================================================================
Input size:        5
Hidden size:       64
Number of layers:  1
Output size:       1
Total parameters:  17,985
======================================================================

Starting training on cpu...
Epoch 1/100 | Train Loss: 0.234567 | Test Loss: 0.245678 | LR: 0.001000
  → Saved best model (test loss: 0.245678)
...
```

#### 3. Model Evaluation

Evaluate the trained model:

```bash
python evaluate.py
```

This will:
- Load the best model checkpoint
- Evaluate on training and test sets
- Compute MSE, RMSE, MAE metrics
- Calculate per-frequency performance
- Reconstruct sample signals
- Save results to JSON

Expected output:
```
======================================================================
Evaluating on TRAINING set...
======================================================================

Training Metrics:
  MSE:  0.012345
  RMSE: 0.111111
  MAE:  0.098765
  Max Error: 0.543210

Per-Frequency MSE (Training):
  1.0 Hz: 0.011234
  3.0 Hz: 0.012345
  5.0 Hz: 0.013456
  7.0 Hz: 0.012567

======================================================================
Evaluating on TEST set...
======================================================================

Test Metrics:
  MSE:  0.013456
  RMSE: 0.116012
  MAE:  0.102345
  Max Error: 0.567890

Per-Frequency MSE (Test):
  1.0 Hz: 0.012345
  3.0 Hz: 0.013456
  5.0 Hz: 0.014567
  7.0 Hz: 0.013678

======================================================================
Generalization Analysis:
======================================================================
Test MSE / Train MSE ratio: 1.0900
✓ Good generalization (ratio between 0.8 and 1.2)
```

#### 4. Visualization

Generate all plots:

```bash
python plot_results.py
```

This creates:
- **Training curves**: Loss over epochs
- **Signal reconstruction**: Predicted vs. target signals
- **Per-frequency metrics**: MSE comparison
- **Error distribution**: Histogram of prediction errors
- **Error over time**: Time-series error analysis

All plots are saved to `outputs/plots/`.

#### 5. Assignment-Specific Graphs

Generate the graphs specifically requested in the assignment document:

```bash
python create_assignment_plots.py
```

This creates:
- **f2_overlay_2sec.png**: Overlay plot for f₂ (3 Hz) showing:
  - Target signal (no noise)
  - Noisy input signal S(t)
  - LSTM output
- **f2_overlay_5sec.png**: Extended 5-second view of f₂
- **all_frequencies_comparison.png**: Separate plots for each frequency (f₁–f₄) comparing Target vs Predicted signals

All assignment plots are saved to `outputs/plots/assignment/`.

### Testing Individual Components

Test the model architecture:

```bash
python model.py
```

Test data generation:

```bash
python data_generator.py
```

## Configuration

Edit `config.yaml` to customize:

```yaml
data:
  num_train_instances: 1000    # Number of training signals
  num_test_instances: 200      # Number of test signals
  frequencies: [1.0, 3.0, 5.0, 7.0]  # Frequency components
  noise_std: 0.1               # Noise level

model:
  hidden_size: 64              # LSTM hidden units
  num_layers: 1                # Number of LSTM layers

training:
  num_epochs: 100              # Maximum epochs
  learning_rate: 0.001         # Initial learning rate
  patience: 20                 # Early stopping patience
```

## Model Architecture

### Input
- **Dimension**: 5
- **Format**: `[S[t], C1, C2, C3, C4]`
  - `S[t]`: Noisy mixed signal value at time t
  - `C1-C4`: One-hot encoded frequency indicator

### Network
- **LSTM Layer**: Configurable hidden size (default: 64)
- **Output Layer**: Fully connected layer
- **Output**: Single scalar (predicted clean sinusoid value)

### Key Features
- Sequence length L=1 (process one time step at a time)
- Hidden state maintained across time steps within each signal instance
- Hidden state reset only between different signal instances
- This allows the LSTM to learn temporal patterns using its memory
- Gradient clipping for stability

## Signal Specifications

### Mixed Signal
```
S(t) = A₁·sin(2π·f₁·t + φ₁) + A₂·sin(2π·f₂·t + φ₂) + 
       A₃·sin(2π·f₃·t + φ₃) + A₄·sin(2π·f₄·t + φ₄) + noise
```

Where:
- **Frequencies**: f = [1, 3, 5, 7] Hz
- **Amplitudes**: A_i ~ Uniform(0.8, 1.2)
- **Phases**: φ_i ~ Uniform(0, 2π)
- **Sampling Rate**: 1000 Hz
- **Duration**: 10 seconds
- **Noise**: Gaussian with configurable std

## Evaluation Metrics

### Primary Metric
- **MSE (Mean Squared Error)**: Main performance indicator
- **Success Criterion**: MSE_train ≈ MSE_test (ratio 0.8-1.2)

### Secondary Metrics
- **RMSE**: Root mean squared error
- **MAE**: Mean absolute error
- **Max Error**: Maximum prediction error
- **Per-Frequency MSE**: Performance breakdown by frequency

## Results

After training, you'll find:

### Model Checkpoints
- `outputs/models/best_model.pth`: Best model based on validation loss
- `outputs/models/final_model_TIMESTAMP.pth`: Final model state

### Logs
- `outputs/logs/training_history_TIMESTAMP.json`: Complete training history

### Results
- `outputs/results/evaluation_results.json`: Evaluation metrics
- `outputs/results/reconstructed_signals.npz`: Sample reconstructions

### Plots
- `outputs/plots/training_curves.png`: Training/test loss curves
- `outputs/plots/signal_reconstruction.png`: Predicted vs. target signals
- `outputs/plots/per_frequency_mse.png`: Per-frequency performance
- `outputs/plots/error_distribution.png`: Error histograms
- `outputs/plots/error_over_time.png`: Time-series error analysis

### Assignment Plots
- `outputs/plots/assignment/f2_overlay_2sec.png`: Overlay plot for f₂ (3 Hz)
- `outputs/plots/assignment/f2_overlay_5sec.png`: Extended 5-second view of f₂
- `outputs/plots/assignment/all_frequencies_comparison.png`: All frequencies comparison

## Troubleshooting

### Model doesn't converge
- Reduce learning rate
- Increase hidden size
- Check data generation (run `data_generator.py`)
- Verify loss is being computed correctly

### Out of memory
- Reduce `num_train_instances` in config
- Use smaller `hidden_size`
- Ensure batch_size is 1

### Poor generalization (high test/train ratio)
- Increase training data
- Add dropout
- Reduce model complexity
- Check for data leakage

### NaN loss
- Reduce learning rate
- Check gradient clipping is enabled
- Verify input normalization
- Check for inf/nan in data

## Advanced Usage

### Custom Frequencies

Modify frequencies in `config.yaml`:

```yaml
data:
  frequencies: [2.0, 4.0, 6.0, 8.0]  # Custom frequencies
```

Update `input_size` in model config to match number of frequencies + 1.

### Enhanced Model

Use the v2 model with batch normalization:

```python
from model import FrequencyExtractorLSTMv2

model = FrequencyExtractorLSTMv2(
    input_size=5,
    hidden_size=128,
    num_layers=2,
    dropout=0.1,
    use_batch_norm=True
)
```

### Custom Training Loop

```python
from data_generator import create_datasets
from model import FrequencyExtractorLSTM
from train import train_model

# Create datasets
train_dataset, test_dataset = create_datasets(...)

# Create model
model = FrequencyExtractorLSTM(...)

# Train with custom parameters
history = train_model(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    num_epochs=200,
    learning_rate=0.0005,
    ...
)
```

## Documentation

- **planning.md**: High-level project approach and phases
- **tasks.md**: Detailed task breakdown and checklist
- **prd.md**: Complete product requirements
- **claude.md**: AI assistant project definition

## License

This project is for educational purposes.

## Acknowledgments

Based on the L2 homework assignment for LSTM-based frequency extraction from mixed noisy signals.

## Contact

For questions or issues, please refer to the documentation files or create an issue in the repository.

