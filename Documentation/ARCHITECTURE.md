# System Architecture

This document describes the architecture and design of the LSTM Frequency Extraction System.

---

## Table of Contents

- [Overview](#overview)
- [System Components](#system-components)
- [Data Flow](#data-flow)
- [Model Architecture](#model-architecture)
- [State Management](#state-management)
- [Training Pipeline](#training-pipeline)
- [File Structure](#file-structure)
- [Design Patterns](#design-patterns)
- [Performance Considerations](#performance-considerations)

---

## Overview

The LSTM Frequency Extraction System is designed to extract individual frequency components from noisy mixed signals using Long Short-Term Memory (LSTM) neural networks.

### Problem Statement

**Input:**
- Mixed signal: `S(t) = Σ A_i * sin(2πf_i * t + φ_i) + noise`
- Frequency selector: One-hot vector `C` indicating which frequency to extract

**Output:**
- Clean sinusoid: `y(t) = A_selected * sin(2πf_selected * t + φ_selected)`

### Key Design Goals

1. **Accurate Extraction:** Minimize reconstruction error for selected frequency
2. **Noise Robustness:** Generalize to different noise conditions
3. **Memory Utilization:** Use LSTM memory to learn temporal patterns
4. **Modularity:** Separate concerns (data, model, training, evaluation)
5. **Reproducibility:** Ensure deterministic results with seeds

---

## System Components

```
┌─────────────────────────────────────────────────────────┐
│                 LSTM Frequency Extraction System        │
└─────────────────────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
    ┌───▼───┐           ┌────▼────┐         ┌────▼────┐
    │ Data  │           │  Model  │         │Training │
    │Gen    │           │         │         │/Eval    │
    └───┬───┘           └────┬────┘         └────┬────┘
        │                    │                   │
┌───────┴────────┐    ┌──────┴──────┐    ┌──────┴──────┐
│ Signal Gen     │    │ LSTM Layer  │    │ Train Loop  │
│ Dataset Class  │    │ FC Layer    │    │ Validation  │
│ Data Loaders   │    │ State Mgmt  │    │ Checkpoints │
└────────────────┘    └─────────────┘    └─────────────┘
```

### Component Descriptions

**1. Data Generation** (`data_generator.py`)
- Generates synthetic sinusoidal signals
- Mixes multiple frequency components
- Adds Gaussian noise
- Creates PyTorch Dataset

**2. Model** (`model.py`)
- LSTM network architecture
- Hidden state management
- Forward pass logic

**3. Training** (`train.py`)
- Training loop with early stopping
- Learning rate scheduling
- Model checkpointing

**4. Evaluation** (`evaluate.py`)
- Performance metrics
- Per-frequency analysis
- Prediction generation

**5. Visualization** (`visualize_data.py`, `plot_results.py`)
- Data visualization
- Training curves
- Result analysis

---

## Data Flow

### Training Data Flow

```
┌──────────────┐
│ Signal Seeds │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────┐
│ Generate Base Signal         │
│ • 4 frequencies (1,3,5,7 Hz) │
│ • Random amplitudes & phases │
│ • 10 seconds, 1000 Hz        │
└──────┬───────────────────────┘
       │
       ├──────────────┬─────────────┐
       ▼              ▼             │
┌────────────┐  ┌───────────┐      │
│ Add Noise  │  │ Add Noise │      │
│ (seed=1)   │  │ (seed=2)  │      │
└─────┬──────┘  └─────┬─────┘      │
      │               │             │
      ▼               ▼             ▼
┌───────────┐   ┌──────────┐  ┌──────────┐
│  Train    │   │   Test   │  │  Clean   │
│  Mixed    │   │  Mixed   │  │Components│
└─────┬─────┘   └─────┬────┘  └─────┬────┘
      │               │             │
      └───────┬───────┴─────────────┘
              ▼
    ┌─────────────────────┐
    │FrequencyExtraction  │
    │      Dataset        │
    └──────────┬──────────┘
               │
               ▼
    ┌──────────────────────┐
    │  DataLoader          │
    │  • Batch size: 1     │
    │  • Sequential order  │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Training Loop       │
    └──────────────────────┘
```

### Inference Data Flow

```
Input: [S(t), one-hot C]
           │
           ▼
    ┌──────────────┐
    │ LSTM Layer   │
    │ (maintains   │
    │  state)      │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ FC Layer     │
    └──────┬───────┘
           │
           ▼
Output: Predicted clean
    sinusoid value
```

---

## Model Architecture

### Network Structure

```
Input Layer                      LSTM Layer                 Output Layer
[5 dims]                         [64 units]                 [1 dim]

┌─────────┐                     ┌──────────┐              ┌─────────┐
│ S(t)    │────┐                │          │              │         │
├─────────┤    │                │          │              │ Linear  │
│ C1      │    ├───────────────▶│  LSTM    │─────────────▶│   FC    │──▶ ŷ(t)
├─────────┤    │                │          │              │         │
│ C2      │    │                │  h_t, c_t│              │ (64→1)  │
├─────────┤    │                │          │              │         │
│ C3      │────┤                │          │              └─────────┘
├─────────┤    │                └──────────┘
│ C4      │────┘                     ▲
└─────────┘                          │
                              State maintained
                              across time steps
```

### Layer Details

**Input Layer:**
- Size: `1 + num_frequencies` (5 for 4 frequencies)
- Components:
  - `S(t)`: Noisy mixed signal value at time t
  - `C`: One-hot vector indicating target frequency

**LSTM Layer:**
- Hidden size: 64 units
- Number of layers: 1
- Sequence length: 1 (processes one time step at a time)
- State persistence: Maintained within instance, reset between instances

**Output Layer:**
- Fully connected layer
- Input: 64 (LSTM hidden size)
- Output: 1 (predicted clean sinusoid value)

### Parameter Count

```python
# LSTM parameters:
# 4 gates * (input_size * hidden_size + hidden_size * hidden_size + hidden_size)
lstm_params = 4 * (5 * 64 + 64 * 64 + 64) = 17,920

# FC parameters:
# hidden_size * output_size + output_size
fc_params = 64 * 1 + 1 = 65

# Total
total_params = 17,920 + 65 = 17,985 (actual: 18,241 due to bias terms)
```

---

## State Management

### Critical Design Decision

The LSTM maintains hidden state `(h_t, c_t)` across time steps **within** the same signal instance, but resets **between** different signal instances.

### Why This Matters

**With State Persistence (Our Approach):**
```
Signal Instance 1:
t=0: [S(0), C] → LSTM(h=0, c=0) → ŷ(0), update state
t=1: [S(1), C] → LSTM(h=h₁, c=c₁) → ŷ(1), update state  ✓ Uses memory
t=2: [S(2), C] → LSTM(h=h₂, c=c₂) → ŷ(2), update state  ✓ Uses memory
...
```

**Without State Persistence (Wrong):**
```
Signal Instance 1:
t=0: [S(0), C] → LSTM(h=0, c=0) → ŷ(0)
t=1: [S(1), C] → LSTM(h=0, c=0) → ŷ(1)  ✗ No memory!
t=2: [S(2), C] → LSTM(h=0, c=0) → ŷ(2)  ✗ No memory!
```

### Implementation

```python
# In training loop:
prev_instance_id = -1

for inputs, targets, instance_ids in dataloader:
    # Reset ONLY when moving to new instance
    if prev_instance_id == -1 or instance_ids[0].item() != prev_instance_id:
        model.reset_hidden_state(batch_size=inputs.size(0), device=device)
        prev_instance_id = instance_ids[0].item()
    
    # Forward pass maintains state
    outputs = model(inputs, reset_state=False)
```

### State Detachment

To prevent backpropagation through time across different instances:

```python
# In model.forward():
if reset_state:
    self.hidden_state = new_hidden
else:
    # Detach to prevent gradient flow to previous instances
    self.hidden_state = (new_hidden[0].detach(), new_hidden[1].detach())
```

---

## Training Pipeline

### Training Process

```
┌─────────────────────┐
│ 1. Load Config      │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ 2. Create Datasets  │
│    • Train (1 inst) │
│    • Test (1 inst)  │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ 3. Initialize Model │
│    • LSTM (64 hid)  │
│    • Random weights │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ 4. Setup Training   │
│    • MSE Loss       │
│    • Adam Optimizer │
│    • LR Scheduler   │
└──────────┬──────────┘
           │
           ▼
   ┌───────────────┐
   │ For each epoch│
   └───────┬───────┘
           │
    ┌──────▼──────┐
    │ Train Phase │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Val Phase  │
    └──────┬──────┘
           │
    ┌──────▼──────────┐
    │ Early Stopping? │
    └──────┬──────────┘
           │
    ┌──────▼──────┐
    │ Save Best   │
    │   Model     │
    └──────┬──────┘
           │
           ▼
┌──────────────────────┐
│ 5. Save Results      │
│    • Best model      │
│    • Training logs   │
│    • Final model     │
└──────────────────────┘
```

### Hyperparameters

```yaml
Learning:
  learning_rate: 0.001
  num_epochs: 100
  batch_size: 1
  optimizer: Adam
  loss: MSELoss

Scheduling:
  scheduler: ReduceLROnPlateau
  mode: min
  factor: 0.5
  patience: 10

Early Stopping:
  patience: 20
  min_delta: 0.0

Regularization:
  gradient_clipping: 1.0
  dropout: 0.0  # Not used with single layer
```

### Training Metrics

Tracked for each epoch:
- Training loss (MSE)
- Validation loss (MSE)
- Learning rate
- Time elapsed

Saved in: `outputs/logs/training_history_TIMESTAMP.json`

---

## File Structure

```
LSTM2/
│
├── Core Implementation
│   ├── data_generator.py      # Data generation and dataset
│   ├── model.py               # LSTM architecture
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   ├── visualize_data.py      # Data visualization
│   └── plot_results.py        # Result visualization
│
├── Testing
│   ├── test_units.py          # Unit tests (35 tests)
│   ├── test_system.py         # System integration test
│   └── pytest.ini             # Pytest configuration
│
├── Configuration
│   ├── config.yaml            # Hyperparameters
│   └── requirements.txt       # Dependencies
│
├── Documentation/
│   ├── README.md              # Main documentation (root)
│   ├── USER_PROMPTS_HISTORY.md
│   ├── API_DOCUMENTATION.md
│   ├── ARCHITECTURE.md (this file)
│   ├── INSTALLATION.md
│   ├── TROUBLESHOOTING.md
│   ├── CONTRIBUTING.md
│   ├── TESTING.md
│   ├── GPU_INFO.md
│   ├── QUICKSTART.md
│   ├── PROJECT_SUMMARY.md
│   ├── TRAINING_RESULTS.md
│   ├── COLAB_INSTRUCTIONS.md
│   ├── planning.md
│   ├── tasks.md
│   ├── prd.md
│   └── claude.md
│
├── Google Colab
│   └── train_colab.py         # Self-contained Colab script
│
├── Outputs/ (gitignored)
│   ├── models/                # Trained models
│   ├── logs/                  # Training logs
│   ├── results/               # Evaluation results
│   └── plots/                 # Visualizations
│
└── Other
    ├── L2-homework.pdf        # Original assignment
    ├── .gitignore             # Git ignore rules
    └── LICENSE                # License file
```

---

## Design Patterns

### 1. Separation of Concerns

Each module has a single, well-defined responsibility:
- `data_generator.py`: Data creation
- `model.py`: Network architecture
- `train.py`: Training logic
- `evaluate.py`: Evaluation logic

### 2. Configuration Management

Centralized configuration in `config.yaml` and inline dicts:

```python
config = {
    'num_train_instances': 1,
    'frequencies': [1.0, 3.0, 5.0, 7.0],
    'hidden_size': 64,
    ...
}
```

### 3. State Pattern

LSTM state management using explicit reset and maintenance:

```python
class FrequencyExtractorLSTM:
    def reset_hidden_state(self, ...):
        # Initialize state
    
    def forward(self, x, reset_state=False):
        # Use or reset state
```

### 4. Factory Pattern

Dataset creation encapsulated in factory function:

```python
def create_datasets(...) -> Tuple[Dataset, Dataset]:
    # Creates and returns train and test datasets
```

### 5. Template Method Pattern

Training and validation share common structure:

```python
def train_epoch(...):
    # Training-specific logic
    # Common: loop, forward, loss

def validate_epoch(...):
    # Validation-specific logic
    # Common: loop, forward, loss
```

---

## Performance Considerations

### Computational Complexity

**Forward Pass:**
- LSTM: `O(4 * hidden_size * (input_size + hidden_size))`
- FC: `O(hidden_size * output_size)`
- Total per sample: `O(hidden_size²)` ≈ O(4096) for hidden_size=64

**Training Time:**
- Samples per epoch: 40,000 (train)
- Time per epoch: ~27 seconds (CPU)
- Total training: ~45 minutes for 100 epochs

### Memory Usage

**Model Parameters:**
- Total: ~18k parameters
- Memory: ~72 KB (float32)

**Batch Processing:**
- Batch size: 1 (required for state management)
- Memory per batch: ~20 bytes (5 floats input)

**Dataset Storage:**
- Training data: 40,000 samples × 5 dims × 4 bytes = ~780 KB
- Test data: ~780 KB
- Total: ~1.5 MB (minimal)

### Optimization Strategies

**Current:**
1. Batch size 1 (required for sequential processing)
2. Gradient clipping (prevents exploding gradients)
3. Learning rate scheduling (ReduceLROnPlateau)
4. Early stopping (prevents overfitting)

**Potential Improvements:**
1. **Larger batches:** Process multiple signal instances in parallel
2. **GPU acceleration:** Use CUDA (MPS has issues)
3. **Model pruning:** Reduce parameter count
4. **Quantization:** Use int8 for inference
5. **ONNX export:** For production deployment

---

## Scalability

### Horizontal Scaling

Process multiple signals in parallel:

```python
# Current: 1 signal instance
batch_size = 1

# Scaled: Multiple instances
batch_size = 8  # 8 different signals
# Requires: Different random seeds per instance
```

### Vertical Scaling

Increase model capacity:

```python
# Current
hidden_size = 64
num_layers = 1

# Scaled
hidden_size = 128  # or 256
num_layers = 2     # or 3
```

### Data Scaling

Handle more frequencies:

```python
# Current
frequencies = [1.0, 3.0, 5.0, 7.0]  # 4 frequencies

# Scaled
frequencies = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]  # 8 frequencies
input_size = 1 + len(frequencies)  # Adjust input size
```

---

## Design Tradeoffs

### Batch Size = 1

**Pros:**
- Enables sequential state management
- Simpler implementation
- Matches problem structure

**Cons:**
- Slower training
- Underutilizes GPU
- Higher overhead

**Decision:** Prioritize correctness over speed for baseline

### Single vs Multiple Layers

**Single Layer (Current):**
- Pros: Faster, fewer parameters, easier to train
- Cons: Limited capacity

**Multiple Layers:**
- Pros: More expressive, better performance
- Cons: Slower, more parameters, harder to train

**Decision:** Start simple, scale if needed

### CPU vs GPU

**CPU (Current):**
- Pros: Stable, no MPS issues, always available
- Cons: Slower (~45 min for 100 epochs)

**GPU (Future):**
- Pros: Much faster (~5-10 min on Colab)
- Cons: MPS compatibility issues, requires CUDA

**Decision:** CPU for development, GPU for production

---

## Extension Points

The architecture is designed to be extensible:

1. **Different Architectures:**
   - Replace LSTM with GRU
   - Add CNN layers for feature extraction
   - Use Transformer architecture

2. **Different Losses:**
   - Replace MSE with MAE
   - Add perceptual loss
   - Use custom frequency-domain loss

3. **Different Data:**
   - Real-world signals (audio, biomedical)
   - Variable-length sequences
   - Multiple simultaneous extractions

4. **Different Training:**
   - Online learning
   - Transfer learning
   - Multi-task learning

---

## Security Considerations

1. **Model Persistence:**
   - Models saved with `torch.save()` - only load from trusted sources
   - No encryption by default - add if needed for production

2. **Data Privacy:**
   - Synthetic data - no privacy concerns
   - For real data: anonymize before processing

3. **Input Validation:**
   - Currently no validation - add for production
   - Check input dimensions, ranges, types

---

## Future Architecture Improvements

1. **Modular Config System:**
   ```python
   from omegaconf import OmegaConf
   config = OmegaConf.load('config.yaml')
   ```

2. **Logging Framework:**
   ```python
   import logging
   logger = logging.getLogger(__name__)
   ```

3. **Experiment Tracking:**
   ```python
   import wandb
   wandb.init(project="lstm-freq-extraction")
   ```

4. **Model Serving:**
   ```python
   # Export to ONNX
   torch.onnx.export(model, ...)
   
   # Serve with TorchServe
   torchserve --start --model-store models/
   ```

---

**Last Updated:** November 2025  
**Version:** 1.0

