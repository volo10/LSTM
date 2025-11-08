# API Documentation

Complete reference for all functions, classes, and methods in the LSTM Frequency Extraction System.

---

## Table of Contents

- [data_generator.py](#data_generatorpy)
- [model.py](#modelpy)
- [train.py](#trainpy)
- [evaluate.py](#evaluatepy)
- [visualize_data.py](#visualize_datapy)
- [plot_results.py](#plot_resultspy)

---

## data_generator.py

Functions and classes for generating synthetic signals and creating datasets.

### Functions

#### `generate_single_sinusoid`

Generates a single sinusoidal signal.

```python
def generate_single_sinusoid(
    frequency: float,
    amplitude: float,
    phase: float,
    sampling_rate: int,
    duration: float
) -> np.ndarray
```

**Parameters:**
- `frequency` (float): Frequency in Hz
- `amplitude` (float): Signal amplitude
- `phase` (float): Phase offset in radians
- `sampling_rate` (int): Sampling rate in Hz
- `duration` (float): Signal duration in seconds

**Returns:**
- `np.ndarray`: Signal samples, shape `(num_samples,)`

**Formula:**
```
signal(t) = amplitude * sin(2π * frequency * t + phase)
```

**Example:**
```python
signal = generate_single_sinusoid(
    frequency=5.0,
    amplitude=1.0,
    phase=0.0,
    sampling_rate=1000,
    duration=1.0
)
# Returns array of 1000 samples
```

---

#### `generate_noisy_signals`

Generates multiple instances of noisy mixed signals.

```python
def generate_noisy_signals(
    num_instances: int,
    frequencies: List[float],
    sampling_rate: int,
    duration: float,
    noise_std: float,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]
```

**Parameters:**
- `num_instances` (int): Number of signal instances to generate
- `frequencies` (List[float]): List of frequency components (e.g., [1.0, 3.0, 5.0, 7.0])
- `sampling_rate` (int): Sampling rate in Hz
- `duration` (float): Signal duration in seconds
- `noise_std` (float): Standard deviation of Gaussian noise
- `seed` (Optional[int]): Random seed for reproducibility

**Returns:**
- Tuple of:
  - `mixed_signals` (np.ndarray): Shape `(num_instances, num_samples)`
  - `clean_components` (np.ndarray): Shape `(num_instances, num_frequencies, num_samples)`

**Process:**
1. For each instance, generates random amplitudes (0.8-1.2) and phases (0-2π)
2. Creates clean sinusoids for each frequency
3. Mixes all components
4. Adds Gaussian noise

**Example:**
```python
mixed, clean = generate_noisy_signals(
    num_instances=10,
    frequencies=[1.0, 3.0, 5.0, 7.0],
    sampling_rate=1000,
    duration=10.0,
    noise_std=0.1,
    seed=42
)
# mixed.shape = (10, 10000)
# clean.shape = (10, 4, 10000)
```

---

#### `create_datasets`

Creates training and test datasets with same signal but different noise.

```python
def create_datasets(
    num_train_instances: int = 1,
    num_test_instances: int = 1,
    frequencies: List[float] = [1.0, 3.0, 5.0, 7.0],
    sampling_rate: int = 1000,
    duration: float = 10.0,
    noise_std: float = 0.1,
    train_seed: int = 1,
    test_seed: int = 2,
    signal_seed: int = 42
) -> Tuple[FrequencyExtractionDataset, FrequencyExtractionDataset]
```

**Parameters:**
- `num_train_instances` (int): Number of training signal instances
- `num_test_instances` (int): Number of test signal instances
- `frequencies` (List[float]): Frequency components to extract
- `sampling_rate` (int): Sampling rate in Hz
- `duration` (float): Signal duration in seconds
- `noise_std` (float): Noise standard deviation
- `train_seed` (int): Seed for training noise
- `test_seed` (int): Seed for test noise
- `signal_seed` (int): Seed for signal generation (same for train/test)

**Returns:**
- Tuple of:
  - `train_dataset` (FrequencyExtractionDataset)
  - `test_dataset` (FrequencyExtractionDataset)

**Key Feature:**
- Uses same base signal for train and test
- Adds different noise for each (tests generalization)

**Example:**
```python
train_dataset, test_dataset = create_datasets(
    num_train_instances=1,
    num_test_instances=1,
    frequencies=[1.0, 3.0, 5.0, 7.0],
    sampling_rate=1000,
    duration=10.0,
    noise_std=0.1,
    train_seed=1,
    test_seed=2,
    signal_seed=42
)
# len(train_dataset) = 1 * 10000 * 4 = 40000
# len(test_dataset) = 1 * 10000 * 4 = 40000
```

---

### Classes

#### `FrequencyExtractionDataset`

PyTorch Dataset for LSTM frequency extraction training.

```python
class FrequencyExtractionDataset(Dataset):
    def __init__(
        self,
        mixed_signals: np.ndarray,
        clean_components: np.ndarray,
        frequencies: List[float]
    )
```

**Attributes:**
- `mixed_signals` (torch.Tensor): Shape `(num_instances, num_samples)`
- `clean_components` (torch.Tensor): Shape `(num_instances, num_frequencies, num_samples)`
- `frequencies` (List[float]): List of frequency values
- `num_instances` (int): Number of signal instances
- `num_samples` (int): Number of time samples per instance
- `num_frequencies` (int): Number of frequency components

**Methods:**

##### `__len__`

```python
def __len__(self) -> int
```

Returns total number of samples: `num_instances * num_samples * num_frequencies`

##### `__getitem__`

```python
def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]
```

**Parameters:**
- `idx` (int): Sample index

**Returns:**
- Tuple of:
  - `input` (torch.Tensor): Shape `(num_frequencies + 1,)` = `[S(t), C1, C2, ..., Cn]`
  - `target` (torch.Tensor): Shape `(1,)` = clean sinusoid value at time t
  - `instance_id` (int): Which signal instance this sample belongs to

**Index Mapping:**
```
instance_id = idx // (num_samples * num_frequencies)
time_step = (idx % (num_samples * num_frequencies)) // num_frequencies
freq_idx = idx % num_frequencies
```

**Example:**
```python
dataset = FrequencyExtractionDataset(mixed, clean, [1.0, 3.0, 5.0, 7.0])
input_vec, target, instance_id = dataset[0]
# input_vec.shape = (5,)  # [S(0), 1, 0, 0, 0]
# target.shape = (1,)     # clean sinusoid for 1Hz at t=0
# instance_id = 0
```

---

## model.py

LSTM neural network architecture for frequency extraction.

### Classes

#### `FrequencyExtractorLSTM`

LSTM model for extracting frequency components from noisy signals.

```python
class FrequencyExtractorLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        output_size: int = 1
    )
```

**Architecture:**
```
Input: [S(t), C1, C2, ..., Cn]
  ↓
LSTM Layer(s)
  ↓
Fully Connected Layer
  ↓
Output: Predicted clean sinusoid value
```

**Attributes:**
- `input_size` (int): Dimension of input vector (default: 5)
- `hidden_size` (int): Number of LSTM hidden units
- `num_layers` (int): Number of stacked LSTM layers
- `output_size` (int): Dimension of output (default: 1)
- `lstm` (nn.LSTM): LSTM layer
- `fc` (nn.Linear): Output layer
- `hidden_state` (Optional[Tuple]): Current hidden state (h_t, c_t)

**Methods:**

##### `reset_hidden_state`

```python
def reset_hidden_state(
    self,
    batch_size: int = 1,
    device: str = 'cpu'
) -> None
```

Resets LSTM hidden state to zeros.

**Parameters:**
- `batch_size` (int): Batch size
- `device` (str): Device ('cpu', 'cuda', or 'mps')

**Usage:**
- Call this method when starting a new signal instance
- Do NOT call between time steps within the same instance

**Example:**
```python
model = FrequencyExtractorLSTM()
model.reset_hidden_state(batch_size=1, device='cpu')
```

##### `forward`

```python
def forward(
    self,
    x: torch.Tensor,
    reset_state: bool = False
) -> torch.Tensor
```

Forward pass through the network.

**Parameters:**
- `x` (torch.Tensor): Input tensor, shape `(batch_size, input_size)` or `(batch_size, 1, input_size)`
- `reset_state` (bool): Whether to reset hidden state before forward pass

**Returns:**
- `torch.Tensor`: Predicted output, shape `(batch_size, 1)`

**State Management:**
- If `reset_state=True`: Resets hidden state to zeros
- If `reset_state=False`: Maintains state from previous forward pass (detached for backprop)

**Example:**
```python
model = FrequencyExtractorLSTM(input_size=5, hidden_size=64)
model.reset_hidden_state(batch_size=4)

# First time step
x1 = torch.randn(4, 5)
out1 = model(x1, reset_state=False)  # State maintained

# Second time step
x2 = torch.randn(4, 5)
out2 = model(x2, reset_state=False)  # Uses state from previous step
```

##### `get_num_parameters`

```python
def get_num_parameters(self) -> int
```

Returns total number of trainable parameters in the model.

**Example:**
```python
model = FrequencyExtractorLSTM(input_size=5, hidden_size=64, num_layers=1)
num_params = model.get_num_parameters()
# Returns: 18241
```

---

## train.py

Training script with proper state management.

### Functions

#### `train_epoch`

```python
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    samples_per_instance: int
) -> float
```

Trains the model for one epoch.

**Parameters:**
- `model` (nn.Module): LSTM model
- `dataloader` (DataLoader): Training data loader
- `criterion` (nn.Module): Loss function (MSELoss)
- `optimizer` (optim.Optimizer): Optimizer (Adam)
- `device` (str): Device for training
- `samples_per_instance` (int): Number of samples per signal instance

**Returns:**
- `float`: Average training loss for the epoch

**State Management:**
- Tracks previous instance_id
- Resets hidden state only when moving to new instance
- Maintains state within instance

**Example:**
```python
train_loss = train_epoch(
    model=model,
    dataloader=train_loader,
    criterion=nn.MSELoss(),
    optimizer=optimizer,
    device='cpu',
    samples_per_instance=40000
)
```

---

#### `validate_epoch`

```python
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    samples_per_instance: int
) -> float
```

Validates the model for one epoch.

**Parameters:**
- Same as `train_epoch`

**Returns:**
- `float`: Average validation loss for the epoch

**Example:**
```python
val_loss = validate_epoch(
    model=model,
    dataloader=val_loader,
    criterion=nn.MSELoss(),
    device='cpu',
    samples_per_instance=40000
)
```

---

#### `main`

```python
def main() -> None
```

Main training function that:
1. Loads configuration
2. Creates datasets
3. Initializes model
4. Trains for specified epochs
5. Saves best model and training history

**Configuration:**
- Reads from inline config dict
- Can be modified to load from `config.yaml`

**Outputs:**
- `outputs/models/best_model.pth` - Best model checkpoint
- `outputs/models/final_model_TIMESTAMP.pth` - Final model
- `outputs/logs/training_history_TIMESTAMP.json` - Training logs

---

## evaluate.py

Model evaluation script.

### Functions

#### `evaluate_model`

```python
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Dict[str, float]
```

Evaluates model on test set with multiple metrics.

**Parameters:**
- `model` (nn.Module): Trained model
- `dataloader` (DataLoader): Test data loader
- `criterion` (nn.Module): Loss function
- `device` (str): Device for evaluation

**Returns:**
- `Dict[str, float]`: Dictionary containing:
  - `'MSE'`: Mean Squared Error
  - `'RMSE'`: Root Mean Squared Error
  - `'MAE'`: Mean Absolute Error
  - `'Max Error'`: Maximum absolute error

**Example:**
```python
metrics = evaluate_model(model, test_loader, criterion, 'cpu')
print(f"Test MSE: {metrics['MSE']:.4f}")
print(f"Test RMSE: {metrics['RMSE']:.4f}")
```

---

#### `evaluate_per_frequency`

```python
def evaluate_per_frequency(
    model: nn.Module,
    dataset: FrequencyExtractionDataset,
    device: str,
    batch_size: int = 1
) -> Dict[float, float]
```

Evaluates model separately for each frequency component.

**Parameters:**
- `model` (nn.Module): Trained model
- `dataset` (FrequencyExtractionDataset): Test dataset
- `device` (str): Device
- `batch_size` (int): Batch size (default: 1)

**Returns:**
- `Dict[float, float]`: MSE for each frequency

**Example:**
```python
per_freq_mse = evaluate_per_frequency(model, test_dataset, 'cpu')
# Returns: {1.0: 0.362, 3.0: 0.501, 5.0: 0.486, 7.0: 0.407}
```

---

#### `generate_predictions`

```python
def generate_predictions(
    model: nn.Module,
    dataset: FrequencyExtractionDataset,
    device: str,
    num_samples: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

Generates predictions for visualization.

**Parameters:**
- `model` (nn.Module): Trained model
- `dataset` (FrequencyExtractionDataset): Dataset
- `device` (str): Device
- `num_samples` (int): Number of time samples to predict

**Returns:**
- Tuple of:
  - `predictions` (np.ndarray): Shape `(num_frequencies, num_samples)`
  - `ground_truth` (np.ndarray): Shape `(num_frequencies, num_samples)`
  - `errors` (np.ndarray): Shape `(num_frequencies, num_samples)`

**Example:**
```python
preds, truth, errors = generate_predictions(model, test_dataset, 'cpu', 1000)
# preds.shape = (4, 1000)  # 4 frequencies, 1000 time steps
```

---

## visualize_data.py

Data visualization utilities.

### Functions

#### `visualize_full_dataset`

```python
def visualize_full_dataset(
    train_dataset: FrequencyExtractionDataset,
    test_dataset: FrequencyExtractionDataset,
    frequencies: List[float],
    sampling_rate: int,
    save_path: str
) -> None
```

Creates comprehensive visualization of train and test datasets.

**Plots:**
1. Full mixed signals (train vs test)
2. Zoomed view to see noise differences
3. Clean frequency components
4. Noise comparison
5. FFT comparison

**Example:**
```python
visualize_full_dataset(
    train_dataset,
    test_dataset,
    [1.0, 3.0, 5.0, 7.0],
    1000,
    'outputs/plots/dataset_viz.png'
)
```

---

#### `visualize_sample_data_points`

```python
def visualize_sample_data_points(
    dataset: FrequencyExtractionDataset,
    frequencies: List[float],
    save_path: str,
    num_samples: int = 10
) -> None
```

Visualizes individual dataset samples showing input format.

**Example:**
```python
visualize_sample_data_points(
    train_dataset,
    [1.0, 3.0, 5.0, 7.0],
    'outputs/plots/samples.png',
    num_samples=10
)
```

---

## plot_results.py

Result visualization after training.

### Functions

#### `plot_training_curves`

Plots training and validation loss over epochs.

#### `plot_signal_reconstruction`

Plots predicted vs ground truth signals for each frequency.

#### `plot_per_frequency_mse`

Bar chart of MSE for each frequency component.

#### `plot_error_distribution`

Histogram of prediction errors.

#### `plot_error_over_time`

Error magnitude over time for each frequency.

---

## Usage Examples

### Complete Training Pipeline

```python
# 1. Create datasets
from data_generator import create_datasets

train_dataset, test_dataset = create_datasets(
    num_train_instances=1,
    num_test_instances=1,
    frequencies=[1.0, 3.0, 5.0, 7.0],
    sampling_rate=1000,
    duration=10.0,
    noise_std=0.1
)

# 2. Create model
from model import FrequencyExtractorLSTM

model = FrequencyExtractorLSTM(
    input_size=5,
    hidden_size=64,
    num_layers=1
)

# 3. Train model
# See train.py for complete training loop

# 4. Evaluate
from evaluate import evaluate_model

metrics = evaluate_model(model, test_loader, criterion, device)
print(f"Test RMSE: {metrics['RMSE']:.3f}")
```

### Inference on New Data

```python
import torch
from model import FrequencyExtractorLSTM

# Load trained model
model = FrequencyExtractorLSTM(input_size=5, hidden_size=64)
model.load_state_dict(torch.load('outputs/models/best_model.pth'))
model.eval()

# Reset state for new signal
model.reset_hidden_state(batch_size=1, device='cpu')

# Process time steps sequentially
for t in range(num_time_steps):
    # input_vec = [S(t), C1, C2, C3, C4]
    input_vec = torch.tensor([[signal[t], 0, 1, 0, 0]])  # Extract 3 Hz
    
    with torch.no_grad():
        prediction = model(input_vec, reset_state=False)
    
    print(f"t={t}: Predicted clean 3Hz value = {prediction.item():.4f}")
```

---

## Type Hints Reference

```python
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
```

---

**Last Updated:** November 2025  
**Version:** 1.0

