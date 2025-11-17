# This is the Python script version - you can upload this to Colab and it will work
# Or convert it to a notebook using: jupyter nbconvert --to notebook train_colab.py

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Check GPU
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ==================== DATA GENERATOR ====================
def generate_single_sinusoid(
    frequency: float,
    amplitude: float,
    phase: float,
    sampling_rate: int,
    duration: float,
) -> np.ndarray:
    num_samples = int(sampling_rate * duration)
    t = np.arange(num_samples) / sampling_rate
    signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return signal


def generate_noisy_signals(
    num_instances: int,
    frequencies: List[float],
    sampling_rate: int,
    duration: float,
    noise_std: float,
    seed: Optional[int] = None,
):
    if seed is not None:
        np.random.seed(seed)
    num_samples = int(sampling_rate * duration)
    num_frequencies = len(frequencies)
    mixed_signals = np.zeros((num_instances, num_samples))
    clean_components = np.zeros((num_instances, num_frequencies, num_samples))
    for i in range(num_instances):
        amplitudes = np.random.uniform(0.8, 1.2, num_frequencies)
        phases = np.random.uniform(0, 0.2 * np.pi, num_frequencies)
        for j, freq in enumerate(frequencies):
            component = generate_single_sinusoid(
                freq, amplitudes[j], phases[j], sampling_rate, duration
            )
            clean_components[i, j, :] = component
        mixed_signal = np.sum(clean_components[i], axis=0)
        noise = np.random.normal(0, noise_std, num_samples)
        mixed_signals[i] = mixed_signal + noise
    return mixed_signals, clean_components


class FrequencyExtractionDataset(Dataset):
    def __init__(self, mixed_signals, clean_components, frequencies):
        self.mixed_signals = torch.FloatTensor(mixed_signals)
        self.clean_components = torch.FloatTensor(clean_components)
        self.frequencies = frequencies
        self.num_instances = mixed_signals.shape[0]
        self.num_samples = mixed_signals.shape[1]
        self.num_frequencies = len(frequencies)

    def __len__(self):
        return self.num_instances * self.num_samples * self.num_frequencies

    def __getitem__(self, idx):
        instance_id = idx // (self.num_samples * self.num_frequencies)
        remainder = idx % (self.num_samples * self.num_frequencies)
        time_step = remainder // self.num_frequencies
        freq_idx = remainder % self.num_frequencies
        signal_value = self.mixed_signals[instance_id, time_step]
        one_hot = torch.zeros(self.num_frequencies)
        one_hot[freq_idx] = 1.0
        input_vector = torch.cat([signal_value.unsqueeze(0), one_hot])
        target = self.clean_components[instance_id, freq_idx, time_step]
        return input_vector, target.unsqueeze(0), instance_id


def create_datasets(
    num_train_instances=1,
    num_test_instances=1,
    frequencies=[1.0, 3.0, 5.0, 7.0],
    sampling_rate=1000,
    duration=10.0,
    noise_std=0.1,
    train_seed=1,
    test_seed=2,
    signal_seed=42,
):
    _, clean_components = generate_noisy_signals(
        1, frequencies, sampling_rate, duration, 0.0, signal_seed
    )
    clean_signal = np.sum(clean_components[0], axis=0)
    np.random.seed(train_seed)
    train_noise = np.random.normal(0, noise_std, clean_signal.shape)
    train_mixed = (clean_signal + train_noise).reshape(1, -1)
    np.random.seed(test_seed)
    test_noise = np.random.normal(0, noise_std, clean_signal.shape)
    test_mixed = (clean_signal + test_noise).reshape(1, -1)
    train_dataset = FrequencyExtractionDataset(
        train_mixed, clean_components, frequencies
    )
    test_dataset = FrequencyExtractionDataset(test_mixed, clean_components, frequencies)
    return train_dataset, test_dataset


# ==================== MODEL ====================
class FrequencyExtractorLSTM(nn.Module):
    def __init__(
        self, input_size=5, hidden_size=64, num_layers=1, dropout=0.0, output_size=1
    ):
        super(FrequencyExtractorLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_state = None

    def reset_hidden_state(self, batch_size=1, device="cpu"):
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        self.hidden_state = (h_0, c_0)

    def forward(self, x, reset_state=False):
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False
        batch_size = x.size(0)
        device = x.device
        if reset_state or self.hidden_state is None:
            self.reset_hidden_state(batch_size, device)
        lstm_out, new_hidden = self.lstm(x, self.hidden_state)
        if reset_state:
            self.hidden_state = new_hidden
        else:
            self.hidden_state = (new_hidden[0].detach(), new_hidden[1].detach())
        output = self.fc(lstm_out)
        if squeeze_output:
            output = output.squeeze(1)
        return output

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==================== TRAINING ====================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    prev_instance_id = -1
    for inputs, targets, instance_ids in tqdm(dataloader, desc="Training"):
        inputs = inputs.to(device)
        targets = targets.to(device)
        if prev_instance_id == -1 or (instance_ids[0].item() != prev_instance_id):
            model.reset_hidden_state(batch_size=inputs.size(0), device=device)
            prev_instance_id = instance_ids[0].item()
        outputs = model(inputs, reset_state=False)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    prev_instance_id = -1
    with torch.no_grad():
        for inputs, targets, instance_ids in tqdm(dataloader, desc="Validation"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            if prev_instance_id == -1 or (instance_ids[0].item() != prev_instance_id):
                model.reset_hidden_state(batch_size=inputs.size(0), device=device)
                prev_instance_id = instance_ids[0].item()
            outputs = model(inputs, reset_state=False)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            num_batches += 1
    return total_loss / num_batches


# ==================== MAIN ====================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTraining on: {device}\n")

    # Create datasets
    print("Creating datasets...")
    train_dataset, test_dataset = create_datasets()
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}\n")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Create model
    model = FrequencyExtractorLSTM().to(device)
    print(f"Model parameters: {model.get_num_parameters():,}\n")

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    # Training loop
    num_epochs = 100
    patience = 20
    best_test_loss = float("inf")
    epochs_without_improvement = 0
    history = {"train_loss": [], "test_loss": []}

    print("Starting training...\n")
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss = validate_epoch(model, test_loader, criterion, device)
        scheduler.step(test_loss)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        print(
            f"Epoch {epoch+1:3d}/{num_epochs}: Train={train_loss:.6f}, Test={test_loss:.6f}"
        )
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                },
                "best_model.pth",
            )
            print(f"  ✓ Saved best model")
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print(f"\nTraining completed! Best test loss: {best_test_loss:.6f}")
    print("Model saved to: best_model.pth")
