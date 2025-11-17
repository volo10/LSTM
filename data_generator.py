"""
Data Generator for LSTM Frequency Extraction

This module generates synthetic noisy mixed signals composed of multiple
sinusoidal components and creates datasets for training and testing the
LSTM frequency extraction model.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt


def generate_single_sinusoid(
    frequency: float,
    amplitude: float,
    phase: float,
    sampling_rate: int,
    duration: float
) -> np.ndarray:
    """
    Generate a single sinusoidal signal.
    
    Args:
        frequency: Frequency in Hz
        amplitude: Signal amplitude
        phase: Phase offset in radians
        sampling_rate: Sampling rate in Hz
        duration: Signal duration in seconds
    
    Returns:
        numpy array of signal samples
    """
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
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate multiple instances of noisy mixed signals.
    
    Args:
        num_instances: Number of signal instances to generate
        frequencies: List of frequency components (e.g., [1, 3, 5, 7])
        sampling_rate: Sampling rate in Hz
        duration: Signal duration in seconds
        noise_std: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (mixed_signals, clean_components)
        - mixed_signals: shape (num_instances, num_samples)
        - clean_components: shape (num_instances, num_frequencies, num_samples)
    """
    if seed is not None:
        np.random.seed(seed)
    
    num_samples = int(sampling_rate * duration)
    num_frequencies = len(frequencies)
    
    # Initialize arrays
    mixed_signals = np.zeros((num_instances, num_samples))
    clean_components = np.zeros((num_instances, num_frequencies, num_samples))
    
    for i in range(num_instances):
        # Generate random amplitudes and phases for this instance
        amplitudes = np.random.uniform(0.8, 1.2, num_frequencies)
        phases = np.random.uniform(0, 0.2 * np.pi, num_frequencies)
        
        # Generate each frequency component
        for j, freq in enumerate(frequencies):
            component = generate_single_sinusoid(
                frequency=freq,
                amplitude=amplitudes[j],
                phase=phases[j],
                sampling_rate=sampling_rate,
                duration=duration
            )
            clean_components[i, j, :] = component
        
        # Mix all components
        mixed_signal = np.sum(clean_components[i], axis=0)
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_std, num_samples)
        mixed_signals[i] = mixed_signal + noise
    
    return mixed_signals, clean_components


class FrequencyExtractionDataset(Dataset):
    """
    PyTorch Dataset for LSTM frequency extraction training.
    
    Each sample consists of:
    - Input: [S[t], C1, C2, C3, C4] where S[t] is noisy signal and C is one-hot
    - Target: clean sinusoid value at time t for selected frequency
    """
    
    def __init__(
        self,
        mixed_signals: np.ndarray,
        clean_components: np.ndarray,
        frequencies: List[float]
    ):
        """
        Initialize dataset.
        
        Args:
            mixed_signals: shape (num_instances, num_samples)
            clean_components: shape (num_instances, num_frequencies, num_samples)
            frequencies: List of frequency values
        """
        self.mixed_signals = torch.FloatTensor(mixed_signals)
        self.clean_components = torch.FloatTensor(clean_components)
        self.frequencies = frequencies
        self.num_instances = mixed_signals.shape[0]
        self.num_samples = mixed_signals.shape[1]
        self.num_frequencies = len(frequencies)
    
    def __len__(self) -> int:
        """Return total number of samples (instances * samples * frequencies)."""
        return self.num_instances * self.num_samples * self.num_frequencies
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (input, target, instance_id)
            - input: [S[t], C1, C2, C3, C4] shape (5,)
            - target: clean sinusoid value, shape (1,)
            - instance_id: which signal instance this sample belongs to
        """
        # Decode index into (instance, time_step, frequency)
        instance_id = idx // (self.num_samples * self.num_frequencies)
        remainder = idx % (self.num_samples * self.num_frequencies)
        time_step = remainder // self.num_frequencies
        freq_idx = remainder % self.num_frequencies
        
        # Get noisy signal value at time t
        signal_value = self.mixed_signals[instance_id, time_step]
        
        # Create one-hot encoding for frequency
        one_hot = torch.zeros(self.num_frequencies)
        one_hot[freq_idx] = 1.0
        
        # Combine signal and one-hot encoding
        input_vector = torch.cat([signal_value.unsqueeze(0), one_hot])
        
        # Get target (clean component at time t)
        target = self.clean_components[instance_id, freq_idx, time_step]
        
        return input_vector, target.unsqueeze(0), instance_id


def create_datasets(
    num_train_instances: int = 1,  # Single signal instance
    num_test_instances: int = 1,   # Single signal instance (same signal, different noise)
    frequencies: List[float] = [1.0, 3.0, 5.0, 7.0],
    sampling_rate: int = 1000,
    duration: float = 10.0,
    noise_std: float = 0.1,
    train_seed: int = 1,
    test_seed: int = 2,
    signal_seed: int = 42  # Seed for the signal itself (same for train and test)
) -> Tuple[FrequencyExtractionDataset, FrequencyExtractionDataset]:
    """
    Create training and test datasets.
    
    Creates ONE signal with fixed amplitudes and phases, then adds different
    noise for training and test sets.
    
    Args:
        num_train_instances: Number of training signal instances (typically 1)
        num_test_instances: Number of test signal instances (typically 1)
        frequencies: List of frequency components in Hz
        sampling_rate: Sampling rate in Hz
        duration: Signal duration in seconds
        noise_std: Standard deviation of Gaussian noise
        train_seed: Random seed for training noise
        test_seed: Random seed for test noise
        signal_seed: Random seed for signal generation (same for both train/test)
    
    Returns:
        Tuple of (train_dataset, test_dataset)
        Each dataset has num_samples * num_frequencies total samples
    """
    # Generate the base signal (same for train and test)
    # Use signal_seed for consistent signal generation
    _, clean_components = generate_noisy_signals(
        num_instances=1,
        frequencies=frequencies,
        sampling_rate=sampling_rate,
        duration=duration,
        noise_std=0.0,  # No noise yet
        seed=signal_seed
    )
    
    # The clean signal (without noise)
    clean_signal = np.sum(clean_components[0], axis=0)
    
    # Add training noise
    np.random.seed(train_seed)
    train_noise = np.random.normal(0, noise_std, clean_signal.shape)
    train_mixed = (clean_signal + train_noise).reshape(1, -1)
    
    # Add test noise (different seed, same signal)
    np.random.seed(test_seed)
    test_noise = np.random.normal(0, noise_std, clean_signal.shape)
    test_mixed = (clean_signal + test_noise).reshape(1, -1)
    
    # Create datasets
    train_dataset = FrequencyExtractionDataset(train_mixed, clean_components, frequencies)
    test_dataset = FrequencyExtractionDataset(test_mixed, clean_components, frequencies)
    
    return train_dataset, test_dataset


def visualize_sample_signals(
    mixed_signals: np.ndarray,
    clean_components: np.ndarray,
    frequencies: List[float],
    instance_idx: int = 0,
    time_window: float = 2.0,
    sampling_rate: int = 1000,
    save_path: Optional[str] = None
):
    """
    Visualize a sample signal and its components.
    
    Args:
        mixed_signals: shape (num_instances, num_samples)
        clean_components: shape (num_instances, num_frequencies, num_samples)
        frequencies: List of frequency values
        instance_idx: Which instance to visualize
        time_window: Time window to display in seconds
        sampling_rate: Sampling rate in Hz
        save_path: Optional path to save figure
    """
    num_samples_to_show = int(time_window * sampling_rate)
    t = np.arange(num_samples_to_show) / sampling_rate
    
    fig, axes = plt.subplots(len(frequencies) + 1, 1, figsize=(12, 10))
    
    # Plot mixed signal
    axes[0].plot(t, mixed_signals[instance_idx, :num_samples_to_show], 'k-', alpha=0.7)
    axes[0].set_title('Mixed Noisy Signal', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # Plot each clean component
    for i, freq in enumerate(frequencies):
        axes[i+1].plot(t, clean_components[instance_idx, i, :num_samples_to_show], 
                       label=f'{freq} Hz', linewidth=2)
        axes[i+1].set_title(f'Clean Component: {freq} Hz', fontsize=12, fontweight='bold')
        axes[i+1].set_ylabel('Amplitude')
        axes[i+1].grid(True, alpha=0.3)
        axes[i+1].legend()
    
    axes[-1].set_xlabel('Time (seconds)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    """
    Test data generation and visualization.
    """
    print("Generating datasets...")
    
    # Create datasets
    train_dataset, test_dataset = create_datasets(
        num_train_instances=10,  # Small for testing
        num_test_instances=2,
        frequencies=[1.0, 3.0, 5.0, 7.0],
        sampling_rate=1000,
        duration=10.0,
        noise_std=0.1,
        train_seed=1,
        test_seed=2
    )
    
    print(f"Training dataset size: {len(train_dataset)} samples")
    print(f"Test dataset size: {len(test_dataset)} samples")
    
    # Test getting a sample
    input_vec, target, instance_id = train_dataset[0]
    print(f"\nSample input shape: {input_vec.shape}")
    print(f"Sample target shape: {target.shape}")
    print(f"Input vector: {input_vec}")
    print(f"Target value: {target.item():.4f}")
    
    # Visualize sample signals
    print("\nVisualizing sample signals...")
    train_mixed = train_dataset.mixed_signals.numpy()
    train_clean = train_dataset.clean_components.numpy()
    
    visualize_sample_signals(
        mixed_signals=train_mixed,
        clean_components=train_clean,
        frequencies=[1.0, 3.0, 5.0, 7.0],
        instance_idx=0,
        time_window=2.0,
        sampling_rate=1000
    )
    
    print("\nData generation test completed successfully!")

