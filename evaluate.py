"""
Evaluation Script for LSTM Frequency Extraction

This module evaluates the trained LSTM model on test data and computes
various performance metrics.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import os
from typing import Dict, List, Tuple
from tqdm import tqdm

from data_generator import create_datasets
from model import FrequencyExtractorLSTM


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    num_frequencies: int
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained LSTM model
        dataloader: Data loader
        device: Device to evaluate on
        num_frequencies: Number of frequency components
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    criterion = nn.MSELoss()
    
    total_loss = 0.0
    num_samples = 0
    
    # Per-frequency metrics
    freq_losses = [0.0] * num_frequencies
    freq_counts = [0] * num_frequencies
    
    # Store predictions and targets for analysis
    all_predictions = []
    all_targets = []
    
    prev_instance_id = -1
    
    with torch.no_grad():
        for inputs, targets, instance_ids in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Reset hidden state ONLY when we move to a new signal instance
            # Within the same instance, state is maintained to learn temporal patterns
            if prev_instance_id == -1 or (instance_ids[0].item() != prev_instance_id):
                model.reset_hidden_state(batch_size=inputs.size(0), device=device)
                prev_instance_id = instance_ids[0].item()
            
            # Forward pass (state is maintained for next time step)
            outputs = model(inputs, reset_state=False)
            loss = criterion(outputs, targets)
            
            # Accumulate total loss
            total_loss += loss.item() * inputs.size(0)
            num_samples += inputs.size(0)
            
            # Determine which frequency this sample corresponds to
            # One-hot encoding is in positions 1-4 of input
            one_hot = inputs[:, 1:].cpu().numpy()
            freq_idx = np.argmax(one_hot, axis=1)
            
            # Accumulate per-frequency loss
            for i in range(inputs.size(0)):
                f_idx = freq_idx[i]
                freq_losses[f_idx] += ((outputs[i] - targets[i]) ** 2).item()
                freq_counts[f_idx] += 1
            
            # Store predictions and targets
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Compute average metrics
    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    
    # Compute per-frequency MSE
    per_freq_mse = []
    for i in range(num_frequencies):
        if freq_counts[i] > 0:
            per_freq_mse.append(freq_losses[i] / freq_counts[i])
        else:
            per_freq_mse.append(0.0)
    
    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Compute additional metrics
    mae = np.mean(np.abs(all_predictions - all_targets))
    rmse = np.sqrt(avg_loss)
    max_error = np.max(np.abs(all_predictions - all_targets))
    
    metrics = {
        'mse': avg_loss,
        'rmse': rmse,
        'mae': mae,
        'max_error': max_error,
        'per_freq_mse': per_freq_mse,
        'num_samples': num_samples
    }
    
    return metrics


def reconstruct_signals(
    model: nn.Module,
    dataset,
    device: str,
    instance_idx: int = 0,
    frequencies: List[float] = [1.0, 3.0, 5.0, 7.0]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct all frequency components for a specific signal instance.
    
    Args:
        model: Trained LSTM model
        dataset: Dataset containing the signals
        device: Device to run on
        instance_idx: Which signal instance to reconstruct
        frequencies: List of frequency values
    
    Returns:
        Tuple of (mixed_signal, predicted_components, target_components)
    """
    model.eval()
    
    num_frequencies = len(frequencies)
    num_samples = dataset.num_samples
    
    # Get mixed signal
    mixed_signal = dataset.mixed_signals[instance_idx].numpy()
    
    # Initialize arrays for predictions and targets
    predicted_components = np.zeros((num_frequencies, num_samples))
    target_components = dataset.clean_components[instance_idx].numpy()
    
    with torch.no_grad():
        for freq_idx in range(num_frequencies):
            # Reset hidden state for each frequency component extraction
            model.reset_hidden_state(batch_size=1, device=device)
            
            # Create one-hot encoding for this frequency
            one_hot = torch.zeros(num_frequencies)
            one_hot[freq_idx] = 1.0
            
            # Process each time step sequentially
            # State is maintained across time steps to use LSTM memory
            for t in range(num_samples):
                # Create input [S[t], C1, C2, C3, C4]
                signal_value = mixed_signal[t]
                input_vec = torch.cat([
                    torch.tensor([signal_value], dtype=torch.float32),
                    one_hot
                ]).unsqueeze(0).to(device)
                
                # Predict (state is automatically maintained for next step)
                output = model(input_vec, reset_state=False)
                predicted_components[freq_idx, t] = output.item()
    
    return mixed_signal, predicted_components, target_components


def main():
    """Main evaluation function."""
    # Configuration (should match training config)
    config = {
        'num_train_instances': 1,      # Single signal instance
        'num_test_instances': 1,       # Single signal instance (same signal, different noise)
        'frequencies': [1.0, 3.0, 5.0, 7.0],  # 4 frequencies
        'sampling_rate': 1000,         # 1000 Hz
        'duration': 10.0,              # 10 seconds = 10,000 samples
        'noise_std': 0.1,
        'train_seed': 1,               # Training noise seed
        'test_seed': 2,                # Test noise seed
        'signal_seed': 42,             # Signal generation seed
        'batch_size': 1,
        'model_path': 'outputs/models/best_model.pth',
        'results_dir': 'outputs/results'
    }
    
    # Create results directory
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # Device - support for CUDA (NVIDIA), MPS (Apple Silicon), and CPU
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset, test_dataset = create_datasets(
        num_train_instances=config['num_train_instances'],
        num_test_instances=config['num_test_instances'],
        frequencies=config['frequencies'],
        sampling_rate=config['sampling_rate'],
        duration=config['duration'],
        noise_std=config['noise_std'],
        train_seed=config['train_seed'],
        test_seed=config['test_seed'],
        signal_seed=config['signal_seed']
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(config['model_path'], map_location=device)
    
    # Create model with same architecture
    model = FrequencyExtractorLSTM(
        input_size=5,
        hidden_size=64,
        num_layers=1,
        dropout=0.0,
        output_size=1
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['train_loss']:.6f}")
    print(f"Test loss: {checkpoint['test_loss']:.6f}")
    
    # Evaluate on training set
    print("\n" + "=" * 70)
    print("Evaluating on TRAINING set...")
    print("=" * 70)
    train_metrics = evaluate_model(model, train_loader, device, len(config['frequencies']))
    
    print(f"\nTraining Metrics:")
    print(f"  MSE:  {train_metrics['mse']:.6f}")
    print(f"  RMSE: {train_metrics['rmse']:.6f}")
    print(f"  MAE:  {train_metrics['mae']:.6f}")
    print(f"  Max Error: {train_metrics['max_error']:.6f}")
    print(f"\nPer-Frequency MSE (Training):")
    for i, freq in enumerate(config['frequencies']):
        print(f"  {freq} Hz: {train_metrics['per_freq_mse'][i]:.6f}")
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("Evaluating on TEST set...")
    print("=" * 70)
    test_metrics = evaluate_model(model, test_loader, device, len(config['frequencies']))
    
    print(f"\nTest Metrics:")
    print(f"  MSE:  {test_metrics['mse']:.6f}")
    print(f"  RMSE: {test_metrics['rmse']:.6f}")
    print(f"  MAE:  {test_metrics['mae']:.6f}")
    print(f"  Max Error: {test_metrics['max_error']:.6f}")
    print(f"\nPer-Frequency MSE (Test):")
    for i, freq in enumerate(config['frequencies']):
        print(f"  {freq} Hz: {test_metrics['per_freq_mse'][i]:.6f}")
    
    # Generalization analysis
    print("\n" + "=" * 70)
    print("Generalization Analysis:")
    print("=" * 70)
    ratio = test_metrics['mse'] / train_metrics['mse'] if train_metrics['mse'] > 0 else float('inf')
    print(f"Test MSE / Train MSE ratio: {ratio:.4f}")
    if 0.8 <= ratio <= 1.2:
        print("✓ Good generalization (ratio between 0.8 and 1.2)")
    else:
        print("✗ Potential overfitting or underfitting issue")
    
    # Reconstruct sample signals
    print("\n" + "=" * 70)
    print("Reconstructing sample signals...")
    print("=" * 70)
    
    # Reconstruct from test set
    mixed, predicted, target = reconstruct_signals(
        model, test_dataset, device, instance_idx=0, frequencies=config['frequencies']
    )
    
    print("Signal reconstruction completed for test instance 0")
    
    # Helper function to convert numpy types to Python types
    def convert_to_python_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_python_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        else:
            return obj
    
    # Save results
    results = {
        'config': config,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'generalization_ratio': ratio
    }
    
    # Convert all numpy types to Python types
    results = convert_to_python_types(results)
    
    results_path = os.path.join(config['results_dir'], 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # Save reconstructed signals
    reconstruction_path = os.path.join(config['results_dir'], 'reconstructed_signals.npz')
    np.savez(
        reconstruction_path,
        mixed_signal=mixed,
        predicted_components=predicted,
        target_components=target,
        frequencies=config['frequencies']
    )
    
    print(f"Reconstructed signals saved to: {reconstruction_path}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

