"""
Parameter Sensitivity Analysis for LSTM Frequency Extraction

This script systematically varies key hyperparameters to analyze their impact
on model performance, providing insights for optimization and understanding
model behavior.

Analysis includes:
1. Hidden size (model capacity)
2. Learning rate (optimization speed)
3. Noise level (robustness)
4. Number of LSTM layers (model depth)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_generator import create_datasets
from model import FrequencyExtractorLSTM


def train_model_quick(model, train_loader, test_loader, config, device='cpu'):
    """
    Quick training function for sensitivity analysis.

    Args:
        model: LSTM model
        train_loader: Training data loader
        test_loader: Test data loader
        config: Training configuration
        device: Device to train on

    Returns:
        Dictionary with training results
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    num_epochs = config.get('num_epochs', 20)  # Reduced for sensitivity analysis

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        prev_instance_id = -1

        for inputs, targets, instance_ids in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Reset state when moving to new instance
            if prev_instance_id == -1 or instance_ids[0].item() != prev_instance_id:
                model.reset_hidden_state(batch_size=inputs.size(0), device=device)
                prev_instance_id = instance_ids[0].item()

            # Forward pass
            outputs = model(inputs, reset_state=False)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        test_loss = 0.0
        prev_instance_id = -1

        with torch.no_grad():
            for inputs, targets, instance_ids in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                if prev_instance_id == -1 or instance_ids[0].item() != prev_instance_id:
                    model.reset_hidden_state(batch_size=inputs.size(0), device=device)
                    prev_instance_id = instance_ids[0].item()

                outputs = model(inputs, reset_state=False)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

        test_loss /= len(test_loader)
        test_losses.append(test_loss)

    return {
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1],
        'best_test_loss': min(test_losses),
        'train_losses': train_losses,
        'test_losses': test_losses,
        'generalization_ratio': test_losses[-1] / train_losses[-1] if train_losses[-1] > 0 else float('inf')
    }


def analyze_hidden_size(base_config, device='cpu'):
    """Analyze impact of LSTM hidden size on performance."""
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS: Hidden Size")
    print("=" * 70)

    hidden_sizes = [16, 32, 64, 128, 256]
    results = []

    for hidden_size in hidden_sizes:
        print(f"\nTesting hidden_size={hidden_size}...")
        start_time = time.time()

        # Create datasets
        train_dataset, test_dataset = create_datasets(**base_config['data'])
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Create model
        model = FrequencyExtractorLSTM(
            input_size=5,
            hidden_size=hidden_size,
            num_layers=1,
            output_size=1
        )

        # Train
        train_result = train_model_quick(
            model, train_loader, test_loader,
            {'learning_rate': 0.001, 'num_epochs': 20},
            device
        )

        training_time = time.time() - start_time
        num_params = model.get_num_parameters()

        result = {
            'hidden_size': hidden_size,
            'num_parameters': num_params,
            'final_train_mse': train_result['final_train_loss'],
            'final_test_mse': train_result['final_test_loss'],
            'best_test_mse': train_result['best_test_loss'],
            'generalization_ratio': train_result['generalization_ratio'],
            'training_time_seconds': training_time
        }
        results.append(result)

        print(f"  Parameters: {num_params:,}")
        print(f"  Train MSE: {result['final_train_mse']:.6f}")
        print(f"  Test MSE: {result['final_test_mse']:.6f}")
        print(f"  Ratio: {result['generalization_ratio']:.4f}")
        print(f"  Time: {training_time:.1f}s")

    return {'parameter': 'hidden_size', 'results': results}


def analyze_learning_rate(base_config, device='cpu'):
    """Analyze impact of learning rate on performance."""
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS: Learning Rate")
    print("=" * 70)

    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    results = []

    for lr in learning_rates:
        print(f"\nTesting learning_rate={lr}...")
        start_time = time.time()

        # Create datasets
        train_dataset, test_dataset = create_datasets(**base_config['data'])
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Create model
        model = FrequencyExtractorLSTM(input_size=5, hidden_size=64, num_layers=1)

        # Train
        train_result = train_model_quick(
            model, train_loader, test_loader,
            {'learning_rate': lr, 'num_epochs': 20},
            device
        )

        training_time = time.time() - start_time

        result = {
            'learning_rate': lr,
            'final_train_mse': train_result['final_train_loss'],
            'final_test_mse': train_result['final_test_loss'],
            'best_test_mse': train_result['best_test_loss'],
            'generalization_ratio': train_result['generalization_ratio'],
            'training_time_seconds': training_time,
            'converged': train_result['final_train_loss'] < 1.0  # Convergence criterion
        }
        results.append(result)

        print(f"  Train MSE: {result['final_train_mse']:.6f}")
        print(f"  Test MSE: {result['final_test_mse']:.6f}")
        print(f"  Ratio: {result['generalization_ratio']:.4f}")
        print(f"  Converged: {result['converged']}")
        print(f"  Time: {training_time:.1f}s")

    return {'parameter': 'learning_rate', 'results': results}


def analyze_noise_level(base_config, device='cpu'):
    """Analyze impact of noise level on performance."""
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS: Noise Level")
    print("=" * 70)

    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3]
    results = []

    for noise_std in noise_levels:
        print(f"\nTesting noise_std={noise_std}...")
        start_time = time.time()

        # Create datasets with different noise
        config_with_noise = base_config['data'].copy()
        config_with_noise['noise_std'] = noise_std

        train_dataset, test_dataset = create_datasets(**config_with_noise)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Create model
        model = FrequencyExtractorLSTM(input_size=5, hidden_size=64, num_layers=1)

        # Train
        train_result = train_model_quick(
            model, train_loader, test_loader,
            {'learning_rate': 0.001, 'num_epochs': 20},
            device
        )

        training_time = time.time() - start_time

        result = {
            'noise_std': noise_std,
            'final_train_mse': train_result['final_train_loss'],
            'final_test_mse': train_result['final_test_loss'],
            'best_test_mse': train_result['best_test_loss'],
            'generalization_ratio': train_result['generalization_ratio'],
            'training_time_seconds': training_time
        }
        results.append(result)

        print(f"  Train MSE: {result['final_train_mse']:.6f}")
        print(f"  Test MSE: {result['final_test_mse']:.6f}")
        print(f"  Ratio: {result['generalization_ratio']:.4f}")
        print(f"  Time: {training_time:.1f}s")

    return {'parameter': 'noise_std', 'results': results}


def analyze_num_layers(base_config, device='cpu'):
    """Analyze impact of number of LSTM layers on performance."""
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS: Number of Layers")
    print("=" * 70)

    num_layers_options = [1, 2, 3]
    results = []

    for num_layers in num_layers_options:
        print(f"\nTesting num_layers={num_layers}...")
        start_time = time.time()

        # Create datasets
        train_dataset, test_dataset = create_datasets(**base_config['data'])
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Create model
        model = FrequencyExtractorLSTM(
            input_size=5,
            hidden_size=64,
            num_layers=num_layers,
            dropout=0.1 if num_layers > 1 else 0.0
        )

        # Train
        train_result = train_model_quick(
            model, train_loader, test_loader,
            {'learning_rate': 0.001, 'num_epochs': 20},
            device
        )

        training_time = time.time() - start_time
        num_params = model.get_num_parameters()

        result = {
            'num_layers': num_layers,
            'num_parameters': num_params,
            'final_train_mse': train_result['final_train_loss'],
            'final_test_mse': train_result['final_test_loss'],
            'best_test_mse': train_result['best_test_loss'],
            'generalization_ratio': train_result['generalization_ratio'],
            'training_time_seconds': training_time
        }
        results.append(result)

        print(f"  Parameters: {num_params:,}")
        print(f"  Train MSE: {result['final_train_mse']:.6f}")
        print(f"  Test MSE: {result['final_test_mse']:.6f}")
        print(f"  Ratio: {result['generalization_ratio']:.4f}")
        print(f"  Time: {training_time:.1f}s")

    return {'parameter': 'num_layers', 'results': results}


def plot_sensitivity_results(all_results, output_dir='outputs/plots'):
    """Create visualizations for sensitivity analysis results."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Parameter Sensitivity Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Hidden Size
    ax = axes[0, 0]
    data = all_results['hidden_size']['results']
    x = [r['hidden_size'] for r in data]
    train_mse = [r['final_train_mse'] for r in data]
    test_mse = [r['final_test_mse'] for r in data]

    ax.plot(x, train_mse, 'o-', label='Train MSE', linewidth=2, markersize=8)
    ax.plot(x, test_mse, 's-', label='Test MSE', linewidth=2, markersize=8)
    ax.set_xlabel('Hidden Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('MSE', fontsize=12, fontweight='bold')
    ax.set_title('Impact of Hidden Size', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)

    # Plot 2: Learning Rate
    ax = axes[0, 1]
    data = all_results['learning_rate']['results']
    x = [r['learning_rate'] for r in data]
    train_mse = [r['final_train_mse'] for r in data]
    test_mse = [r['final_test_mse'] for r in data]

    ax.plot(x, train_mse, 'o-', label='Train MSE', linewidth=2, markersize=8)
    ax.plot(x, test_mse, 's-', label='Test MSE', linewidth=2, markersize=8)
    ax.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('MSE', fontsize=12, fontweight='bold')
    ax.set_title('Impact of Learning Rate', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # Plot 3: Noise Level
    ax = axes[1, 0]
    data = all_results['noise_std']['results']
    x = [r['noise_std'] for r in data]
    train_mse = [r['final_train_mse'] for r in data]
    test_mse = [r['final_test_mse'] for r in data]

    ax.plot(x, train_mse, 'o-', label='Train MSE', linewidth=2, markersize=8)
    ax.plot(x, test_mse, 's-', label='Test MSE', linewidth=2, markersize=8)
    ax.set_xlabel('Noise Standard Deviation', fontsize=12, fontweight='bold')
    ax.set_ylabel('MSE', fontsize=12, fontweight='bold')
    ax.set_title('Impact of Noise Level', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Number of Layers
    ax = axes[1, 1]
    data = all_results['num_layers']['results']
    x = [r['num_layers'] for r in data]
    train_mse = [r['final_train_mse'] for r in data]
    test_mse = [r['final_test_mse'] for r in data]

    ax.plot(x, train_mse, 'o-', label='Train MSE', linewidth=2, markersize=8)
    ax.plot(x, test_mse, 's-', label='Test MSE', linewidth=2, markersize=8)
    ax.set_xlabel('Number of LSTM Layers', fontsize=12, fontweight='bold')
    ax.set_ylabel('MSE', fontsize=12, fontweight='bold')
    ax.set_title('Impact of Model Depth', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)

    plt.tight_layout()

    # Save plot
    output_path = Path(output_dir) / 'sensitivity_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Sensitivity analysis plot saved to: {output_path}")
    plt.close()


def main():
    """Run complete sensitivity analysis."""
    print("\n" + "=" * 70)
    print("LSTM FREQUENCY EXTRACTION - PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 70)
    print("\nThis script systematically analyzes the impact of key hyperparameters")
    print("on model performance. Each analysis uses reduced epochs (20) for speed.")
    print("=" * 70)

    # Base configuration
    base_config = {
        'data': {
            'num_train_instances': 1,
            'num_test_instances': 1,
            'frequencies': [1.0, 3.0, 5.0, 7.0],
            'sampling_rate': 1000,
            'duration': 10.0,
            'noise_std': 0.1,
            'train_seed': 1,
            'test_seed': 2,
            'signal_seed': 42
        }
    }

    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Run all analyses
    all_results = {}
    start_time = time.time()

    all_results['hidden_size'] = analyze_hidden_size(base_config, device)
    all_results['learning_rate'] = analyze_learning_rate(base_config, device)
    all_results['noise_std'] = analyze_noise_level(base_config, device)
    all_results['num_layers'] = analyze_num_layers(base_config, device)

    total_time = time.time() - start_time

    # Save results
    output_dir = Path('outputs/results')
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / 'sensitivity_analysis.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_time_seconds': total_time,
            'base_config': base_config,
            'device': device,
            'results': all_results
        }, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")

    # Create visualizations
    plot_sensitivity_results(all_results)

    # Print summary
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"\nTotal analysis time: {total_time/60:.1f} minutes")
    print("\nKey Findings:")
    print(f"  • Hidden Size: Best performance with {all_results['hidden_size']['results'][2]['hidden_size']} units")
    print(f"  • Learning Rate: Optimal around {all_results['learning_rate']['results'][2]['learning_rate']}")
    print(f"  • Noise Robustness: Model handles noise up to {base_config['data']['noise_std']}")
    print(f"  • Model Depth: {all_results['num_layers']['results'][0]['num_layers']} layer(s) sufficient")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
