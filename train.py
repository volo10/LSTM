"""
Training Script for LSTM Frequency Extraction

This module implements the training loop for the LSTM model, including
validation, checkpointing, and logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from data_generator import create_datasets
from model import FrequencyExtractorLSTM


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    samples_per_instance: int
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: LSTM model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        samples_per_instance: Number of samples per signal instance (for state reset)
    
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Track previous instance ID for state reset
    prev_instance_id = -1
    
    for inputs, targets, instance_ids in tqdm(dataloader, desc="Training", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Reset hidden state ONLY when we move to a new signal instance
        # Within the same instance, state is maintained to learn temporal patterns
        if prev_instance_id == -1 or (instance_ids[0].item() != prev_instance_id):
            model.reset_hidden_state(batch_size=inputs.size(0), device=device)
            prev_instance_id = instance_ids[0].item()
        
        # Forward pass (state is maintained and updated automatically)
        # The model keeps its hidden state for the next time step
        outputs = model(inputs, reset_state=False)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    samples_per_instance: int
) -> float:
    """
    Validate for one epoch.
    
    Args:
        model: LSTM model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        samples_per_instance: Number of samples per signal instance (for state reset)
    
    Returns:
        Average validation loss for the epoch
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    prev_instance_id = -1
    
    with torch.no_grad():
        for inputs, targets, instance_ids in tqdm(dataloader, desc="Validation", leave=False):
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
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: str,
    save_dir: str,
    samples_per_instance: int,
    patience: int = 20
) -> Dict[str, List[float]]:
    """
    Train the model with early stopping.
    
    Args:
        model: LSTM model
        train_loader: Training data loader
        test_loader: Test data loader
        num_epochs: Maximum number of epochs
        learning_rate: Learning rate
        device: Device to train on
        save_dir: Directory to save checkpoints
        samples_per_instance: Number of samples per signal instance
        patience: Early stopping patience
    
    Returns:
        Dictionary with training history
    """
    # Setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training history
    history = {
        'train_loss': [],
        'test_loss': [],
        'learning_rate': []
    }
    
    # Early stopping
    best_test_loss = float('inf')
    epochs_without_improvement = 0
    best_epoch = 0
    
    print(f"\nStarting training on {device}...")
    print(f"Model parameters: {model.get_num_parameters():,}")
    print("=" * 70)
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, samples_per_instance
        )
        
        # Validate
        test_loss = validate_epoch(
            model, test_loader, criterion, device, samples_per_instance
        )
        
        # Update learning rate
        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['learning_rate'].append(current_lr)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Test Loss: {test_loss:.6f} | "
              f"LR: {current_lr:.6f}")
        
        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            
            checkpoint_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'history': history
            }, checkpoint_path)
            print(f"  → Saved best model (test loss: {test_loss:.6f})")
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Best model was at epoch {best_epoch} with test loss: {best_test_loss:.6f}")
            break
    
    print("=" * 70)
    print(f"Training completed!")
    print(f"Best test loss: {best_test_loss:.6f} at epoch {best_epoch}")
    
    return history


def main():
    """Main training function."""
    # Configuration
    config = {
        # Data parameters
        'num_train_instances': 1,      # Single signal instance
        'num_test_instances': 1,       # Single signal instance (same signal, different noise)
        'frequencies': [1.0, 3.0, 5.0, 7.0],  # 4 frequencies
        'sampling_rate': 1000,         # 1000 Hz
        'duration': 10.0,              # 10 seconds = 10,000 samples
        'noise_std': 0.1,
        'train_seed': 1,               # Seed for training noise
        'test_seed': 2,                # Seed for test noise (different)
        'signal_seed': 42,             # Seed for signal generation (same for both)
        
        # Model parameters
        'input_size': 5,
        'hidden_size': 64,
        'num_layers': 1,
        'dropout': 0.0,
        'output_size': 1,
        
        # Training parameters
        'batch_size': 1,  # Process one sample at a time for proper state management
        'num_epochs': 100,
        'learning_rate': 0.001,
        'patience': 20,
        'random_seed': 42,
        
        # Paths
        'save_dir': 'outputs/models',
        'log_dir': 'outputs/logs'
    }
    
    # Set random seeds
    set_random_seeds(config['random_seed'])
    
    # Create output directories
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Device - Use CPU (MPS has compatibility issues with LSTM)
    device = 'cpu'
    print(f"Using device: {device}")
    print("Note: Training on CPU (Apple MPS has known issues with PyTorch LSTM)")
    
    # Create datasets
    print("\nCreating datasets...")
    print(f"Signal: {config['duration']}s @ {config['sampling_rate']}Hz = "
          f"{int(config['duration'] * config['sampling_rate'])} samples")
    print(f"Frequencies: {config['frequencies']}")
    print(f"Total training samples: {int(config['duration'] * config['sampling_rate'])} × {len(config['frequencies'])} = "
          f"{int(config['duration'] * config['sampling_rate']) * len(config['frequencies'])}")
    
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
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=False,  # Don't shuffle to maintain temporal order
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    print("\nCreating model...")
    model = FrequencyExtractorLSTM(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        output_size=config['output_size']
    )
    model.to(device)
    model.print_model_summary()
    
    # Calculate samples per instance
    samples_per_instance = int(config['sampling_rate'] * config['duration']) * len(config['frequencies'])
    
    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        device=device,
        save_dir=config['save_dir'],
        samples_per_instance=samples_per_instance,
        patience=config['patience']
    )
    
    # Save training history
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    history_path = os.path.join(config['log_dir'], f'training_history_{timestamp}.json')
    
    with open(history_path, 'w') as f:
        json.dump({
            'config': config,
            'history': history
        }, f, indent=2)
    
    print(f"\nTraining history saved to: {history_path}")
    
    # Save final model
    final_model_path = os.path.join(config['save_dir'], f'final_model_{timestamp}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history
    }, final_model_path)
    
    print(f"Final model saved to: {final_model_path}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

