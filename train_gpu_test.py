"""
GPU Training Test - Smaller dataset for quick testing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from datetime import datetime

from data_generator import create_datasets
from model import FrequencyExtractorLSTM


def main():
    print("GPU Training Test")
    print("=" * 70)
    
    # Device selection
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Small configuration for testing
    config = {
        'num_train_instances': 5,  # Very small for testing
        'num_test_instances': 2,
        'frequencies': [1.0, 3.0, 5.0, 7.0],
        'sampling_rate': 1000,
        'duration': 1.0,  # Just 1 second
        'noise_std': 0.1,
        'train_seed': 1,
        'test_seed': 2,
        'batch_size': 1,
        'num_epochs': 3,  # Just 3 epochs
        'learning_rate': 0.001,
    }
    
    # Create datasets
    print("\nCreating small datasets...")
    train_dataset, test_dataset = create_datasets(
        num_train_instances=config['num_train_instances'],
        num_test_instances=config['num_test_instances'],
        frequencies=config['frequencies'],
        sampling_rate=config['sampling_rate'],
        duration=config['duration'],
        noise_std=config['noise_std'],
        train_seed=config['train_seed'],
        test_seed=config['test_seed']
    )
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Create model
    print("\nCreating model...")
    model = FrequencyExtractorLSTM(
        input_size=5,
        hidden_size=32,  # Smaller for testing
        num_layers=1,
        dropout=0.0,
        output_size=1
    )
    model.to(device)
    print(f"Model parameters: {model.get_num_parameters():,}")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    print(f"\nTraining for {config['num_epochs']} epochs...")
    
    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0.0
        num_batches = 0
        prev_instance_id = -1
        
        # Train
        for i, (inputs, targets, instance_ids) in enumerate(train_loader):
            if i >= 100:  # Limit batches for quick test
                break
                
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Reset state when moving to new instance
            if prev_instance_id == -1 or (instance_ids[0].item() != prev_instance_id):
                model.reset_hidden_state(batch_size=inputs.size(0), device=device)
                prev_instance_id = instance_ids[0].item()
            
            # Forward pass
            try:
                outputs = model(inputs, reset_state=False)
                loss = criterion(outputs, targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
                
                if (i + 1) % 20 == 0:
                    print(f"  Batch {i+1}/100, Loss: {loss.item():.6f}")
                    
            except Exception as e:
                print(f"Error during training at batch {i}: {e}")
                raise
        
        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0.0
        
        # Validate
        model.eval()
        test_loss = 0.0
        test_batches = 0
        prev_instance_id = -1
        
        with torch.no_grad():
            for i, (inputs, targets, instance_ids) in enumerate(test_loader):
                if i >= 50:  # Limit batches
                    break
                    
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                if prev_instance_id == -1 or (instance_ids[0].item() != prev_instance_id):
                    model.reset_hidden_state(batch_size=inputs.size(0), device=device)
                    prev_instance_id = instance_ids[0].item()
                
                outputs = model(inputs, reset_state=False)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                test_batches += 1
        
        avg_test_loss = test_loss / test_batches if test_batches > 0 else 0.0
        
        print(f"Epoch {epoch+1}/{config['num_epochs']}: "
              f"Train Loss = {avg_train_loss:.6f}, Test Loss = {avg_test_loss:.6f}")
    
    print("\n" + "=" * 70)
    print(f"GPU test completed successfully on {device}!")
    print("=" * 70)
    
    # Save test model
    os.makedirs('outputs/models', exist_ok=True)
    torch.save(model.state_dict(), 'outputs/models/gpu_test_model.pth')
    print("Test model saved to outputs/models/gpu_test_model.pth")


if __name__ == "__main__":
    main()

