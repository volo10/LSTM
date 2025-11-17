"""
Fast Training - Optimized configuration for quicker training on CPU
"""

import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_generator import create_datasets
from model import FrequencyExtractorLSTM


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random

    random.seed(seed)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    prev_instance_id = -1

    for inputs, targets, instance_ids in tqdm(dataloader, desc="Training"):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Reset state when moving to new instance
        if prev_instance_id == -1 or (instance_ids[0].item() != prev_instance_id):
            model.reset_hidden_state(batch_size=inputs.size(0), device=device)
            prev_instance_id = instance_ids[0].item()

        # Forward pass
        outputs = model(inputs, reset_state=False)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
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


def main():
    print("\n" + "=" * 70)
    print("FAST TRAINING MODE - Optimized Configuration")
    print("=" * 70 + "\n")

    # Optimized configuration - smaller dataset for faster training
    config = {
        "num_train_instances": 100,  # Reduced from 1000
        "num_test_instances": 20,  # Reduced from 200
        "frequencies": [1.0, 3.0, 5.0, 7.0],
        "sampling_rate": 1000,
        "duration": 5.0,  # Reduced from 10 seconds
        "noise_std": 0.1,
        "train_seed": 1,
        "test_seed": 2,
        "batch_size": 1,
        "num_epochs": 50,  # Reduced from 100
        "learning_rate": 0.002,  # Slightly higher for faster convergence
        "hidden_size": 64,
        "patience": 15,
        "random_seed": 42,
    }

    set_random_seeds(config["random_seed"])

    # Create output directories
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    # Use CPU (MPS has issues with LSTM)
    device = "cpu"
    print(f"Using device: {device}")
    print("Note: Training on CPU (MPS has compatibility issues with LSTM)")
    print(
        f"Dataset: {config['num_train_instances']} train instances × {config['duration']}s"
    )

    # Create datasets
    print("\nCreating datasets...")
    train_dataset, test_dataset = create_datasets(
        num_train_instances=config["num_train_instances"],
        num_test_instances=config["num_test_instances"],
        frequencies=config["frequencies"],
        sampling_rate=config["sampling_rate"],
        duration=config["duration"],
        noise_std=config["noise_std"],
        train_seed=config["train_seed"],
        test_seed=config["test_seed"],
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
        hidden_size=config["hidden_size"],
        num_layers=1,
        dropout=0.0,
        output_size=1,
    )
    model.to(device)
    model.print_model_summary()

    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Training history
    history = {"train_loss": [], "test_loss": [], "learning_rate": []}

    # Early stopping
    best_test_loss = float("inf")
    epochs_without_improvement = 0
    best_epoch = 0

    print(f"\nStarting training...")
    print("=" * 70)

    for epoch in range(config["num_epochs"]):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        test_loss = validate_epoch(model, test_loader, criterion, device)

        # Update learning rate
        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["learning_rate"].append(current_lr)

        # Print progress
        print(
            f"Epoch {epoch+1:2d}/{config['num_epochs']}: "
            f"Train={train_loss:.6f}, Test={test_loss:.6f}, LR={current_lr:.6f}"
        )

        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "history": history,
                },
                "outputs/models/best_model.pth",
            )
            print(f"  ✓ Saved best model (test loss: {test_loss:.6f})")
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= config["patience"]:
            print(f"\nEarly stopping at epoch {epoch+1}")
            print(
                f"Best model was at epoch {best_epoch} with test loss: {best_test_loss:.6f}"
            )
            break

    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Best test loss: {best_test_loss:.6f} at epoch {best_epoch}")
    print(
        f"Generalization ratio: {best_test_loss / history['train_loss'][best_epoch-1]:.4f}"
    )
    print("=" * 70)

    # Save training history
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = f"outputs/logs/training_history_{timestamp}.json"

    with open(history_path, "w") as f:
        json.dump({"config": config, "history": history}, f, indent=2)

    print(f"\nTraining history saved to: {history_path}")
    print(f"Best model saved to: outputs/models/best_model.pth")

    print("\n" + "=" * 70)
    print("Next steps:")
    print("  1. Run: python3 evaluate.py")
    print("  2. Run: python3 plot_results.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
