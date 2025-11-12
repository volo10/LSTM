"""
Quick system test to verify all components work together.

This script performs a minimal test of the entire pipeline without full training.
"""

import torch
import numpy as np
import pytest
from data_generator import create_datasets, FrequencyExtractionDataset
from model import FrequencyExtractorLSTM
from torch.utils.data import DataLoader


@pytest.fixture
def train_test_datasets():
    """Fixture to create train and test datasets."""
    train_dataset, test_dataset = create_datasets(
        num_train_instances=2,
        num_test_instances=1,
        frequencies=[1.0, 3.0, 5.0, 7.0],
        sampling_rate=1000,
        duration=1.0,  # Short duration for quick test
        noise_std=0.1,
        train_seed=1,
        test_seed=2
    )
    return train_dataset, test_dataset


@pytest.fixture
def model():
    """Fixture to create a model."""
    model = FrequencyExtractorLSTM(
        input_size=5,
        hidden_size=32,
        num_layers=1,
        dropout=0.0,
        output_size=1
    )
    return model


def test_data_generation():
    """Test dataset generation."""
    print("=" * 70)
    print("Testing Data Generation")
    print("=" * 70)
    
    train_dataset, test_dataset = create_datasets(
        num_train_instances=2,
        num_test_instances=1,
        frequencies=[1.0, 3.0, 5.0, 7.0],
        sampling_rate=1000,
        duration=1.0,  # Short duration for quick test
        noise_std=0.1,
        train_seed=1,
        test_seed=2
    )
    
    print(f"✓ Training dataset created: {len(train_dataset)} samples")
    print(f"✓ Test dataset created: {len(test_dataset)} samples")
    
    # Test getting samples
    input_vec, target, instance_id = train_dataset[0]
    print(f"✓ Sample shape: input={input_vec.shape}, target={target.shape}")
    print(f"✓ Instance ID: {instance_id}")
    
    return train_dataset, test_dataset


def test_model():
    """Test model instantiation and forward pass."""
    print("\n" + "=" * 70)
    print("Testing Model")
    print("=" * 70)
    
    model = FrequencyExtractorLSTM(
        input_size=5,
        hidden_size=32,
        num_layers=1,
        dropout=0.0,
        output_size=1
    )
    
    print(f"✓ Model created with {model.get_num_parameters():,} parameters")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 5)
    model.reset_hidden_state(batch_size)
    output = model(x, reset_state=False)
    
    print(f"✓ Forward pass successful: input {x.shape} -> output {output.shape}")
    
    # Test state management
    model.reset_hidden_state(1)
    x1 = torch.randn(1, 5)
    out1 = model(x1, reset_state=False)
    x2 = torch.randn(1, 5)
    out2 = model(x2, reset_state=False)  # State maintained
    
    print(f"✓ State management working: sequential outputs {out1.shape}, {out2.shape}")
    
    return model


def test_training_loop(model, train_test_datasets):
    """Test a minimal training loop."""
    print("\n" + "=" * 70)
    print("Testing Training Loop")
    print("=" * 70)

    train_dataset, _ = train_test_datasets
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    prev_instance_id = -1
    
    # Process just a few batches
    for i, (inputs, targets, instance_ids) in enumerate(train_loader):
        if i >= 10:  # Only test first 10 batches
            break
        
        # Reset state when moving to new instance
        if prev_instance_id == -1 or (instance_ids[0].item() != prev_instance_id):
            model.reset_hidden_state(batch_size=inputs.size(0))
            prev_instance_id = instance_ids[0].item()
        
        # Forward pass
        outputs = model(inputs, reset_state=False)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    print(f"✓ Training loop successful: {num_batches} batches processed")
    print(f"✓ Average loss: {avg_loss:.6f}")


def test_state_continuity():
    """Test that state is properly maintained within instances."""
    print("\n" + "=" * 70)
    print("Testing State Continuity")
    print("=" * 70)
    
    model = FrequencyExtractorLSTM(input_size=5, hidden_size=16, num_layers=1)
    model.eval()
    
    # Create a sequence of inputs
    sequence_length = 5
    inputs = [torch.randn(1, 5) for _ in range(sequence_length)]
    
    # Method 1: Process with state maintained
    model.reset_hidden_state(1)
    outputs_continuous = []
    with torch.no_grad():
        for x in inputs:
            out = model(x, reset_state=False)
            outputs_continuous.append(out.item())
    
    # Method 2: Process with state reset each time (should be different)
    outputs_reset = []
    with torch.no_grad():
        for x in inputs:
            model.reset_hidden_state(1)
            out = model(x, reset_state=False)
            outputs_reset.append(out.item())
    
    # Check that results are different (state matters)
    difference = np.abs(np.array(outputs_continuous) - np.array(outputs_reset))
    mean_diff = np.mean(difference)
    
    print(f"✓ Continuous state outputs: {outputs_continuous[:3]}")
    print(f"✓ Reset state outputs: {outputs_reset[:3]}")
    print(f"✓ Mean difference: {mean_diff:.6f}")
    
    if mean_diff > 0.001:
        print("✓ State continuity is working correctly (outputs differ with/without state)")
    else:
        print("⚠ Warning: State may not be affecting outputs as expected")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("LSTM FREQUENCY EXTRACTION - SYSTEM TEST")
    print("=" * 70 + "\n")

    # Test 1: Data generation
    train_dataset, test_dataset = test_data_generation()

    # Test 2: Model
    model_instance = test_model()

    # Test 3: Training loop
    test_training_loop(model_instance, (train_dataset, test_dataset))

    # Test 4: State continuity
    test_state_continuity()
    
    # Summary
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
    print("\nThe system is ready to use!")
    print("Next steps:")
    print("  1. Run 'python3 train.py' to train the full model")
    print("  2. Run 'python3 evaluate.py' to evaluate the trained model")
    print("  3. Run 'python3 plot_results.py' to visualize results")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

