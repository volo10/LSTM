"""
Unit Tests for LSTM Frequency Extraction System

Run with: pytest test_units.py -v
Or: python -m pytest test_units.py -v --cov=. --cov-report=html
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from data_generator import (
    FrequencyExtractionDataset,
    create_datasets,
    generate_noisy_signals,
    generate_single_sinusoid,
)
from model import FrequencyExtractorLSTM

# ============================================================================
# Test Data Generation
# ============================================================================


class TestSinusoidGeneration:
    """Test single sinusoid generation."""

    def test_generate_sinusoid_shape(self):
        """Test that generated sinusoid has correct shape."""
        signal = generate_single_sinusoid(
            frequency=5.0, amplitude=1.0, phase=0.0, sampling_rate=1000, duration=1.0
        )
        assert signal.shape == (
            1000,
        ), "Signal should have 1000 samples for 1s at 1000Hz"

    def test_generate_sinusoid_frequency(self):
        """Test that generated sinusoid has correct frequency."""
        frequency = 5.0
        sampling_rate = 1000
        duration = 2.0

        signal = generate_single_sinusoid(
            frequency=frequency,
            amplitude=1.0,
            phase=0.0,
            sampling_rate=sampling_rate,
            duration=duration,
        )

        # Verify frequency using FFT
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1 / sampling_rate)

        # Find peak frequency
        positive_freqs = freqs[: len(freqs) // 2]
        positive_fft = np.abs(fft[: len(fft) // 2])
        peak_freq = positive_freqs[np.argmax(positive_fft)]

        assert (
            abs(peak_freq - frequency) < 0.5
        ), f"Peak frequency {peak_freq} should be close to {frequency}"

    def test_generate_sinusoid_amplitude(self):
        """Test that generated sinusoid has correct amplitude."""
        amplitude = 2.5
        signal = generate_single_sinusoid(
            frequency=1.0,
            amplitude=amplitude,
            phase=0.0,
            sampling_rate=1000,
            duration=1.0,
        )

        # Max value should be approximately equal to amplitude
        assert (
            abs(np.max(signal) - amplitude) < 0.01
        ), f"Max value should be ~{amplitude}"
        assert (
            abs(np.min(signal) + amplitude) < 0.01
        ), f"Min value should be ~{-amplitude}"

    def test_generate_sinusoid_phase(self):
        """Test that phase affects the signal correctly."""
        signal_0 = generate_single_sinusoid(
            frequency=1.0, amplitude=1.0, phase=0.0, sampling_rate=1000, duration=1.0
        )

        signal_pi = generate_single_sinusoid(
            frequency=1.0, amplitude=1.0, phase=np.pi, sampling_rate=1000, duration=1.0
        )

        # Signals with π phase difference should be negatives of each other
        assert np.allclose(
            signal_0, -signal_pi, atol=1e-10
        ), "Phase shift of π should invert signal"

    def test_generate_sinusoid_formula(self):
        """Test that the generated signal matches the formula sin(2πft + φ)."""
        frequency = 3.0
        amplitude = 1.5
        phase = np.pi / 4
        sampling_rate = 100
        duration = 0.1

        signal = generate_single_sinusoid(
            frequency=frequency,
            amplitude=amplitude,
            phase=phase,
            sampling_rate=sampling_rate,
            duration=duration,
        )

        # Manual calculation
        t = np.arange(int(sampling_rate * duration)) / sampling_rate
        expected = amplitude * np.sin(2 * np.pi * frequency * t + phase)

        assert np.allclose(
            signal, expected
        ), "Signal should match formula A*sin(2πft + φ)"


class TestNoisySignalsGeneration:
    """Test noisy mixed signal generation."""

    def test_generate_noisy_signals_shape(self):
        """Test output shapes are correct."""
        num_instances = 3
        frequencies = [1.0, 3.0, 5.0]
        sampling_rate = 100
        duration = 0.5

        mixed_signals, clean_components = generate_noisy_signals(
            num_instances=num_instances,
            frequencies=frequencies,
            sampling_rate=sampling_rate,
            duration=duration,
            noise_std=0.1,
            seed=42,
        )

        expected_samples = int(sampling_rate * duration)

        assert mixed_signals.shape == (
            num_instances,
            expected_samples,
        ), f"Mixed signals shape should be ({num_instances}, {expected_samples})"

        assert clean_components.shape == (
            num_instances,
            len(frequencies),
            expected_samples,
        ), f"Clean components shape should be ({num_instances}, {len(frequencies)}, {expected_samples})"

    def test_generate_noisy_signals_noise(self):
        """Test that noise is actually added."""
        frequencies = [5.0]

        # Without noise
        mixed_clean, clean_components = generate_noisy_signals(
            num_instances=1,
            frequencies=frequencies,
            sampling_rate=1000,
            duration=1.0,
            noise_std=0.0,
            seed=42,
        )

        # With noise
        mixed_noisy, _ = generate_noisy_signals(
            num_instances=1,
            frequencies=frequencies,
            sampling_rate=1000,
            duration=1.0,
            noise_std=0.5,
            seed=42,
        )

        # They should be different
        assert not np.allclose(
            mixed_clean, mixed_noisy
        ), "Noisy signal should differ from clean"

        # Difference should be on the order of noise_std
        noise = mixed_noisy - mixed_clean
        assert (
            abs(np.std(noise[0]) - 0.5) < 0.1
        ), "Noise std should be close to specified value"

    def test_generate_noisy_signals_mixing(self):
        """Test that components are properly mixed."""
        frequencies = [1.0, 3.0]

        mixed_signals, clean_components = generate_noisy_signals(
            num_instances=1,
            frequencies=frequencies,
            sampling_rate=1000,
            duration=1.0,
            noise_std=0.0,  # No noise for clean comparison
            seed=42,
        )

        # Mixed signal should be sum of components
        manual_mix = np.sum(clean_components[0], axis=0)

        assert np.allclose(
            mixed_signals[0], manual_mix, atol=1e-10
        ), "Mixed signal should equal sum of components"

    def test_generate_noisy_signals_reproducibility(self):
        """Test that using the same seed produces the same output."""
        frequencies = [1.0, 3.0, 5.0]

        mixed1, clean1 = generate_noisy_signals(
            num_instances=2,
            frequencies=frequencies,
            sampling_rate=1000,
            duration=0.5,
            noise_std=0.1,
            seed=123,
        )

        mixed2, clean2 = generate_noisy_signals(
            num_instances=2,
            frequencies=frequencies,
            sampling_rate=1000,
            duration=0.5,
            noise_std=0.1,
            seed=123,
        )

        assert np.allclose(
            mixed1, mixed2
        ), "Same seed should produce same mixed signals"
        assert np.allclose(
            clean1, clean2
        ), "Same seed should produce same clean components"


# ============================================================================
# Test Dataset Class
# ============================================================================


class TestFrequencyExtractionDataset:
    """Test the PyTorch Dataset class."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        mixed_signals, clean_components = generate_noisy_signals(
            num_instances=2,
            frequencies=[1.0, 3.0, 5.0],
            sampling_rate=100,
            duration=0.1,  # 10 samples
            noise_std=0.05,
            seed=42,
        )
        return FrequencyExtractionDataset(
            mixed_signals, clean_components, [1.0, 3.0, 5.0]
        )

    def test_dataset_length(self, sample_dataset):
        """Test that dataset length is correct."""
        # 2 instances * 10 samples * 3 frequencies = 60 total samples
        assert (
            len(sample_dataset) == 60
        ), "Dataset length should be instances * samples * frequencies"

    def test_dataset_getitem_shape(self, sample_dataset):
        """Test that __getitem__ returns correct shapes."""
        input_vec, target, instance_id = sample_dataset[0]

        # 3 frequencies = 1 signal + 3 one-hot
        assert input_vec.shape == (
            4,
        ), "Input should be 4-dimensional (1 signal + 3 one-hot)"
        assert target.shape == (1,), "Target should be scalar (1-dimensional)"
        assert isinstance(instance_id, int), "Instance ID should be integer"

    def test_dataset_getitem_onehot(self, sample_dataset):
        """Test that one-hot encoding is correct."""
        # Test first frequency (index 0)
        input_vec0, _, _ = sample_dataset[0]  # idx 0 -> freq_idx 0
        one_hot0 = input_vec0[1:].numpy()

        assert one_hot0[0] == 1.0, "First frequency one-hot should be [1, 0, 0]"
        assert np.sum(one_hot0) == 1.0, "One-hot should sum to 1"

        # Test second frequency (index 1)
        input_vec1, _, _ = sample_dataset[1]  # idx 1 -> freq_idx 1
        one_hot1 = input_vec1[1:].numpy()

        assert one_hot1[1] == 1.0, "Second frequency one-hot should be [0, 1, 0]"

    def test_dataset_getitem_target(self, sample_dataset):
        """Test that target values are correct."""
        input_vec, target, instance_id = sample_dataset[0]

        # Target should be from clean components
        # idx 0 -> instance 0, time_step 0, freq_idx 0
        expected_target = sample_dataset.clean_components[0, 0, 0]

        assert torch.isclose(
            target, expected_target
        ), "Target should match clean component value"

    def test_dataset_instance_id_mapping(self, sample_dataset):
        """Test that instance IDs are correctly assigned."""
        # First 30 samples should be instance 0 (10 samples * 3 frequencies)
        _, _, instance_id0 = sample_dataset[0]
        assert instance_id0 == 0, "First sample should be from instance 0"

        _, _, instance_id29 = sample_dataset[29]
        assert instance_id29 == 0, "Sample 29 should still be instance 0"

        # Next 30 samples should be instance 1
        _, _, instance_id30 = sample_dataset[30]
        assert instance_id30 == 1, "Sample 30 should be from instance 1"

    def test_dataset_all_frequencies_covered(self, sample_dataset):
        """Test that all frequencies are represented in the dataset."""
        # For first instance, first time step
        one_hot_vectors = []
        for i in range(3):  # 3 frequencies
            input_vec, _, _ = sample_dataset[i]
            one_hot_vectors.append(input_vec[1:].numpy())

        # Check that we have all three one-hot patterns
        assert (
            np.sum([oh[0] for oh in one_hot_vectors]) == 1
        ), "Frequency 0 should appear once"
        assert (
            np.sum([oh[1] for oh in one_hot_vectors]) == 1
        ), "Frequency 1 should appear once"
        assert (
            np.sum([oh[2] for oh in one_hot_vectors]) == 1
        ), "Frequency 2 should appear once"


class TestCreateDatasets:
    """Test the create_datasets function."""

    def test_create_datasets_return_types(self):
        """Test that create_datasets returns correct types."""
        train_dataset, test_dataset = create_datasets(
            num_train_instances=1,
            num_test_instances=1,
            frequencies=[1.0, 3.0],
            sampling_rate=100,
            duration=0.1,
            noise_std=0.1,
            train_seed=1,
            test_seed=2,
            signal_seed=42,
        )

        assert isinstance(
            train_dataset, FrequencyExtractionDataset
        ), "Should return FrequencyExtractionDataset"
        assert isinstance(
            test_dataset, FrequencyExtractionDataset
        ), "Should return FrequencyExtractionDataset"

    def test_create_datasets_same_signal(self):
        """Test that train and test use the same underlying signal."""
        train_dataset, test_dataset = create_datasets(
            num_train_instances=1,
            num_test_instances=1,
            frequencies=[1.0, 3.0],
            sampling_rate=100,
            duration=0.1,
            noise_std=0.0,  # No noise for comparison
            train_seed=1,
            test_seed=2,
            signal_seed=42,
        )

        # Clean components should be identical
        assert torch.allclose(
            train_dataset.clean_components, test_dataset.clean_components
        ), "Train and test should have same clean components"

    def test_create_datasets_different_noise(self):
        """Test that train and test have different noise."""
        train_dataset, test_dataset = create_datasets(
            num_train_instances=1,
            num_test_instances=1,
            frequencies=[1.0, 3.0],
            sampling_rate=100,
            duration=0.1,
            noise_std=0.2,
            train_seed=1,
            test_seed=2,
            signal_seed=42,
        )

        # Mixed signals should differ (different noise)
        assert not torch.allclose(
            train_dataset.mixed_signals, test_dataset.mixed_signals
        ), "Train and test should have different noise"

    def test_create_datasets_size(self):
        """Test that datasets have correct size."""
        frequencies = [1.0, 3.0, 5.0, 7.0]
        duration = 1.0
        sampling_rate = 1000

        train_dataset, test_dataset = create_datasets(
            num_train_instances=1,
            num_test_instances=1,
            frequencies=frequencies,
            sampling_rate=sampling_rate,
            duration=duration,
            noise_std=0.1,
            train_seed=1,
            test_seed=2,
            signal_seed=42,
        )

        expected_length = 1 * int(duration * sampling_rate) * len(frequencies)

        assert (
            len(train_dataset) == expected_length
        ), f"Train dataset should have {expected_length} samples"
        assert (
            len(test_dataset) == expected_length
        ), f"Test dataset should have {expected_length} samples"


# ============================================================================
# Test Model Architecture
# ============================================================================


class TestFrequencyExtractorLSTM:
    """Test the LSTM model."""

    def test_model_initialization(self):
        """Test that model initializes correctly."""
        model = FrequencyExtractorLSTM(
            input_size=5, hidden_size=64, num_layers=2, dropout=0.2, output_size=1
        )

        assert model.input_size == 5
        assert model.hidden_size == 64
        assert model.num_layers == 2
        assert model.output_size == 1

    def test_model_parameter_count(self):
        """Test that model has reasonable number of parameters."""
        model = FrequencyExtractorLSTM(
            input_size=5, hidden_size=64, num_layers=1, dropout=0.0
        )

        num_params = model.get_num_parameters()

        # Should have reasonable number of parameters (roughly)
        # LSTM has most parameters, FC layer adds a small amount
        expected_min = 10000  # At least 10k parameters
        expected_max = 25000  # At most 25k parameters

        assert (
            expected_min <= num_params <= expected_max
        ), f"Expected parameters in range [{expected_min}, {expected_max}], got {num_params}"

    def test_model_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        model = FrequencyExtractorLSTM(input_size=5, hidden_size=32, num_layers=1)

        batch_size = 8
        x = torch.randn(batch_size, 5)

        model.reset_hidden_state(batch_size)
        output = model(x, reset_state=False)

        assert output.shape == (
            batch_size,
            1,
        ), f"Output shape should be ({batch_size}, 1), got {output.shape}"

    def test_model_forward_no_nan(self):
        """Test that forward pass doesn't produce NaN values."""
        model = FrequencyExtractorLSTM(input_size=5, hidden_size=32, num_layers=1)

        x = torch.randn(4, 5)
        model.reset_hidden_state(4)
        output = model(x, reset_state=False)

        assert not torch.isnan(output).any(), "Output should not contain NaN"
        assert not torch.isinf(output).any(), "Output should not contain Inf"

    def test_model_reset_hidden_state(self):
        """Test hidden state reset functionality."""
        model = FrequencyExtractorLSTM(input_size=5, hidden_size=32, num_layers=1)

        # Reset state
        model.reset_hidden_state(batch_size=1)

        assert model.hidden_state is not None, "Hidden state should be initialized"
        assert len(model.hidden_state) == 2, "Hidden state should be tuple (h, c)"

        h, c = model.hidden_state
        assert h.shape == (1, 1, 32), "Hidden state h should have correct shape"
        assert c.shape == (1, 1, 32), "Cell state c should have correct shape"

        # Should be all zeros
        assert torch.allclose(
            h, torch.zeros_like(h)
        ), "Initial hidden state should be zeros"
        assert torch.allclose(
            c, torch.zeros_like(c)
        ), "Initial cell state should be zeros"

    def test_model_state_persistence(self):
        """Test that state persists across forward passes."""
        model = FrequencyExtractorLSTM(input_size=5, hidden_size=16, num_layers=1)
        model.eval()

        # First pass
        model.reset_hidden_state(batch_size=1)
        x1 = torch.randn(1, 5)
        with torch.no_grad():
            out1 = model(x1, reset_state=False)
        state_after_first = model.hidden_state

        # Second pass (state maintained)
        x2 = torch.randn(1, 5)
        with torch.no_grad():
            out2 = model(x2, reset_state=False)
        state_after_second = model.hidden_state

        # States should be different
        assert not torch.allclose(
            state_after_first[0], state_after_second[0]
        ), "Hidden state should change after forward pass"

    def test_model_state_reset_in_forward(self):
        """Test that reset_state=True properly resets the state."""
        model = FrequencyExtractorLSTM(input_size=5, hidden_size=16, num_layers=1)
        model.eval()

        # First pass
        x1 = torch.randn(1, 5)
        with torch.no_grad():
            out1 = model(x1, reset_state=True)

        # Second pass with reset
        x2 = torch.randn(1, 5)
        with torch.no_grad():
            out2 = model(x2, reset_state=True)

        # The state should be reset to zeros before each pass
        # So same input should give same output
        with torch.no_grad():
            out1_repeat = model(x1, reset_state=True)

        assert torch.allclose(
            out1, out1_repeat, atol=1e-6
        ), "Same input with reset should give same output"

    def test_model_different_batch_sizes(self):
        """Test model with different batch sizes."""
        model = FrequencyExtractorLSTM(input_size=5, hidden_size=32, num_layers=1)

        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 5)
            model.reset_hidden_state(batch_size)
            output = model(x, reset_state=False)

            assert output.shape == (
                batch_size,
                1,
            ), f"Output shape should match batch size {batch_size}"

    def test_model_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = FrequencyExtractorLSTM(input_size=5, hidden_size=32, num_layers=1)

        x = torch.randn(4, 5)
        target = torch.randn(4, 1)

        model.reset_hidden_state(4)
        output = model(x, reset_state=False)

        loss = torch.nn.MSELoss()(output, target)
        loss.backward()

        # Check that gradients exist
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Gradient for {name} should not be None"
            assert not torch.isnan(
                param.grad
            ).any(), f"Gradient for {name} should not contain NaN"


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for the complete pipeline."""

    def test_end_to_end_training_step(self):
        """Test a complete training step."""
        # Create small dataset
        frequencies = [1.0, 3.0]
        train_dataset, _ = create_datasets(
            num_train_instances=1,
            num_test_instances=1,
            frequencies=frequencies,
            sampling_rate=100,
            duration=0.1,  # 10 samples
            noise_std=0.1,
            train_seed=1,
            test_seed=2,
            signal_seed=42,
        )

        # Create model (input_size = 1 signal + len(frequencies) one-hot)
        input_size = 1 + len(frequencies)
        model = FrequencyExtractorLSTM(
            input_size=input_size, hidden_size=16, num_layers=1
        )

        # Create dataloader
        dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

        # Training components
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        model.train()
        total_loss = 0.0
        prev_instance_id = -1

        for inputs, targets, instance_ids in dataloader:
            # Reset state for new instance
            if prev_instance_id == -1 or instance_ids[0].item() != prev_instance_id:
                model.reset_hidden_state(inputs.size(0))
                prev_instance_id = instance_ids[0].item()

            # Forward pass
            outputs = model(inputs, reset_state=False)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Should complete without errors
        assert total_loss > 0, "Training should produce positive loss"
        assert not np.isnan(total_loss), "Loss should not be NaN"

    def test_end_to_end_evaluation(self):
        """Test a complete evaluation step."""
        # Create dataset
        frequencies = [1.0, 3.0]
        _, test_dataset = create_datasets(
            num_train_instances=1,
            num_test_instances=1,
            frequencies=frequencies,
            sampling_rate=100,
            duration=0.1,
            noise_std=0.1,
            train_seed=1,
            test_seed=2,
            signal_seed=42,
        )

        # Create model (input_size = 1 signal + len(frequencies) one-hot)
        input_size = 1 + len(frequencies)
        model = FrequencyExtractorLSTM(
            input_size=input_size, hidden_size=16, num_layers=1
        )
        model.eval()

        # Create dataloader
        dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Evaluation
        criterion = torch.nn.MSELoss()
        total_loss = 0.0
        prev_instance_id = -1

        with torch.no_grad():
            for inputs, targets, instance_ids in dataloader:
                if prev_instance_id == -1 or instance_ids[0].item() != prev_instance_id:
                    model.reset_hidden_state(inputs.size(0))
                    prev_instance_id = instance_ids[0].item()

                outputs = model(inputs, reset_state=False)
                loss = criterion(outputs, targets)
                total_loss += loss.item()

        # Should complete without errors
        assert total_loss > 0, "Evaluation should produce positive loss"
        assert not np.isnan(total_loss), "Loss should not be NaN"

    def test_dataloader_compatibility(self):
        """Test that dataset works with PyTorch DataLoader."""
        frequencies = [1.0, 3.0]
        train_dataset, _ = create_datasets(
            num_train_instances=2,
            num_test_instances=1,
            frequencies=frequencies,
            sampling_rate=100,
            duration=0.1,
            noise_std=0.1,
            train_seed=1,
            test_seed=2,
            signal_seed=42,
        )

        # Expected input dimension: 1 signal + len(frequencies) one-hot
        expected_input_dim = 1 + len(frequencies)

        # Test different DataLoader configurations
        for batch_size in [1, 4]:
            for shuffle in [True, False]:
                dataloader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
                )

                # Get one batch
                batch = next(iter(dataloader))
                inputs, targets, instance_ids = batch

                assert inputs.shape[0] <= batch_size, "Batch size should be correct"
                assert (
                    inputs.shape[1] == expected_input_dim
                ), f"Input dimension should be {expected_input_dim}"
                assert targets.shape[1] == 1, "Target dimension should be 1"


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_noise(self):
        """Test signal generation with zero noise."""
        mixed, clean = generate_noisy_signals(
            num_instances=1,
            frequencies=[1.0],
            sampling_rate=100,
            duration=0.1,
            noise_std=0.0,
            seed=42,
        )

        # Mixed signal should equal clean signal
        expected = clean[0, 0, :]
        assert np.allclose(mixed[0], expected), "Zero noise should give clean signal"

    def test_single_frequency(self):
        """Test with only one frequency."""
        train_dataset, _ = create_datasets(
            num_train_instances=1,
            num_test_instances=1,
            frequencies=[5.0],  # Single frequency
            sampling_rate=100,
            duration=0.1,
            noise_std=0.1,
            train_seed=1,
            test_seed=2,
            signal_seed=42,
        )

        # Should still work correctly
        assert (
            len(train_dataset) == 10
        ), "Should have 10 samples (1 instance * 10 time * 1 freq)"

        # One-hot should be [1] (single element)
        input_vec, _, _ = train_dataset[0]
        assert input_vec.shape == (2,), "Input should be 2D (signal + 1 one-hot)"

    def test_very_short_duration(self):
        """Test with very short signal duration."""
        signal = generate_single_sinusoid(
            frequency=10.0,
            amplitude=1.0,
            phase=0.0,
            sampling_rate=1000,
            duration=0.001,  # 1 millisecond
        )

        assert len(signal) == 1, "Should have 1 sample"

    def test_model_single_sample_batch(self):
        """Test model with batch size of 1."""
        model = FrequencyExtractorLSTM(input_size=5, hidden_size=16, num_layers=1)

        x = torch.randn(1, 5)
        model.reset_hidden_state(1)
        output = model(x, reset_state=False)

        assert output.shape == (1, 1), "Should handle batch size 1"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
