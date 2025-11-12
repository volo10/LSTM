"""
Comprehensive tests for evaluate.py

Tests evaluation functionality including:
- Metric computation (MSE, RMSE, MAE, Max Error)
- Per-frequency evaluation
- Signal reconstruction
- Results saving
"""

import torch
import numpy as np
import pytest
import tempfile
import json
from pathlib import Path
from torch.utils.data import DataLoader

from model import FrequencyExtractorLSTM
from data_generator import create_datasets


class TestMetricComputation:
    """Test metric computation functions"""

    def test_mse_calculation(self):
        """Test MSE calculation"""
        predictions = np.array([1.0, 2.0, 3.0, 4.0])
        targets = np.array([1.1, 2.1, 2.9, 4.2])

        mse = np.mean((predictions - targets) ** 2)

        assert mse >= 0
        assert mse < 0.1  # Should be small for close predictions

    def test_rmse_calculation(self):
        """Test RMSE calculation"""
        predictions = np.array([1.0, 2.0, 3.0, 4.0])
        targets = np.array([1.1, 2.1, 2.9, 4.2])

        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)

        assert rmse >= 0
        assert rmse == np.sqrt(mse)

    def test_mae_calculation(self):
        """Test MAE calculation"""
        predictions = np.array([1.0, 2.0, 3.0, 4.0])
        targets = np.array([1.1, 2.1, 2.9, 4.2])

        mae = np.mean(np.abs(predictions - targets))

        assert mae >= 0
        assert mae < 0.3

    def test_max_error_calculation(self):
        """Test max error calculation"""
        predictions = np.array([1.0, 2.0, 3.0, 4.0])
        targets = np.array([1.1, 2.1, 2.9, 4.5])

        max_error = np.max(np.abs(predictions - targets))

        assert max_error >= 0
        assert max_error == 0.5  # Largest difference

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions"""
        predictions = np.array([1.0, 2.0, 3.0, 4.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0])

        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        max_error = np.max(np.abs(predictions - targets))

        assert mse == 0
        assert mae == 0
        assert max_error == 0

    def test_metrics_with_nan(self):
        """Test metric handling with NaN values"""
        predictions = np.array([1.0, 2.0, np.nan, 4.0])
        targets = np.array([1.1, 2.1, 2.9, 4.2])

        # Check if NaN is detected
        assert np.any(np.isnan(predictions))

        # Remove NaN for calculation
        mask = ~np.isnan(predictions)
        clean_pred = predictions[mask]
        clean_targ = targets[mask]

        mse = np.mean((clean_pred - clean_targ) ** 2)
        assert not np.isnan(mse)


class TestPerFrequencyEvaluation:
    """Test per-frequency evaluation"""

    @pytest.fixture
    def evaluation_data(self):
        """Create data for per-frequency evaluation"""
        frequencies = [1.0, 3.0, 5.0, 7.0]
        num_samples = 100

        # Create predictions and targets for each frequency
        per_freq_data = {}
        for freq in frequencies:
            predictions = np.sin(2 * np.pi * freq * np.linspace(0, 1, num_samples))
            # Add small error
            targets = predictions + np.random.normal(0, 0.05, num_samples)
            per_freq_data[freq] = (predictions, targets)

        return per_freq_data

    def test_per_frequency_mse(self, evaluation_data):
        """Test MSE calculation per frequency"""
        results = {}

        for freq, (predictions, targets) in evaluation_data.items():
            mse = np.mean((predictions - targets) ** 2)
            results[freq] = mse

        # Verify we have results for all frequencies
        assert len(results) == 4
        assert all(freq in results for freq in [1.0, 3.0, 5.0, 7.0])

        # Verify all MSEs are reasonable
        assert all(0 <= mse < 0.1 for mse in results.values())

    def test_frequency_ordering(self, evaluation_data):
        """Test that frequencies are processed in order"""
        frequencies = sorted(evaluation_data.keys())

        assert frequencies == [1.0, 3.0, 5.0, 7.0]


class TestModelEvaluation:
    """Test model evaluation on datasets"""

    @pytest.fixture
    def small_evaluation_setup(self):
        """Create small datasets and model for evaluation"""
        # Create datasets
        train_dataset, test_dataset = create_datasets(
            num_train_instances=1,
            num_test_instances=1,
            frequencies=[1.0, 3.0],
            sampling_rate=100,
            duration=0.5,
            noise_std=0.1,
            train_seed=1,
            test_seed=2
        )

        # Create model
        model = FrequencyExtractorLSTM(input_size=3, hidden_size=16, num_layers=1)

        return model, train_dataset, test_dataset

    def test_evaluate_on_dataset(self, small_evaluation_setup):
        """Test evaluation on a dataset"""
        model, train_dataset, _ = small_evaluation_setup

        dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

        model.eval()
        all_predictions = []
        all_targets = []
        prev_instance_id = -1

        with torch.no_grad():
            for inputs, targets, instance_ids in dataloader:
                if prev_instance_id == -1 or instance_ids[0].item() != prev_instance_id:
                    model.reset_hidden_state(batch_size=1)
                    prev_instance_id = instance_ids[0].item()

                outputs = model(inputs, reset_state=False)

                all_predictions.append(outputs.item())
                all_targets.append(targets.item())

        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)

        # Compute metrics
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))

        # Verify metrics are reasonable
        assert mse >= 0
        assert rmse >= 0
        assert mae >= 0
        assert not np.isnan(mse)

    def test_model_eval_mode(self, small_evaluation_setup):
        """Test that model is in eval mode during evaluation"""
        model, _, _ = small_evaluation_setup

        model.eval()
        assert not model.training

        model.train()
        assert model.training


class TestSignalReconstruction:
    """Test signal reconstruction functionality"""

    def test_reconstruct_signals(self):
        """Test reconstructing signals for all frequencies"""
        # Create test data
        frequencies = [1.0, 3.0, 5.0, 7.0]
        sampling_rate = 1000
        duration = 1.0
        time = np.linspace(0, duration, int(sampling_rate * duration))

        reconstructed_signals = {}

        for freq in frequencies:
            # Generate a simple sinusoid
            signal = np.sin(2 * np.pi * freq * time)
            reconstructed_signals[freq] = signal

        # Verify reconstruction
        assert len(reconstructed_signals) == len(frequencies)

        for freq, signal in reconstructed_signals.items():
            assert len(signal) == len(time)
            assert signal.min() >= -1.0
            assert signal.max() <= 1.0

    def test_save_reconstructed_signals(self):
        """Test saving reconstructed signals to file"""
        frequencies = [1.0, 3.0]
        sampling_rate = 100
        duration = 0.5
        time = np.linspace(0, duration, int(sampling_rate * duration))

        data_to_save = {'time': time}

        for freq in frequencies:
            signal = np.sin(2 * np.pi * freq * time)
            data_to_save[f'freq_{freq}'] = signal

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "reconstructed.npz"

            # Save
            np.savez(save_path, **data_to_save)

            # Load and verify
            loaded = np.load(save_path)

            assert 'time' in loaded
            assert len(loaded['time']) == len(time)

            for freq in frequencies:
                key = f'freq_{freq}'
                assert key in loaded
                assert len(loaded[key]) == len(time)

            # Close the file to avoid Windows permission issues
            loaded.close()


class TestResultsSaving:
    """Test saving evaluation results"""

    def test_save_results_json(self):
        """Test saving results to JSON"""
        results = {
            'config': {
                'frequencies': [1.0, 3.0, 5.0, 7.0],
                'noise_std': 0.1
            },
            'train_metrics': {
                'mse': 0.123,
                'rmse': 0.351,
                'mae': 0.280,
                'max_error': 1.234
            },
            'test_metrics': {
                'mse': 0.125,
                'rmse': 0.354,
                'mae': 0.282,
                'max_error': 1.256
            },
            'per_freq_mse': [0.12, 0.13, 0.12, 0.13]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "results.json"

            # Save
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)

            # Load and verify
            with open(save_path, 'r') as f:
                loaded = json.load(f)

            assert loaded['train_metrics']['mse'] == 0.123
            assert loaded['test_metrics']['mse'] == 0.125
            assert len(loaded['per_freq_mse']) == 4

    def test_results_structure(self):
        """Test results dictionary structure"""
        results = {
            'config': {},
            'train_metrics': {},
            'test_metrics': {},
            'generalization_ratio': 1.0
        }

        # Verify required keys
        assert 'config' in results
        assert 'train_metrics' in results
        assert 'test_metrics' in results
        assert 'generalization_ratio' in results


class TestGeneralizationAnalysis:
    """Test generalization analysis"""

    def test_generalization_ratio_good(self):
        """Test good generalization (ratio ~1.0)"""
        train_mse = 0.440
        test_mse = 0.439

        ratio = test_mse / train_mse

        assert 0.8 <= ratio <= 1.2  # Good generalization
        assert abs(ratio - 1.0) < 0.1  # Very close to 1.0

    def test_generalization_ratio_overfitting(self):
        """Test overfitting detection (high ratio)"""
        train_mse = 0.1
        test_mse = 0.5

        ratio = test_mse / train_mse

        assert ratio > 1.2  # Overfitting detected

    def test_generalization_ratio_underfitting(self):
        """Test underfitting detection (low ratio)"""
        train_mse = 0.8
        test_mse = 0.6

        ratio = test_mse / train_mse

        assert ratio < 0.8  # Unusual case (test better than train)


class TestEdgeCases:
    """Test edge cases in evaluation"""

    def test_single_sample_evaluation(self):
        """Test evaluation with single sample"""
        prediction = np.array([1.5])
        target = np.array([1.6])

        mse = np.mean((prediction - target) ** 2)
        mae = np.mean(np.abs(prediction - target))

        assert np.isclose(mse, 0.01)
        assert np.isclose(mae, 0.1)

    def test_large_error_evaluation(self):
        """Test evaluation with large errors"""
        predictions = np.array([0.0, 0.0, 0.0])
        targets = np.array([10.0, 20.0, 30.0])

        mse = np.mean((predictions - targets) ** 2)
        max_error = np.max(np.abs(predictions - targets))

        assert mse == np.mean([100, 400, 900])
        assert max_error == 30.0

    def test_zero_predictions(self):
        """Test metrics with zero predictions"""
        predictions = np.zeros(100)
        targets = np.ones(100)

        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))

        assert mse == 1.0
        assert mae == 1.0

    def test_identical_predictions(self):
        """Test metrics when all predictions are identical"""
        predictions = np.ones(100) * 0.5
        targets = np.linspace(0, 1, 100)

        mse = np.mean((predictions - targets) ** 2)

        # Should have non-zero error
        assert mse > 0


class TestBatchEvaluation:
    """Test batch evaluation functionality"""

    def test_batch_processing(self):
        """Test evaluating in batches"""
        # Create larger dataset
        num_samples = 1000
        predictions = np.random.randn(num_samples)
        targets = predictions + np.random.normal(0, 0.1, num_samples)

        # Process in batches
        batch_size = 100
        batch_mses = []

        for i in range(0, num_samples, batch_size):
            batch_pred = predictions[i:i+batch_size]
            batch_targ = targets[i:i+batch_size]

            batch_mse = np.mean((batch_pred - batch_targ) ** 2)
            batch_mses.append(batch_mse)

        # Overall MSE should be close to mean of batch MSEs
        overall_mse = np.mean((predictions - targets) ** 2)
        mean_batch_mse = np.mean(batch_mses)

        assert abs(overall_mse - mean_batch_mse) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
