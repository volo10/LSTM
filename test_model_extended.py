"""
Extended edge case tests for model.py to increase coverage

Tests additional model functionality and edge cases.
"""

import torch
import torch.nn as nn
import pytest
import numpy as np

from model import FrequencyExtractorLSTM


class TestModelEdgeCases:
    """Test edge cases for the LSTM model"""

    def test_model_with_dropout(self):
        """Test model with dropout enabled"""
        model = FrequencyExtractorLSTM(
            input_size=5,
            hidden_size=32,
            num_layers=2,
            dropout=0.5
        )

        # Dropout should only be active in training mode
        model.train()
        x = torch.randn(4, 5)
        model.reset_hidden_state(4)
        output_train = model(x)

        model.eval()
        model.reset_hidden_state(4)
        output_eval = model(x)

        # Outputs should be different due to dropout
        assert output_train.shape == output_eval.shape
        assert output_train.shape == (4, 1)

    def test_model_with_different_input_sizes(self):
        """Test model with various input sizes"""
        input_sizes = [3, 5, 10, 20]

        for input_size in input_sizes:
            model = FrequencyExtractorLSTM(
                input_size=input_size,
                hidden_size=16,
                num_layers=1
            )

            x = torch.randn(2, input_size)
            model.reset_hidden_state(2)
            output = model(x)

            assert output.shape == (2, 1)

    def test_model_with_large_hidden_size(self):
        """Test model with large hidden size"""
        model = FrequencyExtractorLSTM(
            input_size=5,
            hidden_size=512,
            num_layers=1
        )

        x = torch.randn(1, 5)
        model.reset_hidden_state(1)
        output = model(x)

        assert output.shape == (1, 1)
        assert not torch.isnan(output).any()

    def test_model_with_many_layers(self):
        """Test model with multiple LSTM layers"""
        for num_layers in [1, 2, 3, 4]:
            model = FrequencyExtractorLSTM(
                input_size=5,
                hidden_size=32,
                num_layers=num_layers,
                dropout=0.1 if num_layers > 1 else 0.0
            )

            x = torch.randn(2, 5)
            model.reset_hidden_state(2)
            output = model(x)

            assert output.shape == (2, 1)

    def test_state_detachment(self):
        """Test that hidden state is properly detached"""
        model = FrequencyExtractorLSTM(input_size=5, hidden_size=16)

        x = torch.randn(1, 5)
        model.reset_hidden_state(1)

        # First forward pass
        output1 = model(x, reset_state=False)

        # Check that state was detached (no grad_fn)
        h, c = model.hidden_state
        assert h.grad_fn is None
        assert c.grad_fn is None

    def test_forward_with_3d_input(self):
        """Test forward pass with 3D input (with sequence dimension)"""
        model = FrequencyExtractorLSTM(input_size=5, hidden_size=16)

        # Input with sequence dimension [batch, seq_len, features]
        x = torch.randn(2, 1, 5)  # batch=2, seq_len=1, features=5
        model.reset_hidden_state(2)
        output = model(x)

        # Model preserves sequence dimension: [batch, seq_len, output_size]
        assert output.shape[0] == 2  # batch dimension
        assert output.shape == (2, 1, 1)  # 3D output: (batch, seq_len, output_size)

    def test_forward_with_reset_state_true(self):
        """Test forward pass with reset_state=True"""
        model = FrequencyExtractorLSTM(input_size=5, hidden_size=16)

        x = torch.randn(1, 5)

        # First call with reset_state=True (should reset internally)
        output1 = model(x, reset_state=True)

        # Second call with reset_state=True (should reset again)
        output2 = model(x, reset_state=True)

        # Outputs should be identical for same input
        assert torch.allclose(output1, output2)

    def test_get_num_parameters(self):
        """Test parameter counting"""
        model = FrequencyExtractorLSTM(
            input_size=5,
            hidden_size=64,
            num_layers=1
        )

        num_params = model.get_num_parameters()

        # Verify it's a positive integer
        assert num_params > 0
        assert isinstance(num_params, int)

        # Check it matches actual parameters
        actual_params = sum(p.numel() for p in model.parameters())
        assert num_params == actual_params

    def test_model_summary_method(self):
        """Test model summary/info methods if they exist"""
        model = FrequencyExtractorLSTM(input_size=5, hidden_size=32)

        # Test basic attributes
        assert model.input_size == 5
        assert model.hidden_size == 32
        assert model.num_layers == 1
        assert model.output_size == 1

    def test_zero_input(self):
        """Test model with zero input"""
        model = FrequencyExtractorLSTM(input_size=5, hidden_size=16)

        x = torch.zeros(1, 5)
        model.reset_hidden_state(1)
        output = model(x)

        # Should still produce output (not necessarily zero)
        assert output.shape == (1, 1)
        assert not torch.isnan(output).any()

    def test_very_large_input(self):
        """Test model with very large input values"""
        model = FrequencyExtractorLSTM(input_size=5, hidden_size=16)

        x = torch.ones(1, 5) * 1000.0
        model.reset_hidden_state(1)
        output = model(x)

        # Should not produce NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_very_small_input(self):
        """Test model with very small input values"""
        model = FrequencyExtractorLSTM(input_size=5, hidden_size=16)

        x = torch.ones(1, 5) * 1e-10
        model.reset_hidden_state(1)
        output = model(x)

        # Should not produce NaN
        assert not torch.isnan(output).any()

    def test_negative_input(self):
        """Test model with negative input values"""
        model = FrequencyExtractorLSTM(input_size=5, hidden_size=16)

        x = torch.randn(1, 5) * -1
        model.reset_hidden_state(1)
        output = model(x)

        assert output.shape == (1, 1)
        assert not torch.isnan(output).any()

    def test_mixed_sign_input(self):
        """Test model with mixed positive/negative values"""
        model = FrequencyExtractorLSTM(input_size=5, hidden_size=16)

        x = torch.tensor([[-1.0, 0.5, -0.3, 0.8, -0.2]])
        model.reset_hidden_state(1)
        output = model(x)

        assert output.shape == (1, 1)

    def test_state_persistence_across_calls(self):
        """Test that state persists correctly across multiple calls"""
        model = FrequencyExtractorLSTM(input_size=5, hidden_size=16)

        # Initialize
        model.reset_hidden_state(1)

        # Multiple forward passes
        outputs = []
        for i in range(5):
            x = torch.randn(1, 5)
            output = model(x, reset_state=False)
            outputs.append(output.item())

        # Verify outputs were generated
        assert len(outputs) == 5
        assert all(not np.isnan(o) for o in outputs)

    def test_batch_size_consistency(self):
        """Test model works with consistent batch sizes"""
        model = FrequencyExtractorLSTM(input_size=5, hidden_size=16)

        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 5)
            model.reset_hidden_state(batch_size)
            output = model(x)

            assert output.shape == (batch_size, 1)

    def test_model_device_transfer(self):
        """Test transferring model between devices"""
        model = FrequencyExtractorLSTM(input_size=5, hidden_size=16)

        # Test CPU
        model_cpu = model.to('cpu')
        x_cpu = torch.randn(1, 5)
        model_cpu.reset_hidden_state(1, device='cpu')
        output_cpu = model_cpu(x_cpu)

        assert output_cpu.device.type == 'cpu'

        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.to('cuda')
            x_cuda = torch.randn(1, 5).cuda()
            model_cuda.reset_hidden_state(1, device='cuda')
            output_cuda = model_cuda(x_cuda)

            assert output_cuda.device.type == 'cuda'

    def test_model_train_eval_mode_switching(self):
        """Test switching between train and eval modes"""
        model = FrequencyExtractorLSTM(input_size=5, hidden_size=16)

        # Start in train mode
        model.train()
        assert model.training

        # Switch to eval
        model.eval()
        assert not model.training

        # Switch back
        model.train()
        assert model.training

    def test_gradient_computation_enabled(self):
        """Test that gradients are computed in training mode"""
        model = FrequencyExtractorLSTM(input_size=5, hidden_size=16)
        model.train()

        x = torch.randn(1, 5, requires_grad=True)
        model.reset_hidden_state(1)
        output = model(x)

        # Compute dummy loss and backward
        loss = output.sum()
        loss.backward()

        # Check that input has gradients
        assert x.grad is not None

    def test_no_gradient_in_eval_mode(self):
        """Test that no gradients are computed in eval mode with torch.no_grad()"""
        model = FrequencyExtractorLSTM(input_size=5, hidden_size=16)
        model.eval()

        with torch.no_grad():
            x = torch.randn(1, 5)
            model.reset_hidden_state(1)
            output = model(x)

            # Output should not require grad
            assert not output.requires_grad

    def test_model_reproducibility(self):
        """Test model produces same output for same input with same seed"""
        torch.manual_seed(42)
        model1 = FrequencyExtractorLSTM(input_size=5, hidden_size=16)

        torch.manual_seed(42)
        model2 = FrequencyExtractorLSTM(input_size=5, hidden_size=16)

        torch.manual_seed(42)
        x = torch.randn(1, 5)

        model1.reset_hidden_state(1)
        model2.reset_hidden_state(1)

        output1 = model1(x)
        output2 = model2(x)

        # Should produce identical outputs
        assert torch.allclose(output1, output2)

    def test_multiple_sequential_resets(self):
        """Test multiple sequential state resets"""
        model = FrequencyExtractorLSTM(input_size=5, hidden_size=16)

        for _ in range(10):
            model.reset_hidden_state(1)
            h, c = model.hidden_state

            # State should be zeros
            assert torch.allclose(h, torch.zeros_like(h))
            assert torch.allclose(c, torch.zeros_like(c))

    def test_output_range(self):
        """Test that output values are in reasonable range"""
        model = FrequencyExtractorLSTM(input_size=5, hidden_size=16)

        x = torch.randn(100, 5)  # 100 samples
        model.reset_hidden_state(100)
        outputs = model(x)

        # Outputs should not be too extreme
        assert outputs.abs().max() < 100  # Reasonable upper bound
        assert not torch.isnan(outputs).any()
        assert not torch.isinf(outputs).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
