"""
LSTM Model for Frequency Extraction

This module defines the LSTM neural network architecture for extracting
individual frequency components from noisy mixed signals.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class FrequencyExtractorLSTM(nn.Module):
    """
    LSTM model for frequency extraction from noisy signals.

    Architecture:
    - Input: [S[t], C1, C2, C3, C4] (5-dimensional)
      - S[t]: noisy signal value at time t
      - C1-C4: one-hot encoded frequency indicator
    - LSTM layer(s) with configurable hidden size
    - Output: Single scalar (predicted clean sinusoid value)

    The model processes one time step at a time (sequence length L=1)
    and maintains internal state (h_t, c_t) that must be reset between
    different signal instances.
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        output_size: int = 1,
    ):
        """
        Initialize the LSTM model.

        Args:
            input_size: Dimension of input vector (default: 5)
            hidden_size: Number of LSTM hidden units
            num_layers: Number of stacked LSTM layers
            dropout: Dropout probability (applied if num_layers > 1)
            output_size: Dimension of output (default: 1)
        """
        super(FrequencyExtractorLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Output layer (fully connected)
        self.fc = nn.Linear(hidden_size, output_size)

        # Internal state (will be set during forward pass)
        self.hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def reset_hidden_state(self, batch_size: int = 1, device: str = "cpu"):
        """
        Reset the LSTM hidden and cell states to zeros.

        This should be called between different signal instances to prevent
        information leakage.

        Args:
            batch_size: Batch size for the hidden state
            device: Device to create tensors on ('cpu' or 'cuda')
        """
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        self.hidden_state = (h_0, c_0)

    def forward(self, x: torch.Tensor, reset_state: bool = False) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_size) or (batch_size, seq_len, input_size)
            reset_state: If True, reset hidden state before processing

        Returns:
            Output tensor of shape (batch_size, output_size) or (batch_size, seq_len, output_size)
        """
        # Handle input shape
        if x.dim() == 2:
            # Add sequence dimension: (batch_size, input_size) -> (batch_size, 1, input_size)
            x = x.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = x.size(0)
        device = x.device

        # Reset or initialize hidden state if needed
        if reset_state or self.hidden_state is None:
            self.reset_hidden_state(batch_size, device)

        # LSTM forward pass
        lstm_out, new_hidden = self.lstm(x, self.hidden_state)

        # Detach hidden state to prevent backprop through time across instances
        # Move to CPU first if on MPS to avoid potential MPS issues, then back
        if reset_state:
            self.hidden_state = new_hidden
        else:
            self.hidden_state = (new_hidden[0].detach(), new_hidden[1].detach())

        # Output layer
        output = self.fc(lstm_out)

        # Remove sequence dimension if input was 2D
        if squeeze_output:
            output = output.squeeze(1)

        return output

    def get_num_parameters(self) -> int:
        """
        Get the total number of trainable parameters.

        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_model_summary(self):
        """Print a summary of the model architecture."""
        print("=" * 70)
        print("LSTM Frequency Extractor Model Summary")
        print("=" * 70)
        print(f"Input size:        {self.input_size}")
        print(f"Hidden size:       {self.hidden_size}")
        print(f"Number of layers:  {self.num_layers}")
        print(f"Output size:       {self.output_size}")
        print(f"Total parameters:  {self.get_num_parameters():,}")
        print("=" * 70)
        print("\nModel Architecture:")
        print(self)
        print("=" * 70)


class FrequencyExtractorLSTMv2(nn.Module):
    """
    Enhanced version of LSTM model with additional features.

    This version includes:
    - Optional batch normalization
    - Optional residual connections
    - More flexible architecture
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_size: int = 1,
        use_batch_norm: bool = False,
    ):
        """
        Initialize the enhanced LSTM model.

        Args:
            input_size: Dimension of input vector
            hidden_size: Number of LSTM hidden units
            num_layers: Number of stacked LSTM layers
            dropout: Dropout probability
            output_size: Dimension of output
            use_batch_norm: Whether to use batch normalization
        """
        super(FrequencyExtractorLSTMv2, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.use_batch_norm = use_batch_norm

        # Input layer (optional preprocessing)
        self.input_fc = nn.Linear(input_size, hidden_size)

        # Batch normalization (optional)
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(hidden_size)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

        # Internal state
        self.hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def reset_hidden_state(self, batch_size: int = 1, device: str = "cpu"):
        """Reset LSTM hidden and cell states."""
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        self.hidden_state = (h_0, c_0)

    def forward(self, x: torch.Tensor, reset_state: bool = False) -> torch.Tensor:
        """Forward pass through the enhanced network."""
        # Handle input shape
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = x.size(0)
        device = x.device

        # Input preprocessing
        x = self.input_fc(x)

        # Batch normalization (if enabled)
        if self.use_batch_norm:
            # Reshape for batch norm: (batch, seq, hidden) -> (batch, hidden, seq)
            x = x.transpose(1, 2)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)

        # Reset or initialize hidden state
        if reset_state or self.hidden_state is None:
            self.reset_hidden_state(batch_size, device)

        # LSTM forward pass
        lstm_out, self.hidden_state = self.lstm(x, self.hidden_state)
        self.hidden_state = (
            self.hidden_state[0].detach(),
            self.hidden_state[1].detach(),
        )

        # Output layers
        out = self.fc1(lstm_out)
        out = self.relu(out)
        out = self.dropout(out)
        output = self.fc2(out)

        # Remove sequence dimension if needed
        if squeeze_output:
            output = output.squeeze(1)

        return output

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    """
    Test model instantiation and forward pass.
    """
    print("Testing LSTM model...\n")

    # Create model
    model = FrequencyExtractorLSTM(
        input_size=5, hidden_size=64, num_layers=1, dropout=0.0, output_size=1
    )

    # Print model summary
    model.print_model_summary()

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 32
    x = torch.randn(batch_size, 5)  # Random input

    # Reset state and forward pass
    model.reset_hidden_state(batch_size)
    output = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sample: {output[0].item():.4f}")

    # Test with sequence input
    print("\nTesting with sequence input...")
    seq_length = 10
    x_seq = torch.randn(batch_size, seq_length, 5)
    model.reset_hidden_state(batch_size)
    output_seq = model(x_seq)

    print(f"Input shape:  {x_seq.shape}")
    print(f"Output shape: {output_seq.shape}")

    print("\nModel test completed successfully!")
