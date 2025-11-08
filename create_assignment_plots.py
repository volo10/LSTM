"""
Create the specific plots requested in the assignment:
1. Overlay plot for f₂ showing Target, Noisy input, and LSTM output
2. Separate plots for each frequency comparing Target vs Predicted
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os


def create_f2_overlay_plot(
    mixed_signal: np.ndarray,
    predicted_components: np.ndarray,
    target_components: np.ndarray,
    frequencies: np.ndarray,
    sampling_rate: int = 1000,
    time_window: float = 2.0,
    save_path: str = None
):
    """
    Create overlay plot for f₂ showing:
    - Target signal (no noise)
    - Noisy input signal S(t)
    - LSTM output

    Args:
        mixed_signal: Noisy mixed signal, shape (num_samples,)
        predicted_components: Predicted components, shape (num_frequencies, num_samples)
        target_components: Target components, shape (num_frequencies, num_samples)
        frequencies: Array of frequency values
        sampling_rate: Sampling rate in Hz
        time_window: Time window to display in seconds
        save_path: Path to save the figure
    """
    # f₂ is the second frequency (index 1)
    f2_idx = 1
    f2_freq = frequencies[f2_idx]

    num_samples_to_show = int(time_window * sampling_rate)
    t = np.arange(num_samples_to_show) / sampling_rate

    # Create figure
    plt.figure(figsize=(16, 8))

    # Plot all three signals
    plt.plot(t, target_components[f2_idx, :num_samples_to_show],
             'b-', label=f'Target (Pure {f2_freq} Hz)', linewidth=2.5, alpha=0.8)
    plt.plot(t, mixed_signal[:num_samples_to_show],
             'gray', label='Noisy Mixed Signal S(t)', linewidth=1.5, alpha=0.5)
    plt.plot(t, predicted_components[f2_idx, :num_samples_to_show],
             'r--', label=f'LSTM Output ({f2_freq} Hz)', linewidth=2, alpha=0.8)

    # Calculate MSE for this component
    mse = np.mean((predicted_components[f2_idx] - target_components[f2_idx]) ** 2)

    plt.title(f'Overlay Plot for f₂ = {f2_freq} Hz (MSE: {mse:.6f})',
              fontsize=16, fontweight='bold')
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('Amplitude', fontsize=14)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, time_window])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] f2 overlay plot saved to: {save_path}")

    plt.close()


def create_per_frequency_plots(
    predicted_components: np.ndarray,
    target_components: np.ndarray,
    frequencies: np.ndarray,
    sampling_rate: int = 1000,
    time_window: float = 2.0,
    save_path: str = None
):
    """
    Create separate plots for each frequency (f₁–f₄) comparing Target vs Predicted.

    Args:
        predicted_components: Predicted components, shape (num_frequencies, num_samples)
        target_components: Target components, shape (num_frequencies, num_samples)
        frequencies: Array of frequency values
        sampling_rate: Sampling rate in Hz
        time_window: Time window to display in seconds
        save_path: Path to save the figure
    """
    num_frequencies = len(frequencies)
    num_samples_to_show = int(time_window * sampling_rate)
    t = np.arange(num_samples_to_show) / sampling_rate

    # Create subplots
    fig, axes = plt.subplots(num_frequencies, 1, figsize=(16, 12))

    for i, freq in enumerate(frequencies):
        # Calculate MSE for this component
        mse = np.mean((predicted_components[i] - target_components[i]) ** 2)

        # Plot target and predicted
        axes[i].plot(t, target_components[i, :num_samples_to_show],
                    'b-', label='Target (Ground Truth)', linewidth=2.5, alpha=0.7)
        axes[i].plot(t, predicted_components[i, :num_samples_to_show],
                    'r--', label='LSTM Predicted', linewidth=2, alpha=0.8)

        axes[i].set_title(f'f{i+1} = {freq} Hz - Target vs Predicted (MSE: {mse:.6f})',
                         fontsize=13, fontweight='bold')
        axes[i].set_ylabel('Amplitude', fontsize=11)
        axes[i].legend(loc='upper right', fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim([0, time_window])

    axes[-1].set_xlabel('Time (seconds)', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] Per-frequency comparison plots saved to: {save_path}")

    plt.close()


def create_extended_overlay_plot(
    mixed_signal: np.ndarray,
    predicted_components: np.ndarray,
    target_components: np.ndarray,
    frequencies: np.ndarray,
    sampling_rate: int = 1000,
    time_window: float = 5.0,
    save_path: str = None
):
    """
    Create an extended overlay plot for f₂ with longer time window.
    """
    f2_idx = 1
    f2_freq = frequencies[f2_idx]

    num_samples_to_show = int(time_window * sampling_rate)
    t = np.arange(num_samples_to_show) / sampling_rate

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # Top plot: All three signals
    ax1.plot(t, target_components[f2_idx, :num_samples_to_show],
             'b-', label=f'Target (Pure {f2_freq} Hz)', linewidth=2.5, alpha=0.8)
    ax1.plot(t, mixed_signal[:num_samples_to_show],
             'gray', label='Noisy Mixed Signal S(t)', linewidth=1.5, alpha=0.5)
    ax1.plot(t, predicted_components[f2_idx, :num_samples_to_show],
             'r--', label=f'LSTM Output ({f2_freq} Hz)', linewidth=2, alpha=0.8)

    mse = np.mean((predicted_components[f2_idx] - target_components[f2_idx]) ** 2)
    ax1.set_title(f'Extended View: f₂ = {f2_freq} Hz (MSE: {mse:.6f})',
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, time_window])

    # Bottom plot: Target vs LSTM output only (clearer comparison)
    ax2.plot(t, target_components[f2_idx, :num_samples_to_show],
             'b-', label=f'Target (Pure {f2_freq} Hz)', linewidth=2.5, alpha=0.8)
    ax2.plot(t, predicted_components[f2_idx, :num_samples_to_show],
             'r--', label=f'LSTM Output ({f2_freq} Hz)', linewidth=2, alpha=0.8)

    ax2.set_title(f'Target vs LSTM Output Comparison',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Amplitude', fontsize=12)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, time_window])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] Extended f2 overlay plot saved to: {save_path}")

    plt.close()


def main():
    """Main function to create all assignment plots."""
    print("=" * 70)
    print("Creating Assignment Plots")
    print("=" * 70)

    # Load data
    data_path = 'outputs/results/reconstructed_signals.npz'

    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        print("Please run evaluate.py first!")
        return

    print(f"\nLoading data from: {data_path}")
    data = np.load(data_path)

    mixed_signal = data['mixed_signal']
    predicted_components = data['predicted_components']
    target_components = data['target_components']
    frequencies = data['frequencies']

    print(f"  - Mixed signal shape: {mixed_signal.shape}")
    print(f"  - Predicted components shape: {predicted_components.shape}")
    print(f"  - Frequencies: {frequencies}")

    # Create output directory
    output_dir = 'outputs/plots/assignment'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving plots to: {output_dir}")

    print("\n" + "=" * 70)
    print("1. Creating overlay plot for f2...")
    print("=" * 70)
    create_f2_overlay_plot(
        mixed_signal,
        predicted_components,
        target_components,
        frequencies,
        time_window=2.0,
        save_path=os.path.join(output_dir, 'f2_overlay_2sec.png')
    )

    print("\n" + "=" * 70)
    print("2. Creating extended overlay plot for f2...")
    print("=" * 70)
    create_extended_overlay_plot(
        mixed_signal,
        predicted_components,
        target_components,
        frequencies,
        time_window=5.0,
        save_path=os.path.join(output_dir, 'f2_overlay_5sec.png')
    )

    print("\n" + "=" * 70)
    print("3. Creating per-frequency comparison plots...")
    print("=" * 70)
    create_per_frequency_plots(
        predicted_components,
        target_components,
        frequencies,
        time_window=2.0,
        save_path=os.path.join(output_dir, 'all_frequencies_comparison.png')
    )

    print("\n" + "=" * 70)
    print("Assignment plots created successfully!")
    print("=" * 70)
    print(f"\nAll plots saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  1. f2_overlay_2sec.png - Overlay plot for f2 (2 seconds)")
    print("  2. f2_overlay_5sec.png - Extended overlay plot for f2 (5 seconds)")
    print("  3. all_frequencies_comparison.png - Separate plots for f1-f4")


if __name__ == "__main__":
    main()
