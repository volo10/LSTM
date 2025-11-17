"""
Visualization Script for LSTM Frequency Extraction

This module creates various plots to visualize training progress and
model performance.
"""

import glob
import json
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(history: Dict, save_path: Optional[str] = None):
    """
    Plot training and validation loss curves.

    Args:
        history: Dictionary with 'train_loss' and 'test_loss' lists
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    ax1.plot(epochs, history["train_loss"], "b-", label="Training Loss", linewidth=2)
    ax1.plot(epochs, history["test_loss"], "r-", label="Test Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("MSE Loss", fontsize=12)
    ax1.set_title("Training and Test Loss", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Log scale loss curves
    ax2.semilogy(
        epochs, history["train_loss"], "b-", label="Training Loss", linewidth=2
    )
    ax2.semilogy(epochs, history["test_loss"], "r-", label="Test Loss", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("MSE Loss (log scale)", fontsize=12)
    ax2.set_title("Training and Test Loss (Log Scale)", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved to: {save_path}")

    plt.show()


def plot_signal_reconstruction(
    mixed_signal: np.ndarray,
    predicted_components: np.ndarray,
    target_components: np.ndarray,
    frequencies: List[float],
    sampling_rate: int = 1000,
    time_window: float = 2.0,
    save_path: Optional[str] = None,
):
    """
    Plot reconstructed frequency components vs. ground truth.

    Args:
        mixed_signal: Noisy mixed signal, shape (num_samples,)
        predicted_components: Predicted components, shape (num_frequencies, num_samples)
        target_components: Target components, shape (num_frequencies, num_samples)
        frequencies: List of frequency values
        sampling_rate: Sampling rate in Hz
        time_window: Time window to display in seconds
        save_path: Optional path to save figure
    """
    num_frequencies = len(frequencies)
    num_samples_to_show = int(time_window * sampling_rate)
    t = np.arange(num_samples_to_show) / sampling_rate

    fig, axes = plt.subplots(num_frequencies + 1, 1, figsize=(14, 12))

    # Plot mixed signal
    axes[0].plot(t, mixed_signal[:num_samples_to_show], "k-", alpha=0.6, linewidth=1)
    axes[0].set_title("Mixed Noisy Signal", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Amplitude", fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, time_window])

    # Plot each frequency component
    for i, freq in enumerate(frequencies):
        # Calculate MSE for this component
        mse = np.mean((predicted_components[i] - target_components[i]) ** 2)

        # Plot target and predicted
        axes[i + 1].plot(
            t,
            target_components[i, :num_samples_to_show],
            "b-",
            label="Target",
            linewidth=2,
            alpha=0.7,
        )
        axes[i + 1].plot(
            t,
            predicted_components[i, :num_samples_to_show],
            "r--",
            label="Predicted",
            linewidth=2,
            alpha=0.7,
        )

        axes[i + 1].set_title(
            f"{freq} Hz Component (MSE: {mse:.6f})", fontsize=12, fontweight="bold"
        )
        axes[i + 1].set_ylabel("Amplitude", fontsize=10)
        axes[i + 1].legend(loc="upper right", fontsize=9)
        axes[i + 1].grid(True, alpha=0.3)
        axes[i + 1].set_xlim([0, time_window])

    axes[-1].set_xlabel("Time (seconds)", fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Signal reconstruction plot saved to: {save_path}")

    plt.show()


def plot_per_frequency_metrics(
    train_mse: List[float],
    test_mse: List[float],
    frequencies: List[float],
    save_path: Optional[str] = None,
):
    """
    Plot per-frequency MSE comparison.

    Args:
        train_mse: List of training MSE per frequency
        test_mse: List of test MSE per frequency
        frequencies: List of frequency values
        save_path: Optional path to save figure
    """
    x = np.arange(len(frequencies))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(
        x - width / 2,
        train_mse,
        width,
        label="Training",
        color="skyblue",
        edgecolor="black",
        linewidth=1.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        test_mse,
        width,
        label="Test",
        color="salmon",
        edgecolor="black",
        linewidth=1.5,
    )

    ax.set_xlabel("Frequency (Hz)", fontsize=12, fontweight="bold")
    ax.set_ylabel("MSE", fontsize=12, fontweight="bold")
    ax.set_title("Per-Frequency MSE Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f} Hz" for f in frequencies])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Per-frequency metrics plot saved to: {save_path}")

    plt.show()


def plot_error_distribution(
    predicted_components: np.ndarray,
    target_components: np.ndarray,
    frequencies: List[float],
    save_path: Optional[str] = None,
):
    """
    Plot error distribution for each frequency.

    Args:
        predicted_components: Predicted components, shape (num_frequencies, num_samples)
        target_components: Target components, shape (num_frequencies, num_samples)
        frequencies: List of frequency values
        save_path: Optional path to save figure
    """
    num_frequencies = len(frequencies)
    errors = predicted_components - target_components

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, freq in enumerate(frequencies):
        error = errors[i]

        # Histogram
        axes[i].hist(error, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
        axes[i].axvline(0, color="red", linestyle="--", linewidth=2, label="Zero Error")

        # Statistics
        mean_error = np.mean(error)
        std_error = np.std(error)

        axes[i].set_title(
            f"{freq} Hz - Error Distribution\n"
            f"Mean: {mean_error:.6f}, Std: {std_error:.6f}",
            fontsize=11,
            fontweight="bold",
        )
        axes[i].set_xlabel("Prediction Error", fontsize=10)
        axes[i].set_ylabel("Frequency", fontsize=10)
        axes[i].legend(fontsize=9)
        axes[i].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Error distribution plot saved to: {save_path}")

    plt.show()


def plot_error_over_time(
    predicted_components: np.ndarray,
    target_components: np.ndarray,
    frequencies: List[float],
    sampling_rate: int = 1000,
    time_window: float = 2.0,
    save_path: Optional[str] = None,
):
    """
    Plot prediction error over time for each frequency.

    Args:
        predicted_components: Predicted components, shape (num_frequencies, num_samples)
        target_components: Target components, shape (num_frequencies, num_samples)
        frequencies: List of frequency values
        sampling_rate: Sampling rate in Hz
        time_window: Time window to display in seconds
        save_path: Optional path to save figure
    """
    num_samples_to_show = int(time_window * sampling_rate)
    t = np.arange(num_samples_to_show) / sampling_rate
    errors = predicted_components - target_components

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, freq in enumerate(frequencies):
        error = errors[i, :num_samples_to_show]

        axes[i].plot(t, error, "purple", linewidth=1.5, alpha=0.7)
        axes[i].axhline(0, color="red", linestyle="--", linewidth=2)
        axes[i].fill_between(t, error, 0, alpha=0.3, color="purple")

        axes[i].set_title(
            f"{freq} Hz - Error Over Time", fontsize=11, fontweight="bold"
        )
        axes[i].set_xlabel("Time (seconds)", fontsize=10)
        axes[i].set_ylabel("Prediction Error", fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim([0, time_window])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Error over time plot saved to: {save_path}")

    plt.show()


def create_all_plots(
    history_path: str,
    reconstruction_path: str,
    results_path: str,
    output_dir: str = "outputs/plots",
):
    """
    Create all visualization plots.

    Args:
        history_path: Path to training history JSON
        reconstruction_path: Path to reconstructed signals NPZ
        results_path: Path to evaluation results JSON
        output_dir: Directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("Creating visualizations...")
    print("=" * 70)

    # Load data
    with open(history_path, "r") as f:
        history_data = json.load(f)
        history = history_data["history"]

    with open(results_path, "r") as f:
        results = json.load(f)

    reconstruction_data = np.load(reconstruction_path)
    mixed_signal = reconstruction_data["mixed_signal"]
    predicted_components = reconstruction_data["predicted_components"]
    target_components = reconstruction_data["target_components"]
    frequencies = reconstruction_data["frequencies"].tolist()

    # 1. Training curves
    print("\n1. Plotting training curves...")
    plot_training_curves(
        history, save_path=os.path.join(output_dir, "training_curves.png")
    )

    # 2. Signal reconstruction
    print("\n2. Plotting signal reconstruction...")
    plot_signal_reconstruction(
        mixed_signal,
        predicted_components,
        target_components,
        frequencies,
        save_path=os.path.join(output_dir, "signal_reconstruction.png"),
    )

    # 3. Per-frequency metrics
    print("\n3. Plotting per-frequency metrics...")
    plot_per_frequency_metrics(
        results["train_metrics"]["per_freq_mse"],
        results["test_metrics"]["per_freq_mse"],
        frequencies,
        save_path=os.path.join(output_dir, "per_frequency_mse.png"),
    )

    # 4. Error distribution
    print("\n4. Plotting error distribution...")
    plot_error_distribution(
        predicted_components,
        target_components,
        frequencies,
        save_path=os.path.join(output_dir, "error_distribution.png"),
    )

    # 5. Error over time
    print("\n5. Plotting error over time...")
    plot_error_over_time(
        predicted_components,
        target_components,
        frequencies,
        save_path=os.path.join(output_dir, "error_over_time.png"),
    )

    print("\n" + "=" * 70)
    print(f"All plots saved to: {output_dir}")
    print("Visualization complete!")


def main():
    """Main visualization function."""
    # Paths
    log_dir = "outputs/logs"
    results_dir = "outputs/results"
    plots_dir = "outputs/plots"

    # Find most recent history file
    history_files = glob.glob(os.path.join(log_dir, "training_history_*.json"))
    if not history_files:
        print("Error: No training history files found!")
        return

    history_path = max(history_files, key=os.path.getctime)
    reconstruction_path = os.path.join(results_dir, "reconstructed_signals.npz")
    results_path = os.path.join(results_dir, "evaluation_results.json")

    # Check if files exist
    if not os.path.exists(reconstruction_path):
        print(f"Error: Reconstruction file not found: {reconstruction_path}")
        print("Please run evaluate.py first!")
        return

    if not os.path.exists(results_path):
        print(f"Error: Results file not found: {results_path}")
        print("Please run evaluate.py first!")
        return

    print(f"Using training history: {history_path}")
    print(f"Using reconstruction: {reconstruction_path}")
    print(f"Using results: {results_path}")

    # Create all plots
    create_all_plots(history_path, reconstruction_path, results_path, plots_dir)


if __name__ == "__main__":
    main()
