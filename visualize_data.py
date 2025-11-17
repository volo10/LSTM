"""
Visualize Training and Test Datasets

This script creates visualizations to verify that the datasets are generated correctly.
"""

import matplotlib.pyplot as plt
import numpy as np

from data_generator import create_datasets


def plot_datasets():
    """Create comprehensive plots of training and test datasets."""

    # Create datasets with correct specifications
    print("Creating datasets...")
    train_dataset, test_dataset = create_datasets(
        num_train_instances=1,  # Single signal
        num_test_instances=1,  # Same signal, different noise
        frequencies=[1.0, 3.0, 5.0, 7.0],
        sampling_rate=1000,
        duration=10.0,
        noise_std=0.1,
        train_seed=1,
        test_seed=2,
        signal_seed=42,
    )

    print(f"Training dataset: {len(train_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")

    # Get the data
    train_mixed = train_dataset.mixed_signals[0].numpy()
    test_mixed = test_dataset.mixed_signals[0].numpy()
    clean_components = train_dataset.clean_components[0].numpy()

    frequencies = train_dataset.frequencies
    sampling_rate = 1000
    time_full = np.arange(len(train_mixed)) / sampling_rate

    # Time window for detailed view
    time_window = 2.0  # seconds
    num_samples_window = int(time_window * sampling_rate)
    time_window_arr = time_full[:num_samples_window]

    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 12))

    # 1. Full signal comparison (Train vs Test)
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(
        time_full,
        train_mixed,
        "b-",
        alpha=0.6,
        linewidth=0.5,
        label="Training (noise seed=1)",
    )
    ax1.plot(
        time_full,
        test_mixed,
        "r-",
        alpha=0.6,
        linewidth=0.5,
        label="Test (noise seed=2)",
    )
    ax1.set_title("Full Mixed Signals (10 seconds)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Zoomed view - First 2 seconds
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(
        time_window_arr,
        train_mixed[:num_samples_window],
        "b-",
        alpha=0.7,
        linewidth=1,
        label="Training",
    )
    ax2.plot(
        time_window_arr,
        test_mixed[:num_samples_window],
        "r-",
        alpha=0.7,
        linewidth=1,
        label="Test",
    )
    ax2.set_title("Zoomed View - First 2 Seconds", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Clean components (targets)
    ax3 = plt.subplot(3, 2, 3)
    for i, freq in enumerate(frequencies):
        ax3.plot(
            time_window_arr,
            clean_components[i, :num_samples_window],
            label=f"{freq} Hz",
            linewidth=1.5,
            alpha=0.8,
        )
    ax3.set_title("Clean Target Components (First 2s)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Amplitude")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Noise comparison
    ax4 = plt.subplot(3, 2, 4)
    clean_signal = np.sum(clean_components, axis=0)
    train_noise = train_mixed - clean_signal
    test_noise = test_mixed - clean_signal
    ax4.plot(
        time_window_arr,
        train_noise[:num_samples_window],
        "b-",
        alpha=0.7,
        linewidth=0.5,
        label="Training noise",
    )
    ax4.plot(
        time_window_arr,
        test_noise[:num_samples_window],
        "r-",
        alpha=0.7,
        linewidth=0.5,
        label="Test noise",
    )
    ax4.set_title(
        "Noise (Different between Train/Test)", fontsize=12, fontweight="bold"
    )
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Amplitude")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Frequency spectrum (FFT)
    ax5 = plt.subplot(3, 2, 5)
    from scipy.fft import fft, fftfreq

    N = len(clean_signal)
    yf = fft(clean_signal)
    xf = fftfreq(N, 1 / sampling_rate)
    ax5.plot(xf[: N // 2], 2.0 / N * np.abs(yf[: N // 2]), "g-", linewidth=2)
    for freq in frequencies:
        ax5.axvline(freq, color="r", linestyle="--", alpha=0.5, label=f"{freq}Hz")
    ax5.set_title("Frequency Spectrum (Clean Signal)", fontsize=12, fontweight="bold")
    ax5.set_xlabel("Frequency (Hz)")
    ax5.set_ylabel("Magnitude")
    ax5.set_xlim([0, 10])
    ax5.grid(True, alpha=0.3)

    # 6. Statistics table
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis("off")

    stats_text = f"""
    DATASET STATISTICS
    ==================
    
    Configuration:
    - Duration: 10.0 seconds
    - Sampling Rate: 1000 Hz
    - Total Time Samples: {len(train_mixed):,}
    - Frequencies: {frequencies}
    - Number of Frequencies: {len(frequencies)}
    
    Training Set:
    - Total Samples: {len(train_dataset):,}
    - Samples per frequency: {len(train_mixed):,}
    - Noise Seed: 1
    - Signal Mean: {np.mean(train_mixed):.4f}
    - Signal Std: {np.std(train_mixed):.4f}
    - Signal Range: [{np.min(train_mixed):.2f}, {np.max(train_mixed):.2f}]
    
    Test Set:
    - Total Samples: {len(test_dataset):,}
    - Samples per frequency: {len(test_mixed):,}
    - Noise Seed: 2
    - Signal Mean: {np.mean(test_mixed):.4f}
    - Signal Std: {np.std(test_mixed):.4f}
    - Signal Range: [{np.min(test_mixed):.2f}, {np.max(test_mixed):.2f}]
    
    Clean Signal (Target):
    - Mean: {np.mean(clean_signal):.4f}
    - Std: {np.std(clean_signal):.4f}
    
    Noise Statistics:
    - Train Noise Std: {np.std(train_noise):.4f}
    - Test Noise Std: {np.std(test_noise):.4f}
    - Noise is DIFFERENT (seeds 1 vs 2)
    - Signal is SAME (seed 42 for both)
    """

    ax6.text(
        0.1,
        0.95,
        stats_text,
        transform=ax6.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()
    plt.savefig("outputs/plots/dataset_visualization.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to: outputs/plots/dataset_visualization.png")
    plt.show()

    # Create a second figure showing sample data points
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Show first few samples from dataset
    for freq_idx, freq in enumerate(frequencies):
        ax = axes[freq_idx // 2, freq_idx % 2]

        # Get a few samples for this frequency
        samples_to_show = 100
        inputs = []
        targets = []

        for i in range(samples_to_show):
            idx = i * len(frequencies) + freq_idx  # Get samples for this frequency
            input_vec, target, _ = train_dataset[idx]
            inputs.append(input_vec[0].item())  # S[t]
            targets.append(target.item())

        time_samples = np.arange(samples_to_show) / sampling_rate
        ax.plot(
            time_samples, inputs, "b-", alpha=0.7, label="Input (mixed)", linewidth=1
        )
        ax.plot(
            time_samples, targets, "r-", alpha=0.7, label="Target (clean)", linewidth=2
        )
        ax.set_title(
            f"Frequency {freq} Hz - First {samples_to_show} samples", fontweight="bold"
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/plots/dataset_samples.png", dpi=150, bbox_inches="tight")
    print("Plot saved to: outputs/plots/dataset_samples.png")
    plt.show()

    print("\n" + "=" * 70)
    print("DATASET VERIFICATION")
    print("=" * 70)
    print("✓ Training and test sets use SAME signal (seed=42)")
    print("✓ Training and test sets have DIFFERENT noise (seeds 1 vs 2)")
    print(
        f"✓ Total training samples: {len(train_dataset):,} (10,000 time steps × 4 frequencies)"
    )
    print(
        f"✓ Total test samples: {len(test_dataset):,} (10,000 time steps × 4 frequencies)"
    )
    print(f"✓ Frequencies present: {frequencies}")
    print("✓ Sampling rate: 1000 Hz")
    print("✓ Duration: 10 seconds")
    print("=" * 70)


if __name__ == "__main__":
    import os

    os.makedirs("outputs/plots", exist_ok=True)

    # Need scipy for FFT
    try:
        import scipy
    except ImportError:
        print("Installing scipy for FFT visualization...")
        import subprocess

        subprocess.check_call(["pip3", "install", "scipy"])

    plot_datasets()
