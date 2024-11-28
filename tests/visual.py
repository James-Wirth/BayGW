import torch
import matplotlib.pyplot as plt
from src.data.dataset import GWDataset

def test_signal_visualization_with_full_inspiral():
    """Ensure the full signal, including low-amplitude inspiral, is being visualized correctly."""
    m1_range = (10, 20)
    m2_range = (10, 20)
    num_samples = 1  # Load one sample for testing
    dataset = GWDataset(num_samples, m1_range, m2_range)

    # Retrieve a single sample (gravitational wave signal)
    signal = dataset[0]  # GWDataset automatically generates signals

    # Check the first few samples to ensure the signal starts at a low amplitude
    print("First few samples:", signal[:20].numpy())

    # Plot the full signal to confirm the full duration
    plt.figure(figsize=(10, 5))
    plt.plot(signal.numpy())
    plt.title(f"Gravitational Wave Signal (Full Duration)")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.show()

