import torch
import matplotlib.pyplot as plt
from src.models.normalizing_flow import NormalizingFlow
from src.data.dataset import GWDataset

# Define the model parameters (should match the ones used during training)
input_dim = 4096  # Length of the signal after padding/trimming (duration * sample_rate)
hidden_dim = 512
num_flows = 4

# Load the trained model
model = NormalizingFlow(input_dim, hidden_dim, num_flows)
model.load_state_dict(torch.load('normalizing_flow_model.pth'))
model.eval()  # Set the model to evaluation mode

# Load the dataset
m1_range = (10, 20)
m2_range = (10, 20)
num_samples = 10  # Let's generate a few samples
train_data = GWDataset(num_samples, m1_range, m2_range, target_length=input_dim)
3
# Generate predictions from the trained model
num_samples = 5  # You can change this as needed
generated_signals = model.sample(num_samples)

# Pick a random real signal from the dataset for comparison
real_signal_idx = 0  # Change this index to compare different signals
real_signal = train_data[real_signal_idx]

# Plot the comparison of real vs generated signals
plt.figure(figsize=(12, 6))

# Plot the real signal
plt.subplot(1, 2, 1)
plt.plot(real_signal.numpy(), label="Real Signal")
plt.title("Real Gravitational Wave Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()

# Plot a generated signal
plt.subplot(1, 2, 2)
generated_signal = generated_signals[0].detach().cpu().numpy()  # Pick the first generated signal
plt.plot(generated_signal, label="Generated Signal", color='r')
plt.title("Generated Signal by Normalizing Flow")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.show()

import numpy as np
from scipy.signal import spectrogram

def plot_spectrogram(signal, title="Spectrogram"):
    f, t, Sxx = spectrogram(signal, fs=2048)  # Sampling rate is 2048 Hz
    plt.pcolormesh(t, f, 10 * np.log10(Sxx))  # Plot in dB scale
    plt.colorbar(label="Power (dB)")
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title(title)

# Plot spectrograms for real and generated signals
plt.figure(figsize=(12, 6))

# Spectrogram of the real signal
plt.subplot(1, 2, 1)
plot_spectrogram(real_signal.numpy(), title="Real Signal Spectrogram")

# Spectrogram of the generated signal
plt.subplot(1, 2, 2)
plot_spectrogram(generated_signal, title="Generated Signal Spectrogram")

plt.tight_layout()
plt.show()

