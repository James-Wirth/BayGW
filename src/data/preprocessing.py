import numpy as np
import torch

class Preprocessor:
    def __init__(self, signal_length, sampling_rate, target_dim=None):
        self.signal_length = signal_length
        self.sampling_rate = sampling_rate
        self.target_dim = target_dim or int(signal_length * sampling_rate)

    def pad_or_truncate(self, signal):
        if len(signal) > self.target_dim:
            return signal[:self.target_dim]
        else:
            return np.pad(signal, (0, self.target_dim - len(signal)), 'constant')

    def preprocess(self, signal):
        signal = self.pad_or_truncate(signal)
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)  # Normalize
        signal = np.clip(signal, -5.0, 5.0)  # Clip extreme values
        return torch.tensor(signal, dtype=torch.float32)
