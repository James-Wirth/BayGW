import numpy as np
import torch


class Preprocessor:
    def __init__(self, signal_length, sampling_rate, target_dim=None):
        self.signal_length = signal_length
        self.sampling_rate = sampling_rate
        # Calculate target_dim if not provided
        self.target_dim = target_dim or signal_length

    def pad_or_truncate(self, signal):
        """
        Pads or truncates the signal to the desired length.
        :param signal: Input signal
        :return: Padded or truncated signal
        """
        if len(signal) > self.target_dim:
            return signal[:self.target_dim]
        else:
            return np.pad(signal, (0, self.target_dim - len(signal)), 'constant')

    def preprocess(self, signal):
        """
        Preprocesses a signal (padding, truncating, and normalizing).
        :param signal: Input signal
        :return: Preprocessed signal as a PyTorch tensor
        """
        signal = self.pad_or_truncate(signal)  # Ensure the signal is the right length
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)  # Normalize the signal
        signal = np.clip(signal, -5.0, 5.0)  # Clip extreme values for stability
        return torch.tensor(signal, dtype=torch.float32)  # Return as a tensor
