import torch
from torch.utils.data import Dataset
from src.data.signal_generator import generate_gw_signal


class GWDataset(Dataset):
    """
    A custom dataset class to load gravitational wave signals.
    """

    def __init__(self, num_samples, m1_range, m2_range, f_lower=30.0, duration=4, sample_rate=2048, target_length=None):
        """
        Initialize the dataset with parameters for generating GW signals.

        Parameters:
        - num_samples: Number of samples to generate in the dataset
        - m1_range: Tuple for range of mass1 values (min, max)
        - m2_range: Tuple for range of mass2 values (min, max)
        - f_lower: Lower frequency cutoff for the waveform (default 30 Hz)
        - duration: Duration of the signal (default 4 seconds)
        - sample_rate: Sampling rate (default 2048 Hz)
        - target_length: The fixed length of each signal. If None, defaults to duration * sample_rate
        """
        self.num_samples = num_samples
        self.m1_range = m1_range
        self.m2_range = m2_range
        self.f_lower = f_lower
        self.duration = duration
        self.sample_rate = sample_rate

        # Set target length (default to duration * sample_rate)
        self.target_length = target_length if target_length is not None else int(duration * sample_rate)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        m1 = self.m1_range[0] + (self.m1_range[1] - self.m1_range[0]) * torch.rand(size=(1,))
        m2 = self.m2_range[0] + (self.m2_range[1] - self.m2_range[0]) * torch.rand(size=(1,))

        signal = generate_gw_signal(m1.item(), m2.item(), f_lower=self.f_lower,
                                    duration=self.duration, sample_rate=self.sample_rate)

        signal_length = len(signal)

        if signal_length < self.target_length:
            padding = self.target_length - signal_length
            signal = torch.cat([signal, torch.zeros(padding)], dim=0)
        elif signal_length > self.target_length:
            signal = signal[:self.target_length]

        return signal
