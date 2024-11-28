import torch
from torch.utils.data import Dataset


class GWSignalDataset(Dataset):
    def __init__(self, signals):
        self.signals = signals

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        return signal.clone().detach()
