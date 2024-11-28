from src.data.dataset import GWSignalDataset
import unittest
import torch

class TestNormalizingFlow(unittest.TestCase):
    def test_preprocessing(self):
        signals = [torch.randn(4096) for _ in range(10)]  # Example data
        dataset = GWSignalDataset(signals)
        print(f"Dataset Length: {len(dataset)}")
        print(f"First Signal Shape: {dataset[0].shape}")