from src.data.preprocessing import Preprocessor
import unittest
import numpy as np

class TestNormalizingFlow(unittest.TestCase):
    def test_preprocessing(self):
        preprocessor = Preprocessor(signal_length=1.0, sampling_rate=4096)
        raw_signal = np.random.randn(4100)  # Example signal
        processed_signal = preprocessor.preprocess(raw_signal)

        print(f"Processed Signal Shape: {processed_signal.shape}")
        print(f"Processed Signal Mean: {processed_signal.mean()}, Std: {processed_signal.std()}")
