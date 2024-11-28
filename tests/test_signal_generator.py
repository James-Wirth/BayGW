import unittest
from src.data.signal_generator import SignalGenerator
from src.config import Config
import matplotlib.pyplot as plt

class TestSignalGenerator(unittest.TestCase):
    def test_generate_signal(self):
        config = Config()
        gen = SignalGenerator(config)
        signal = gen.generate_signal(30, 30)
        self.assertTrue(len(signal) > 0)

    def test_plot_signal(self):
        config = Config()
        gen = SignalGenerator(config)
        signal = gen.generate_signal(30, 30)  # Example masses
        print(f"Signal length: {len(signal)}, Min: {signal.min()}, Max: {signal.max()}")

        plt.figure(figsize=(10, 4))
        plt.plot(signal)
        plt.title("Generated Gravitational Wave Signal")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.show()

if __name__ == "__main__":
    unittest.main()
