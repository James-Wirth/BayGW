import unittest
from src.data.signal_generator import SignalGenerator
from src.config import Config

class TestSignalGenerator(unittest.TestCase):
    def test_generate_signal(self):
        config = Config()
        gen = SignalGenerator(config)
        signal = gen.generate_signal(30, 30)
        self.assertTrue(len(signal) > 0)

if __name__ == "__main__":
    unittest.main()
