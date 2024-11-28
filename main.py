import torch
import numpy as np
from src.data.signal_generator import SignalGenerator
from src.data.preprocessing import Preprocessor
from src.training.trainer import NormalizingFlow, train_flow

class Config:
    SIGNAL_LENGTH = 1024                # Length of the signal in samples
    SAMPLING_RATE = 4096                # Sampling rate (Hz)
    NUM_SIGNALS = 1000                  # Number of signals to generate
    MASS1_RANGE = (1.0, 100.0)          # Mass range for the first object (in solar masses)
    MASS2_RANGE = (1.0, 100.0)          # Mass range for the second object (in solar masses)
    TARGET_DIM = 1024                   # Target dimension for padding/truncating the signal
    HIDDEN_DIM = 512                    # Hidden dimension for the normalizing flow model


def main():
    config = Config()
    preprocessor = Preprocessor(config.SIGNAL_LENGTH, config.SAMPLING_RATE, config.TARGET_DIM)
    signal_generator = SignalGenerator(config, preprocessor)
    signals = signal_generator.generate_signals(config.NUM_SIGNALS, config.MASS1_RANGE, config.MASS2_RANGE)

    processed_signals = [preprocessor.preprocess(signal) for signal in signals]
    processed_signals = torch.stack(processed_signals)

    model = NormalizingFlow(input_dim=config.TARGET_DIM, hidden_dim=config.HIDDEN_DIM)
    train_flow(model, processed_signals, batch_size=64, lr=1e-3, epochs=10)


if __name__ == "__main__":
    main()
