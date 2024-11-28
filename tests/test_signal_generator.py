import pytest
import numpy as np
from src.data.signal_generator import SignalGenerator
from src.data.preprocessing import Preprocessor
from unittest.mock import MagicMock

@pytest.fixture
def config():
    class Config:
        SIGNAL_LENGTH = 1024
        SAMPLING_RATE = 4096
        NUM_SIGNALS = 10
        MASS1_RANGE = (1.0, 100.0)
        MASS2_RANGE = (1.0, 100.0)
        TARGET_DIM = 1024
    return Config()

@pytest.fixture
def preprocessor(config):
    return Preprocessor(config.SIGNAL_LENGTH, config.SAMPLING_RATE, config.TARGET_DIM)

@pytest.fixture
def signal_generator(config, preprocessor):
    return SignalGenerator(config, preprocessor)

def test_generate_signal_shape(signal_generator):
    mass1, mass2 = 30.0, 30.0
    signal = signal_generator.generate_signal(mass1, mass2)
    assert len(signal) == 1024

def test_generate_signals_shape(signal_generator):
    signals = signal_generator.generate_signals(10, (1.0, 50.0), (1.0, 50.0))
    assert signals.shape == (10, 1024)

def test_mass_range(signal_generator):
    mass1, mass2 = 30.0, 30.0
    signal = signal_generator.generate_signal(mass1, mass2)
    assert np.min(signal) >= -1e-15, f"Signal minimum is {np.min(signal)} which is below tolerance"
