import pytest
import numpy as np
import torch
from src.data.preprocessing import Preprocessor


@pytest.fixture
def config():
    class Config:
        SIGNAL_LENGTH = 1024
        SAMPLING_RATE = 4096
        TARGET_DIM = 1024

    return Config()


@pytest.fixture
def preprocessor(config):
    return Preprocessor(config.SIGNAL_LENGTH, config.SAMPLING_RATE, config.TARGET_DIM)


def test_pad_or_truncate(preprocessor):
    signal = np.random.randn(500)
    processed_signal = preprocessor.pad_or_truncate(signal)
    assert len(processed_signal) == 1024

    signal = np.random.randn(1500)  # Long signal
    processed_signal = preprocessor.pad_or_truncate(signal)
    assert len(processed_signal) == 1024


def test_preprocess_normalization(preprocessor):
    signal = np.random.randn(1024)
    processed_signal = preprocessor.preprocess(signal)

    assert torch.abs(processed_signal.mean()).item() < 1e-5

    tolerance = 1e-3
    assert torch.abs(processed_signal.std() - 1.0).item() < tolerance, \
        f"Expected std close to 1.0, but got {processed_signal.std().item()}"
