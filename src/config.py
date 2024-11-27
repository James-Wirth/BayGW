import torch

class Config:
    SEED = 42
    SAVE_DIR = "output/"

    SAMPLING_RATE = 4096
    SIGNAL_LENGTH = 1.0  # seconds

    INPUT_DIM = int(SAMPLING_RATE * SIGNAL_LENGTH)  # Matches signal dimension
    HIDDEN_DIMS = [INPUT_DIM, INPUT_DIM]
    FLOW_LAYERS = 10

    LEARNING_RATE = 1e-3
    BATCH_SIZE = 32
    EPOCHS = 100

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
