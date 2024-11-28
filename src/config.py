import torch

class Config:
    SEED = 42
    SAVE_DIR = "output/"

    SAMPLING_RATE = 4096
    SIGNAL_LENGTH = 1.0

    INPUT_DIM = int(SAMPLING_RATE * SIGNAL_LENGTH)
    HIDDEN_DIMS = [int(INPUT_DIM * 1.5), int(INPUT_DIM * 1.5)]

    FLOW_LAYERS = 5
    USE_SPLINE = True

    LEARNING_RATE = 5e-5
    LR_SCHEDULER_STEP = 10
    LR_SCHEDULER_GAMMA = 0.7

    BATCH_SIZE = 16
    EPOCHS = 50

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
