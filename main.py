from tqdm import tqdm
import torch
from src.config import Config
from src.data.signal_generator import SignalGenerator
from src.data.dataset import GWSignalDataset
from src.models.normalizing_flow import NormalizingFlow
from src.training.trainer import Trainer
from src.data.preprocessing import Preprocessor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def main():
    config = Config()
    preprocessor = Preprocessor(signal_length=config.SIGNAL_LENGTH, sampling_rate=config.SAMPLING_RATE)

    signal_gen = SignalGenerator(config)
    print("Generating signals...")
    signals = [preprocessor.preprocess(signal_gen.generate_signal(30, 30)) for _ in tqdm(range(1000))]

    dataset = GWSignalDataset(signals)
    data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    model = NormalizingFlow(input_dim=config.INPUT_DIM, hidden_dims=config.HIDDEN_DIMS, num_layers=config.FLOW_LAYERS)
    model.to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print("Starting training...")
    trainer = Trainer(model, optimizer, data_loader, config)
    trainer.train()

    model_save_path = "output/trained_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

def load_model(model_path, config):
    model = NormalizingFlow(input_dim=config.INPUT_DIM, hidden_dims=config.HIDDEN_DIMS, num_layers=config.FLOW_LAYERS)
    model.load_state_dict(torch.load(model_path))
    model.to(config.DEVICE)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

def sample_signals(model, num_samples, config):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, config.INPUT_DIM).to(config.DEVICE)
        generated_signals = model.inverse(z)
    return generated_signals.cpu().numpy()

if __name__ == "__main__":
    main()

    trained_model_path = "output/trained_model.pth"
    trained_model = load_model(trained_model_path, Config())

    generated_signals = sample_signals(trained_model, num_samples=5, config=Config())
    for i, signal in enumerate(generated_signals):
        plt.figure(figsize=(10, 4))
        plt.plot(signal)
        plt.title(f"Generated Signal {i + 1}")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.show()
