import torch
from src.models.normalizing_flow import NormalizingFlow
from src.data.dataset import GWDataset
from src.training.trainer import Trainer

def main():
    input_dim = 4096  # Length of the signal after padding/trimming (duration * sample_rate)
    hidden_dim = 512
    num_flows = 4
    num_samples = 1000
    batch_size = 64
    lr = 1e-3
    epochs = 10
    
    m1_range = (10, 20)
    m2_range = (10, 20)
    train_data = GWDataset(num_samples, m1_range, m2_range, target_length=input_dim)

    model = NormalizingFlow(input_dim, hidden_dim, num_flows)

    model_save_path = 'normalizing_flow_model.pth'
    trainer = Trainer(model, train_data, batch_size=batch_size, lr=lr, epochs=epochs, model_save_path=model_save_path)
    trainer.train()

if __name__ == "__main__":
    main()
