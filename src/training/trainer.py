from tqdm import tqdm
import torch


class Trainer:
    def __init__(self, model, optimizer, data_loader, config):
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.config = config

    def train(self):
        self.model.train()
        for epoch in range(self.config.EPOCHS):
            total_loss = 0
            print(f"Epoch {epoch + 1}/{self.config.EPOCHS}")

            for batch in tqdm(self.data_loader, desc="Training Batches"):
                batch = batch.to(self.config.DEVICE)
                z, log_jacobians = self.model(batch)
                loss = -torch.mean(log_jacobians + torch.distributions.Normal(0, 1).log_prob(z).sum(dim=-1))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1} Loss: {total_loss / len(self.data_loader):.4f}")
