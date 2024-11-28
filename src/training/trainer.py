import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class NormalizingFlow(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NormalizingFlow, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.net(x)


class Trainer:
    def __init__(self, model, data, batch_size=64, lr=1e-3, epochs=10):
        self.model = model
        self.data = data
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dataloader = DataLoader(TensorDataset(self.data), batch_size=self.batch_size, shuffle=True)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        self.model.to(self.device)

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in self.dataloader:
                signals = batch[0].to(self.device)

                self.optimizer.zero_grad()

                output = self.model(signals)

                loss = self.loss_fn(output, signals)
                total_loss += loss.item()

                loss.backward()
                self.optimizer.step()

            print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {total_loss / len(self.dataloader)}")


def train_flow(model, data, batch_size=64, lr=1e-3, epochs=10):
    trainer = Trainer(model, data, batch_size, lr, epochs)
    trainer.train()
