import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # For the progress bar
from src.models.normalizing_flow import NormalizingFlow
from src.data.dataset import GWDataset

class Trainer:
    def __init__(self, model, train_data, batch_size=32, lr=1e-3, epochs=100, model_save_path=None):
        """
        Initialize the training setup.

        Parameters:
        - model: The normalizing flow model
        - train_data: The training dataset
        - batch_size: The batch size for training
        - lr: The learning rate
        - epochs: The number of epochs to train
        """
        self.model = model
        self.train_data = train_data
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.model_save_path = model_save_path

        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.MSELoss()

    def train(self):
        dataloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            total_loss = 0
            epoch_bar = tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{self.epochs}]", ncols=100)

            for batch in epoch_bar:
                self.optimizer.zero_grad()

                log_prob, log_det_jacobian = self.model(batch)

                loss = -log_prob.mean()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                epoch_bar.set_postfix(loss=total_loss / (epoch_bar.n + 1))

            print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {total_loss / len(dataloader):.4f}")

        # Save the model after training is complete
        if self.model_save_path:
            print(f"Saving model to {self.model_save_path}")
            torch.save(self.model.state_dict(), self.model_save_path)