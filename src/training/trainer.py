from torch.optim.lr_scheduler import StepLR
import torch
import tqdm

class Trainer:
    def __init__(self, model, optimizer, data_loader, config):
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.config = config
        self.scheduler = StepLR(optimizer, step_size=config.LR_SCHEDULER_STEP, gamma=config.LR_SCHEDULER_GAMMA)

    def train(self):
        self.model.train()
        for epoch in range(self.config.EPOCHS):
            total_loss = 0
            print(f"Epoch {epoch + 1}/{self.config.EPOCHS}")

            for batch in tqdm.tqdm(self.data_loader, desc="Training Batches"):
                batch = batch.to(self.config.DEVICE)

                batch += torch.randn_like(batch) * 0.01
                z, log_jacobians = self.model(batch)

                base_log_prob = self.model.base_dist.log_prob(z).sum(dim=-1)
                scaled_base_log_prob = base_log_prob / self.config.INPUT_DIM
                scaled_log_jacobians = log_jacobians / self.config.INPUT_DIM

                log_jacobian_penalty = torch.mean(torch.abs(scaled_log_jacobians))
                loss = torch.mean(-scaled_log_jacobians + scaled_base_log_prob) + 0.01 * log_jacobian_penalty

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()

                print(f"Epoch {epoch + 1}, Batch Loss: {loss.item()}")

            self.scheduler.step()
