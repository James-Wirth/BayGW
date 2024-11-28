import torch
import torch.nn as nn
import torch.nn.functional as F


class RealNVP(nn.Module):
    def __init__(self, input_dim, hidden_dim, conditioning_dim):
        super(RealNVP, self).__init__()

        self.input_dim = input_dim
        self.conditioning_dim = conditioning_dim
        self.hidden_dim = hidden_dim

        # Neural network for computing shift and scale
        self.nn = nn.Sequential(
            nn.Linear(input_dim // 2 + conditioning_dim, hidden_dim),
            nn.LeakyReLU(),  # LeakyReLU to avoid dead neurons
            nn.Linear(hidden_dim, input_dim // 2 * 2),  # Output size is input_dim (for shift and scale)
        )

        # Weight initialization (Xavier/Glorot initialization is a good default)
        for m in self.nn:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, conditioning):
        half_size = self.input_dim // 2
        x1, x2 = x[:, :half_size], x[:, half_size:]

        # Concatenate conditioning information to the first part of the input
        conditioned_input = torch.cat([x1, conditioning], dim=-1)

        # Pass through the network to get shift and scale
        shift_scale = self.nn(conditioned_input)  # This should have size (batch_size, input_dim)

        # Split the output into shift and scale, both should have size (batch_size, input_dim // 2)
        shift, scale = shift_scale.chunk(2, dim=-1)

        # Apply the transformation to x2
        z = x2 * torch.exp(scale) + shift
        log_det_jacobian = scale.sum(dim=-1)

        return torch.cat([x1, z], dim=-1), log_det_jacobian

    def nll_loss(self, x, conditioning):
        # Forward pass
        z, log_det_jacobian = self(x, conditioning)

        # Assuming a standard Gaussian prior (N(0, I) for the latent z)
        log_prob = -0.5 * torch.sum(z ** 2, dim=-1)  # log p(z) for each sample

        # Combine log-likelihood of z and log-det-Jacobian from flow transformation
        loss = -log_prob - log_det_jacobian

        return loss.mean()  # Return the mean loss over the batch
