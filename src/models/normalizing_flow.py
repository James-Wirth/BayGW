import torch
import torch.nn as nn


class RealNVP(nn.Module):
    def __init__(self, input_dim, hidden_dim, conditioning_dim):
        super(RealNVP, self).__init__()

        self.input_dim = input_dim
        self.conditioning_dim = conditioning_dim
        self.hidden_dim = hidden_dim

        self.nn = nn.Sequential(
            nn.Linear(input_dim // 2 + conditioning_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim // 2 * 2),
        )

        for m in self.nn:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, conditioning):
        half_size = self.input_dim // 2
        x1, x2 = x[:, :half_size], x[:, half_size:]

        conditioned_input = torch.cat([x1, conditioning], dim=-1)

        shift_scale = self.nn(conditioned_input)
        shift, scale = shift_scale.chunk(2, dim=-1)

        z = x2 * torch.exp(scale) + shift
        log_det_jacobian = scale.sum(dim=-1)

        return torch.cat([x1, z], dim=-1), log_det_jacobian

    def inverse(self, z, conditioning):
        """
        Perform the inverse pass of the RealNVP layer.

        Parameters:
        - z: Latent variables (batch_size, input_dim)
        - conditioning: Conditioning parameters (batch_size, conditioning_dim)

        Returns:
        - x: Reconstructed input in data space
        """
        half_size = self.input_dim // 2
        z1, z2 = z[:, :half_size], z[:, half_size:]

        conditioned_input = torch.cat([z1, conditioning], dim=-1)

        shift_scale = self.nn(conditioned_input)
        shift, scale = shift_scale.chunk(2, dim=-1)

        x2 = (z2 - shift) * torch.exp(-scale)
        return torch.cat([z1, x2], dim=-1)

    def sample(self, num_samples, conditioning):
        """
        Generate samples conditioned on the given parameters.

        Parameters:
        - num_samples: Number of samples to generate
        - conditioning: Conditioning parameters (batch_size, conditioning_dim)

        Returns:
        - samples: Generated samples in data space
        """

        z = torch.randn(num_samples, self.input_dim, device=conditioning.device)
        samples = self.inverse(z, conditioning)
        return samples

    def nll_loss(self, x, conditioning):
        z, log_det_jacobian = self(x, conditioning)
        log_prob = -0.5 * torch.sum(z ** 2, dim=-1)
        loss = -log_prob - log_det_jacobian
        return loss.mean()
