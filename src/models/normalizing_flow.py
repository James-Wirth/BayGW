import torch
from torch import nn
from pyro.distributions.transforms import AffineAutoregressive
from pyro.nn import AutoRegressiveNN

class NormalizingFlow(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_layers):
        super().__init__()
        self.transforms = nn.ModuleList([
            AffineAutoregressive(AutoRegressiveNN(input_dim, hidden_dims))
            for _ in range(num_layers)
        ])
        self.base_dist = torch.distributions.Normal(torch.zeros(input_dim), torch.ones(input_dim))

    def forward(self, x):
        log_jacobians = []
        for transform in self.transforms:
            y = transform(x)
            log_jacobian = transform.log_abs_det_jacobian(x, y)
            log_jacobians.append(log_jacobian)
            x = y
        return x, sum(log_jacobians)

    def inverse(self, z):
        for transform in reversed(self.transforms):
            z = transform.inv(z)
        return z
