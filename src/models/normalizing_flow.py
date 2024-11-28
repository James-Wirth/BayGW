import torch
from torch import nn
from pyro.distributions.transforms import AffineAutoregressive, SplineAutoregressive, BatchNorm
from pyro.nn import AutoRegressiveNN

def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class ScalingLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        return x * torch.exp(self.log_scale)

class NormalizingFlow(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_layers, use_spline=False):
        super().__init__()
        Transform = SplineAutoregressive if use_spline else AffineAutoregressive

        self.transforms = nn.ModuleList([
            ScalingLayer(input_dim),
            *[nn.Sequential(BatchNorm(input_dim), Transform(AutoRegressiveNN(input_dim, hidden_dims)))
              for _ in range(num_layers)]
        ])

        self.base_dist = torch.distributions.MultivariateNormal(
            torch.zeros(input_dim), torch.eye(input_dim)
        )

    def forward(self, x):
        log_jacobians = []
        for transform in self.transforms:
            y = transform(x)
            log_jacobian = (
                transform.log_abs_det_jacobian(x, y)
                if hasattr(transform, "log_abs_det_jacobian")
                else torch.zeros_like(x)
            )
            log_jacobians.append(log_jacobian)
            x = y
        total_log_jacobian = torch.stack(log_jacobians).sum(dim=0)
        return x, total_log_jacobian

    def inverse(self, z):
        for transform in reversed(self.transforms):
            z = transform.inv(z) if hasattr(transform, "inv") else z
        return z
