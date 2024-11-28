import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalizingFlow(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, n_flows=5):
        """
        input_dim: Dimension of input data (signal length)
        hidden_dim: Size of hidden layers in the flow model
        n_flows: Number of transformations (flow layers) to use
        """
        super(NormalizingFlow, self).__init__()

        self.input_dim = input_dim
        self.n_flows = n_flows
        self.flows = nn.ModuleList()

        for _ in range(n_flows):
            self.flows.append(FlowLayer(input_dim, hidden_dim))

    def forward(self, x):
        """
        x: Input tensor of shape (batch_size, input_dim)

        :return: Transformed tensor and log determinant of Jacobian
        """
        log_det_Jacobian = 0

        for flow in self.flows:
            x, log_det = flow(x)
            log_det_Jacobian += log_det

        return x, log_det_Jacobian

    def reverse(self, z):
        """
        z: Tensor of transformed data (latent space)

        :return: Reversed data (original space)
        """
        for flow in reversed(self.flows):
            z = flow.reverse(z)
        return z

    def sample(self, num_samples):
        """
        num_samples: Number of samples to generate

        :return: Tensor of shape (num_samples, input_dim) with new samples
        """
        z = torch.randn(num_samples, self.input_dim)
        return self.reverse(z)


class FlowLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        input_dim: Dimension of input data (signal length)
        hidden_dim: Size of hidden layers in the flow layer
        """
        super(FlowLayer, self).__init__()

        self.input_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim // 2),
            nn.Tanh()
        )
        self.mask = self.create_mask(input_dim)

    def forward(self, x):
        """
        x: Input tensor of shape (batch_size, input_dim)

        :return: Transformed tensor and log determinant of Jacobian
        """
        x1 = x * self.mask
        x2 = x * (1 - self.mask)
        s = self.net(x2)
        y = x1 + s

        log_det_Jacobian = torch.sum(torch.log(torch.abs(1 + s)), dim=1)
        return y, log_det_Jacobian

    def reverse(self, y):
        """
        y: Transformed tensor

        :return: Reversed data
        """
        y1 = y * self.mask
        y2 = y * (1 - self.mask)

        s = self.net(y2)

        x = y1 - s
        return x

    def create_mask(self, input_dim):
        """
        input_dim: Dimension of input data

        :return: Mask tensor (binary mask)
        """
        mask = torch.zeros(input_dim)
        mask[::2] = 1
        return mask
