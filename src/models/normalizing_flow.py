import torch
import torch.nn as nn

class NormalizingFlow(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, n_flows=5):
        super(NormalizingFlow, self).__init__()
        self.input_dim = input_dim
        self.n_flows = n_flows
        self.flows = nn.ModuleList([FlowLayer(input_dim, hidden_dim) for _ in range(n_flows)])
        self.base_distribution = torch.distributions.Normal(torch.zeros(input_dim), torch.ones(input_dim))


    def forward(self, x):
        log_det_Jacobian = 0
        z = x
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_Jacobian += log_det

        log_prob_base = self.base_distribution.log_prob(z).sum(dim=1)
        log_prob = log_prob_base + log_det_Jacobian

        return log_prob, log_det_Jacobian

    def reverse(self, z):
        for flow in reversed(self.flows):
            z = flow.reverse(z)
        return z

    def sample(self, num_samples):
        z = torch.randn(num_samples, self.input_dim)
        return self.reverse(z)


class FlowLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
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
        x1 = x[:, :self.input_dim // 2]
        x2 = x[:, self.input_dim // 2:]

        s = self.net(x2)
        y = torch.cat((x1, x2 + s), dim=1)

        log_det_Jacobian = torch.sum(torch.log(torch.abs(1 + s)), dim=1)
        return y, log_det_Jacobian

    def reverse(self, y):
        y1 = y[:, :self.input_dim // 2]
        y2 = y[:, self.input_dim // 2:]

        s = self.net(y2)
        x = torch.cat((y1, y2 - s), dim=1)
        return x

    def create_mask(self, input_dim):
        mask = torch.zeros(input_dim)
        mask[::2] = 1
        return mask

