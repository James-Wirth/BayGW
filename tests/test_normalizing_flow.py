import unittest
import torch
from src.models.normalizing_flow import NormalizingFlow

class TestNormalizingFlow(unittest.TestCase):
    def test_forward_and_inverse(self):
        model = NormalizingFlow(input_dim=10, hidden_dims=[64, 64], num_layers=3)
        x = torch.randn(5, 10)
        z, _ = model.forward(x)
        x_reconstructed = model.inverse(z)
        self.assertTrue(torch.allclose(x, x_reconstructed, atol=1e-4))