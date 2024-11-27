import torch

def negative_log_likelihood(z, log_jacobians):
    prior = torch.distributions.Normal(0, 1)
    log_prior = prior.log_prob(z).sum(dim=-1)
    return -(log_prior + log_jacobians).mean()
