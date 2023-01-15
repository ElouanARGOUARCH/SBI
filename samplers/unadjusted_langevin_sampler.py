import torch
from torch import nn
from tqdm import tqdm


class ULA(nn.Module):
    def __init__(self, target_log_density, d,proposal_distribution=None, number_chains=1):
        super().__init__()
        self.target_log_density = target_log_density
        self.d = d

        if proposal_distribution is None:
            self.proposal_distribution = torch.distributions.MultivariateNormal(torch.zeros(self.d), torch.eye(self.d))
        else:
            self.proposal_distribution = proposal_distribution
        self.number_chains = number_chains

    def unadjusted_langevin_step(self,x, tau):
        u = torch.sum(self.target_log_density(x))
        grad = torch.autograd.grad(u,x)[0]
        return x + tau * grad + (2 * tau) ** (1 / 2) * torch.randn(x.shape)


    def sample(self, number_steps, tau=0.001):
        x = self.proposal_distribution.sample([self.number_chains])
        x.requires_grad_()
        pbar = tqdm(range(number_steps))
        for t in pbar:
            x = self.unadjusted_langevin_step(x,tau)
        return x