import torch
from torch import nn
from torch.distributions import MultivariateNormal
from tqdm import tqdm


class MALA(nn.Module):
    def __init__(self, target_log_density, d,proposal_distribution=None, number_chains=1):
        super().__init__()
        self.target_log_density = target_log_density
        self.d = d
        if proposal_distribution is None:
            self.proposal_distribution = torch.distributions.MultivariateNormal(torch.zeros(self.d), torch.eye(self.d))
        else:
            self.proposal_distribution = proposal_distribution
        self.number_chains = number_chains

    def log_Q(self, x,x_prime, tau):
        u = torch.sum(self.target_log_density(x))
        grad = torch.autograd.grad(u, x)[0]
        return MultivariateNormal(x + tau * grad, 2 * tau * torch.eye(x.shape[-1])).log_prob(x_prime)

    def metropolis_adjusted_langevin_step(self,x, tau):
        u = torch.sum(self.target_log_density(x))
        grad = torch.autograd.grad(u, x)[0]
        x_prime = x + tau * grad + (2 * tau) ** (1 / 2) * torch.randn(x.shape)

        acceptance_log_prob = self.target_log_density(x_prime) - self.target_log_density(x) - self.log_Q(x,
                                                                                                              x_prime,
                                                                                                              tau) + self.log_Q(
            x_prime, x, tau)
        mask = ((torch.rand(x.shape[0]) < torch.exp(acceptance_log_prob)) * 1.).unsqueeze(-1)
        x = (mask) * x_prime + (1 - (mask)) * x
        return x,mask

    def sample(self, number_steps, tau=0.1):
        x = self.proposal_distribution.sample([self.number_chains])
        x.requires_grad_()
        pbar = tqdm(range(number_steps))
        for t in pbar:
            x,mask = self.metropolis_adjusted_langevin_step(x,tau)
            pbar.set_postfix_str('acceptance = ' + str(torch.mean(mask * 1.)))
        return x