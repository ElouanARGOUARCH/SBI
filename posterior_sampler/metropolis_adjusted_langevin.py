import torch
from torch import nn
from torch.distributions import MultivariateNormal
from tqdm import tqdm

class MALA(nn.Module):
    def __init__(self, target_log_density, tau, d, x0 = None):
        super().__init__()
        self.target_log_density = target_log_density
        self.tau = tau
        self.d = d
        self.accepted = []
        self.x0 = x0

    def log_Q(self, x_prime, x):
        x.requires_grad_()
        u = torch.sum(self.target_log_density(x))
        grad = torch.autograd.grad(u, x)[0]
        return MultivariateNormal(x+self.tau*grad,2*self.tau*torch.eye(x.shape[-1])).log_prob(x_prime)

    def metropolis_adjusted_langevin_step(self, x):
        x.requires_grad_()
        u = self.target_log_density(x)
        grad = torch.autograd.grad(u, x)[0]
        x_prime = x + self.tau*grad + (2*self.tau)**(1/2)*torch.randn(x.shape)

        acceptance_log_prob = self.target_log_density(x_prime) - self.target_log_density(x) + self.log_Q(x, x_prime) - self.log_Q(x_prime, x)
        if torch.rand(1)<torch.exp(acceptance_log_prob):
            self.accepted.append(1.)
            return x_prime
        else:
            self.accepted.append(0.)
            return x

    def sample(self, num_samples, burn):
        if self.x0 is None:
            self.x0 = torch.randn(1, self.d)
        list_x = [self.x0]
        pbar = tqdm(range(num_samples + burn))
        for t in pbar:
            list_x.append(self.metropolis_adjusted_langevin_step(list_x[-1]))
            pbar.set_postfix_str('acceptance = ' + str(torch.tensor(self.accepted).mean()))
        return torch.cat(list_x[burn:], dim =0)



