import torch
from torch import nn
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal
from tqdm import tqdm

import matplotlib.pyplot as plt
class SMCSampler1(nn.Module):
    def __init__(self, target_log_densities,x0, w0):
        super().__init__()
        self.particles = [x0]
        self.weights = [w0]
        self.N = x0.shape[0]
        self.d = x0.shape[1]
        assert w0.shape[0] == self.N, 'number of weights is different from number of particles'
        self.T = len(target_log_densities)
        self.target_log_densities = target_log_densities

    def log_Q(self, target_log_density, x_prime, x, tau):
        x.requires_grad_()
        u = torch.sum(target_log_density(x))
        grad = torch.autograd.grad(u, x)[0]
        return MultivariateNormal(x+tau*grad,2*tau*torch.eye(x.shape[-1])).log_prob(x_prime)

    def metropolis_adjusted_langevin_step(self,target_log_density, x, tau):
        x.requires_grad_()
        u = torch.sum(target_log_density(x), dim=0)
        grad = torch.autograd.grad(u, x)[0]
        new_x = x + tau*grad + (2*tau)**(1/2)*torch.randn(x.shape)

        acceptance_log_prob = target_log_density(new_x) - target_log_density(x) + self.log_Q(target_log_density,x, new_x, tau) - self.log_Q(target_log_density,new_x, x, tau)
        mask = torch.rand(self.N)<torch.exp(acceptance_log_prob)
        to_return = mask.unsqueeze(-1).repeat(1,self.d).int()*new_x + (1-mask.unsqueeze(-1).repeat(1,self.d).int())*x

        return to_return

    def propagate_particles(self, K,t):
        current_particles = self.particles[t]
        for k in range(1,K+1):
            current_particles = self.metropolis_adjusted_langevin_step(target_log_density=self.target_log_densities[t], x = current_particles,tau = float(1/k))
        return current_particles

    def resample_particles(self, propagated_particles,t):
        pick = Categorical(self.weights[t]).sample([self.N])
        to_append = torch.stack([propagated_particles[pick[i], :] for i in range(self.N)])
        self.particles.append(to_append)

    def reweight_particles(self,t):
        resampled_particles = self.particles[t+1]
        unormalized_log_weights = self.target_log_densities[t+1](resampled_particles) - torch.log(torch.tensor([self.N])) - self.target_log_densities[t](resampled_particles)
        normalized_weights = torch.exp(unormalized_log_weights - torch.logsumexp(unormalized_log_weights, dim = 0) )
        self.weights.append(normalized_weights)

    def plot_density(self, t):
        linspace = torch.linspace(-15,15,200).unsqueeze(-1)
        plt.plot(linspace, torch.exp(self.target_log_densities[t](linspace)).detach().numpy())
        plt.show()

    def plot_all_densities(self):
        linspace = torch.linspace(-15, 15, 1000).unsqueeze(-1)
        for i in range(self.T):
            plt.figure()
            plt.plot(linspace, torch.exp(self.target_log_densities[i](linspace)))
            plt.show()

    def sample(self):
        for t in tqdm(range(self.T-1)):
            propagated_particles = self.propagate_particles(10,t)
            self.resample_particles(propagated_particles,t)
            self.reweight_particles(t)

