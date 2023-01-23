import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from torch.distributions import Normal
from .misc import Uniform
import math


class ConditionalDensityEstimationTarget(nn.Module):
    def __init__(self):
        super().__init__()

    def simulate(self, thetas):
        raise NotImplementedError

    def sample_prior(self):
        raise NotImplementedError

    def prior_log_prob(self):
        raise NotImplementedError

    def make_dataset(self, num_samples):
        theta = self.sample_prior(num_samples)
        x = self.simulate(theta)
        return theta, x

    def target_visual(self):
        raise NotImplementedError

class Wave(ConditionalDensityEstimationTarget):
    def __init__(self):
        super().__init__()
        self.p = 1
        self.d = 1
        self.prior = Uniform(torch.tensor([-8.]),torch.tensor([8.]))

    def sample_prior(self, num_samples):
        return self.prior.sample([num_samples])

    def prior_log_prob(self, theta):
        return self.prior.log_prob(theta)

    def mu(self,theta):
        return torch.sin(math.pi * theta)/(1+theta**2)+ torch.sin(math.pi * theta / 3.0)

    def sigma2(self,theta):
        return torch.square(.5 * (1.2 - 1 / (1 + 0.1 * theta ** 2))) + 0.05

    def simulate(self, thetas):
        return torch.cat([Normal(self.mu(theta), self.sigma2(theta)).sample().unsqueeze(-1) for theta in thetas], dim=0)

    def target_visual(self):
        theta_samples, x_samples = self.make_dataset(5000)
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot()
        ax.scatter(theta_samples,x_samples, color='red', alpha=.4,
                   label='(x|theta) samples')
        ax.set_xlabel('theta')
        ax.set_ylabel('x')
        ax.legend()

class DoubleWave():
    def __init__(self):
        super().__init__()
        self.p = 1
        self.d = 1
        self.prior = Uniform(torch.tensor([-8.]),torch.tensor([8.]))

    def sample_prior(self, num_samples):
        return self.prior.sample([num_samples])

    def prior_log_prob(self, theta):
        return self.prior.log_prob(theta)

    def mu(self,theta):
        return torch.sin(math.pi * theta)/(1+theta**2)+ torch.sin(math.pi * theta / 3.0)

    def sigma2(self,theta):
        return torch.square(.5 * (1.2 - 1 / (1 + 0.1 * theta ** 2))) + 0.05

    def simulate(self, thetas):
        return torch.cat([torch.distributions.MixtureSameFamily(torch.distributions.Categorical(torch.tensor([.5,.5])),Normal(torch.cat([self.mu(theta),-self.mu(theta)]), torch.cat([self.sigma2(theta),self.sigma2(theta)]))).sample([1]).unsqueeze(-1) for theta in thetas], dim=0)

class GaussianField(ConditionalDensityEstimationTarget):
    def __init__(self):
        super().__init__()
        self.p = 1
        self.d = 1
        self.prior = torch.distributions.MultivariateNormal(torch.tensor([0.]),torch.tensor([[3.]]))

    def sample_prior(self, num_samples):
        return self.prior.sample([num_samples])

    def prior_log_prob(self, theta):
        return self.prior.log_prob(theta)

    def mu(self,theta):
       PI = torch.acos(torch.zeros(1)).item() * 2
       thetac = theta + PI
       return (torch.sin(thetac) if 0 < thetac < 2. * PI else torch.tanh(
          thetac * .5) * 2 if thetac < 0 else torch.tanh((thetac - 2. * PI) * .5) * 2)

    def sigma2(self,theta):
        PI = torch.acos(torch.zeros(1)).item() * 2
        return torch.tensor(0.1) + torch.exp(.5 * (theta - PI))

    def simulate(self, thetas):
        return torch.cat([Normal(self.mu(theta), self.sigma2(theta)).sample().unsqueeze(-1) for theta in thetas], dim=0)

    def target_visual(self):
        theta_samples, x_samples = self.make_dataset(5000)
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot()
        ax.scatter(theta_samples, x_samples, color='red', alpha=.4,
                   label='(x|theta) samples')
        ax.set_xlabel('theta')
        ax.set_ylabel('x')
        ax.legend()

class DeformedCircles(ConditionalDensityEstimationTarget):
    def __init__(self):
        super().__init__()

    def sample(self, theta, means=torch.tensor([1., 2.]), weights=torch.tensor([.5, .5]), noise=0.125):
        angle = torch.rand(theta.shape[0]) * 2 * math.pi
        cat = torch.distributions.Categorical(weights).sample([theta.shape[0]])
        x, y = means[cat] * torch.cos(angle) + torch.randn_like(angle) * noise, means[cat] * torch.sin(
            angle) + torch.randn_like(angle) * noise
        return torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)*theta

    def log_prob(self, samples,theta, means=torch.tensor([1., 2.]), weights=torch.tensor([.5, .5]), noise=0.125):
        r = torch.norm(samples/theta, dim=-1).unsqueeze(-1)
        cat = torch.distributions.Categorical(weights)
        mvn = torch.distributions.MultivariateNormal(means.unsqueeze(-1),
                                                     torch.eye(1).unsqueeze(0).repeat(2, 1, 1) * noise)
        mixt = torch.distributions.MixtureSameFamily(cat, mvn)
        return mixt.log_prob(r)


class SymmetricalDeformedCircles(ConditionalDensityEstimationTarget):
    def __init__(self):
        super().__init__()

    def sample(self, theta, means=torch.tensor([1., 2.]), weights=torch.tensor([.5, .5]), noise=0.125):
        angle = torch.rand(theta.shape[0]) * 2 * math.pi
        cat = torch.distributions.Categorical(weights).sample([theta.shape[0]])
        x, y = means[cat] * torch.cos(angle) + torch.randn_like(angle) * noise, means[cat] * torch.sin(
            angle) + torch.randn_like(angle) * noise
        return torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)*theta

    def log_prob(self, samples,theta, means=torch.tensor([1., 2.]), weights=torch.tensor([.5, .5]), noise=0.125):
        r = torch.norm(samples/theta, dim=-1).unsqueeze(-1)
        cat = torch.distributions.Categorical(weights)
        mvn = torch.distributions.MultivariateNormal(means.unsqueeze(-1),
                                                     torch.eye(1).unsqueeze(0).repeat(2, 1, 1) * noise)
        mixt = torch.distributions.MixtureSameFamily(cat, mvn)
        return mixt.log_prob(r)

class MoonsRotation(ConditionalDensityEstimationTarget):
    def __init__(self):
        super().__init__()
        self.prior = Uniform(torch.tensor([0.]), torch.tensor([3.14159265]))

    def prior_log_prob(self, theta):
        return self.prior.log_prob(theta)

    def sample_prior(self, num_samples):
        return self.prior.sample([num_samples])

    def simulate(self, thetas):
        num_samples = min(thetas.shape[0], 100)
        X, y = datasets.make_moons(num_samples, noise=0.05)
        X = StandardScaler().fit_transform(X)
        return torch.cat([torch.tensor(X[torch.randperm(X.shape[0])][0]).unsqueeze(0).float() @ torch.tensor(
            [[torch.cos(theta), torch.sin(
                theta)], [torch.cos(theta), -torch.sin(
                theta)]]) for theta in thetas], dim=0)

    def target_visual(self):
        fig = plt.figure(figsize=(15, 15))
        for i in range(4):
            ax = fig.add_subplot(2, 2, i + 1)
            theta = torch.tensor(3.141592 / 8 * (1 + 2 * i))
            T = theta.unsqueeze(-1)
            rotation_matrix = torch.zeros(1, 2, 2)
            rotation_matrix[0, 0, 0], rotation_matrix[0, 0, 1], rotation_matrix[0, 1, 0], rotation_matrix[
                0, 1, 1] = torch.cos(T), torch.sin(T), -torch.sin(T), torch.cos(T)
            rotation_matrix = rotation_matrix.repeat(5000, 1, 1)
            X, y = datasets.make_moons(5000, noise=0.05)
            X = (torch.tensor(X).float().unsqueeze(-2) @ rotation_matrix).squeeze(-2)
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
            ax.scatter(X[:, 0], X[:, 1], color='red', alpha=.3,
                       label='theta = ' + str(np.round(theta.item(), 3)))
            ax.scatter([0], [0], color='black')
            ax.axline([0, 0], [torch.cos(theta), torch.sin(theta)], color='black', linestyle='--',
                      label='Axis Rotation with angle theta')
            ax.axline([0, 0], [1., 0.], color='black')
            ax.legend()
