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
        self.prior = torch.distributions.MultivariateNormal(torch.tensor([0.]),torch.tensor([[3.]]))

    def sample_prior(self, num_samples):
        return self.prior.sample([num_samples])

    def prior_log_prob(self, theta):
        return self.prior.log_prob(theta)

    def simulate(self, thetas):
        num_samples = min(thetas.shape[0], 100)
        X, y = datasets.make_circles(num_samples, factor=.5, noise=0.025)
        X = StandardScaler().fit_transform(X)
        return torch.cat([torch.tensor(X[torch.randperm(X.shape[0])][0]).unsqueeze(0).float() * torch.abs(theta) for theta in
                   thetas], dim=0)

    def target_visual(self):
        fig = plt.figure(figsize=(10, 10))
        for i in range(2):
            for j in range(2):
                theta = torch.tensor([[.75 + i * 1.25, .75 + j * 1.25]])
                T = theta.repeat(5000, 1)
                X, y = datasets.make_circles(5000, factor=0.5, noise=0.025)
                X = torch.tensor(StandardScaler().fit_transform(X)).float() * T
                ax = fig.add_subplot(2, 2, i + 2 * j + 1)
                ax.set_xlim(-5, 5)
                ax.set_ylim(-5, 5)
                ax.scatter(X[:, 0], X[:, 1], color='red', alpha=.3,
                           label='theta = [' + str(np.round(theta[0, 0].item(), 3)) + ',' + str(
                               np.round(theta[0, 1].item(), 3)) + ']')
                ax.scatter([0], [0], color='black')
                ax.arrow(0., 0., theta[0, 0], 0., color='black', head_width=0.2, head_length=0.2)
                ax.text(theta[0, 0] - .3, -.4, "theta_x = " + str(np.round(theta[0, 0].item(), 3)))
                ax.arrow(0., 0., 0., theta[0, 1], color='black', head_width=0.2, head_length=0.2)
                ax.text(-.3, theta[0, 1] + .4, "theta_y = " + str(np.round(theta[0, 1].item(), 3)))
                ax.legend()

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
