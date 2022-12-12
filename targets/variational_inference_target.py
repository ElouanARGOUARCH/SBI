import torch
from torch import nn
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily, Uniform
import math

class VariationalInferenceTarget(nn.Module):
    def __init__(self):
        super().__init__()

    def log_prob(self):
        raise NotImplementedError

class Orbits(VariationalInferenceTarget):
    def __init__(self):
        super().__init__()
        self.p = 2
        number_planets = 7
        covs_target = 0.04*torch.eye(self.p).unsqueeze(0).repeat(number_planets,1,1)
        means_target = 2.5*torch.view_as_real(torch.pow(torch.exp(torch.tensor([2j * math.pi / number_planets])), torch.arange(0, number_planets)))
        weights_target = torch.ones(number_planets)
        weights_target = weights_target

        mvn_target = MultivariateNormal(means_target, covs_target)
        cat = Categorical(torch.exp(weights_target) / torch.sum(torch.exp(weights_target)))
        self.mix_target = MixtureSameFamily(cat, mvn_target)

    def log_prob(self, samples):
        return self.mix_target.log_prob(samples.cpu()).to(samples.device)

class Banana(VariationalInferenceTarget):
    def __init__(self):
        super().__init__()
        var = 2
        self.p = 50
        self.even = torch.arange(0, self.p, 2)
        self.odd = torch.arange(1, self.p, 2)
        self.mvn = torch.distributions.MultivariateNormal(torch.zeros(self.p), var * torch.eye(self.p))

    def inv_transform(self, z):
        x = z.clone()
        x[..., self.odd] -= x[..., self.even] ** 2

        return x

    def log_prob(self, samples):
        return self.mvn.log_prob(self.inv_transform(samples))

class Dimension1(VariationalInferenceTarget):
    def __init__(self):
        super().__init__()
        self.p = 1
        num_component = 6
        means = torch.tensor([[-0.25], [1.875], [4.125], [6.25], [-5.5], [-8.5]])
        covs = torch.tensor([[[1.]], [[.5]], [[.5]], [[2.]], [[1]], [[1]]])
        comp = torch.ones(num_component)
        mvn_target = MultivariateNormal(means, covs)
        cat = Categorical(comp / torch.sum(comp))
        self.mix_target = MixtureSameFamily(cat, mvn_target)

    def log_prob(self, samples):
        return self.mix_target.log_prob(samples.cpu()).to(samples.device)

class BlobDimension64(VariationalInferenceTarget):
    def __init__(self):
        super().__init__()
        self.p = 64
        mixtures_target = 4 + 2*torch.randint(0,6,[1])
        L = torch.randn(mixtures_target, self.p, self.p)
        covs_target = L @ L.transpose(-1, -2) + torch.eye(self.p)
        covs_target = 2*covs_target
        means_target = self.p*torch.randn(mixtures_target, self.p)/2
        means_target = means_target
        weights_target = torch.ones(mixtures_target)
        weights_target = weights_target
        mvn_target = MultivariateNormal(means_target, covs_target)
        cat = Categorical(torch.exp(weights_target) / torch.sum(torch.exp(weights_target)))
        self.mvn = MixtureSameFamily(cat, mvn_target)

    def log_prob(self, samples):
        return self.mvn.log_prob(samples.cpu()).to(samples.device)

class Funnel(VariationalInferenceTarget):

    def __init__(self):
        super().__init__()
        self.a = torch.tensor(1.)
        self.b = torch.tensor(0.5)
        self.dim = 20

        self.distrib_x1 = MultivariateNormal(torch.zeros(1), torch.tensor(self.a))

    def log_prob(self, x):
        log_probx1 = self.distrib_x1.log_prob(x[..., 0].unsqueeze(1))
        logprob_rem = (- x[..., 1:] ** 2 * (-2 * self.b * x[..., 0].unsqueeze(-1)).exp() - \
                       2 * self.b * x[:, 0].unsqueeze(-1) - torch.tensor(2 * math.pi).log()) / 2
        logprob_rem = logprob_rem.sum(-1)
        return (log_probx1 + logprob_rem.unsqueeze(-1)).flatten()

