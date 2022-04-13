import torch
from tqdm import tqdm

class WLBSampler:
    def __init__(self, log_likelihood_function,x0,prior_sampler, epochs = 500):
        self.epochs = epochs
        self.prior_sampler = prior_sampler
        self.log_likelihood_function = log_likelihood_function
        self.x0 = x0
        self.Dirichlet = torch.distributions.Dirichlet(torch.ones(self.x0.shape[0]))

    def perturbed_log_likelihood(self,x0, theta, dirichlet_samples):
        theta_augmented = theta.unsqueeze(1).repeat(1, x0.shape[0], 1)
        x0_augmented = x0.unsqueeze(0)
        return torch.sum(dirichlet_samples*self.log_likelihood_function(x0_augmented, theta_augmented).squeeze(0))

    def perturbed_log_likelihood_new(self,x0, theta, dirichlet_samples):
        theta_augmented = theta.unsqueeze(1).repeat(1, x0.shape[0], 1)
        x0_augmented = x0.unsqueeze(0).repeat(theta.shape[0],1,1)
        temp = torch.sum(torch.square(torch.sum(dirichlet_samples*self.log_likelihood_function(x0_augmented, theta_augmented), dim = -1)))
        print(temp.shape)
        return torch.sum(torch.square(torch.sum(dirichlet_samples*self.log_likelihood_function(x0_augmented, theta_augmented), dim = -1)))

    def maximize_perturbed_log_likelihood(self, dirichlet_samples):
        theta = self.prior_sampler(1)
        theta.requires_grad = True
        optimizer = torch.optim.Adam([theta], lr = 5e-2)
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            perturbed_log_likelihood = - self.perturbed_log_likelihood(self.x0,theta, dirichlet_samples)
            perturbed_log_likelihood.backward()
            optimizer.step()
        return theta

    def maximize_perturbed_log_likelihood_new(self, dirichlet_samples):
        theta = self.prior_sampler(dirichlet_samples.shape[0])
        theta.requires_grad = True
        optimizer = torch.optim.Adam([theta], lr = 5e-2)
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            perturbed_log_likelihood = - self.perturbed_log_likelihood_new(self.x0,theta, dirichlet_samples)
            perturbed_log_likelihood.backward()
            optimizer.step()
        return theta

    def sample(self, num_samples):
        samples = []
        pbar = tqdm(range(num_samples))
        for t in pbar:
            sample = self.maximize_perturbed_log_likelihood(self.Dirichlet.sample())
            samples.append(sample)
            pbar.set_postfix_str('sample ' + str(t))
        return torch.cat(samples, dim =0)

    def sample_new(self, num_samples):
        return self.maximize_perturbed_log_likelihood_new(self.Dirichlet.sample([num_samples]))
