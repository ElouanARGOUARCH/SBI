import torch
from tqdm import tqdm

class WLBSampler:
    def __init__(self, log_likelihood_function,d,observations, prior_distribution = None):
        self.log_likelihood_function = log_likelihood_function
        self.d = d
        self.observations = observations
        self.p = observations.shape[-1]
        self.Dirichlet = torch.distributions.Dirichlet(torch.ones(self.observations.shape[0]))
        if prior_distribution is None:
            self.prior_distribution = torch.distributions.MultivariateNormal(torch.zeros(self.d), torch.eye(self.d))
        else:
            self.prior_distribution = prior_distribution
    def sample(self,number_samples, epochs = 500, lr = 5e-3):
        parameter = self.prior_distribution.sample([number_samples])
        parameter.requires_grad_()
        w = torch.distributions.Dirichlet(torch.ones(self.observations.shape[0])).sample([number_samples]).T
        optimizer = torch.optim.Adam([parameter], lr = lr)
        pbar = tqdm(range(epochs))
        for t in pbar:
            optimizer.zero_grad()
            loss = -torch.sum(w*self.log_likelihood_function(self.observations.unsqueeze(1).repeat(1, number_samples,1),parameter.unsqueeze(0).repeat(self.observations.shape[0], 1,1)))
            loss.backward()
            optimizer.step()
            pbar.set_postfix_str(str(loss.item()))
        return parameter
