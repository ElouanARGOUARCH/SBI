import torch
from torch import nn
from tqdm import tqdm
class LARS_Density_Estimation(nn.Module):
    def __init__(self, target_samples, hidden_dims):
        super().__init__()
        self.target_samples = target_samples
        self.p = target_samples.shape[-1]

        network_dimensions = [self.p] + hidden_dims + [1]
        network = []
        for h0, h1 in zip(network_dimensions, network_dimensions[1:]):
            network.extend([nn.Linear(h0, h1), nn.SiLU(), ])
        network.pop()
        network.extend([nn.LogSigmoid(), ])

        self.log_alpha = nn.Sequential(*network)
        self.proposal = torch.distributions.MultivariateNormal(torch.mean(target_samples, dim=0),
                                                               torch.cov(target_samples.T))
        self.loss_values = []

    def sample(self, num_samples):
        proposed_samples = self.proposal.sample([num_samples])
        acceptance_probability = torch.exp(self.log_alpha(proposed_samples)).squeeze(-1)
        mask = torch.rand(acceptance_probability.shape) < acceptance_probability
        return proposed_samples[mask]

    def estimate_log_constant(self, num_samples):
        proposed_samples = self.proposal.sample([num_samples])
        self.log_constant = torch.logsumexp(self.log_alpha(proposed_samples).squeeze(-1), dim=0) - torch.log(
            torch.tensor([proposed_samples.shape[0]]))

    def log_prob(self, x):
        return self.proposal.log_prob(x) + self.log_alpha(x).squeeze(-1) - self.log_constant

    def loss(self, target_samples):
        loss = -torch.mean(self.log_prob(target_samples))
        return loss

    def train(self, epochs, batch_size=None):
        self.para_list = list(self.parameters())
        self.optimizer = torch.optim.Adam(self.para_list, lr=5e-3)
        if batch_size is None:
            batch_size = self.target_samples.shape[0]
        dataset = torch.utils.data.TensorDataset(self.target_samples)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        pbar = tqdm(range(epochs))
        for t in pbar:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for i, batch in enumerate(dataloader):
                x = batch[0].to(device)
                self.optimizer.zero_grad()
                self.estimate_log_constant(5000)
                batch_loss = self.loss(x)
                batch_loss.backward()
                self.optimizer.step()
            with torch.no_grad():
                iteration_loss = torch.tensor(
                    [self.loss(batch[0].to(device)) for i, batch in enumerate(dataloader)]).mean().item()
                self.loss_values.append(batch_loss)
            pbar.set_postfix_str('loss = ' + str(round(iteration_loss, 6)))
        self.to(torch.device('cpu'))


class LARS_Variational_Inference(nn.Module):
    def __init__(self, target_log_prob, p, hidden_dims):
        super().__init__()
        self.target_log_prob = target_log_prob
        self.p = p

        network_dimensions = [self.p] + hidden_dims + [1]
        network = []
        for h0, h1 in zip(network_dimensions, network_dimensions[1:]):
            network.extend([nn.Linear(h0, h1), nn.SiLU(), ])
        network.pop()
        network.extend([nn.LogSigmoid(), ])

        self.log_alpha = nn.Sequential(*network)
        self.proposal = torch.distributions.MultivariateNormal(torch.zeros(self.p), 20 * torch.eye(self.p))
        self.loss_values = []

    def sample(self, num_samples):
        proposed_samples = self.proposal.sample([num_samples])
        acceptance_probability = torch.exp(self.log_alpha(proposed_samples)).squeeze(-1)
        mask = torch.rand(acceptance_probability.shape) < acceptance_probability
        return proposed_samples[mask]

    def estimate_log_constant(self, num_samples):
        proposed_samples = self.proposal.sample([num_samples])
        self.log_constant = torch.logsumexp(self.log_alpha(proposed_samples).squeeze(-1), dim=0) - torch.log(
            torch.tensor([proposed_samples.shape[0]]))

    def log_prob(self, x):
        return self.proposal.log_prob(x) + self.log_alpha(x).squeeze(-1) - self.log_constant

    def DKL(self, num_samples):
        model_samples = self.sample(num_samples)
        return torch.mean(self.log_prob(model_samples) - self.target_log_prob(model_samples))

    def loss(self, num_samples):
        model_samples = self.sample(num_samples).detach()
        loss = torch.mean((self.log_alpha(model_samples).squeeze(-1) - self.log_constant) * (
                    self.log_prob(model_samples).detach() - self.target_log_prob(model_samples).detach() + 1))
        return loss

    def second_loss(self, num_samples):
        proposed_samples = self.proposal.sample([num_samples])
        log_constant = torch.logsumexp(self.log_alpha(proposed_samples).squeeze(-1), dim=0) - torch.log(
            torch.tensor([proposed_samples.shape[0]]))
        self.log_constant= log_constant
        acceptance_probability = torch.exp(self.log_alpha(proposed_samples)).squeeze(-1)
        mask = torch.rand(acceptance_probability.shape) < acceptance_probability
        accepted_samples = proposed_samples[mask]
        loss = torch.mean((self.log_alpha(accepted_samples).squeeze(-1) - log_constant) * (
                    self.log_prob(accepted_samples).detach() - self.target_log_prob(accepted_samples).detach() + 1))
        return loss

    def second_train(self, epochs):
        self.loss_values = []
        self.para_list = list(self.parameters())

        self.optimizer = torch.optim.Adam(self.para_list, lr=5e-4)
        pbar = tqdm(range(epochs))
        for t in pbar:
            self.optimizer.zero_grad()
            batch_loss = self.second_loss(50000)
            batch_loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                iteration_loss = self.DKL(5000).item()
            self.loss_values.append(iteration_loss)
            pbar.set_postfix_str('loss = ' + str(round(iteration_loss, 6)))
        self.to(torch.device('cpu'))

    def train(self, epochs):
            self.loss_values = []
            self.para_list = list(self.parameters())

            self.optimizer = torch.optim.Adam(self.para_list, lr=5e-4)
            pbar = tqdm(range(epochs))
            for t in pbar:
                self.optimizer.zero_grad()
                self.estimate_log_constant(10000)
                batch_loss = self.loss(50000)
                batch_loss.backward()
                self.optimizer.step()
                with torch.no_grad():
                    iteration_loss = self.DKL(5000).item()
                self.loss_values.append(iteration_loss)
                pbar.set_postfix_str('loss = ' + str(round(iteration_loss, 6)))
            self.to(torch.device('cpu'))