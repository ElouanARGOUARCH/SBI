import torch
from tqdm import tqdm
import pyro
import matplotlib.pyplot as plt

class bayesian_linear_regression:
    def __init__(self, x0, sigma_simulateur, mu_theta, sigma_theta):
        self.d = mu_theta.shape[-1]

        self.sigma_simulateur = sigma_simulateur
        self.mu_theta = mu_theta
        self.sigma_theta = sigma_theta

        self.mu_phi = torch.zeros(self.d + 1)
        self.sigma_phi = torch.eye(self.d + 1)

        self.x0 = x0

    def compute_parameter_likelihood(self, D_theta, D_x):
        assert D_theta.shape[0] >= 2, 'Must have more than 1 dataset sample'
        assert D_theta.shape[0] == D_x.shape[0], 'Mismatch in number samples'
        temp = torch.cat([D_theta, torch.ones(D_theta.shape[0]).unsqueeze(-1)], dim=-1)
        sigma_d_phi = torch.inverse(temp.T @ temp) * self.sigma_simulateur ** 2
        mu_d_phi = D_x @ temp @ torch.inverse(temp.T @ temp)
        return mu_d_phi, sigma_d_phi

    def dataset_likelihood(self, beta, D_theta, D_x):
        assert D_theta >= 1, 'No dataset'
        temp = torch.cat([D_theta, torch.ones(D_theta.shape[0]).unsqueeze(-1)], dim=-1)
        mean = beta @ temp.T
        sigma = (self.sigma_simulateur ** 2) * torch.eye(self.d).unsqueeze(0).repeat(self.theta_samples.shape[0], 1, 1)
        return torch.distributions.MultivariateNormal(mean.unsqueeze(-1), sigma).log_prob(D_x.unsqueeze(-1))

    def compute_parameter_posterior_parameter(self, D_theta, D_x):
        assert D_theta.shape[0] == D_x.shape[0], 'Mismatch in number samples'
        if D_theta.shape[0] >= 1:
            temp = torch.cat([D_theta, torch.ones(D_theta.shape[0]).unsqueeze(-1)], dim=-1)
            sigma_phi_d = torch.inverse(temp.T @ temp / self.sigma_simulateur ** 2 + torch.inverse(self.sigma_phi))
            mu_phi_d = sigma_phi_d @ (
                        D_x @ temp / self.sigma_simulateur ** 2 + torch.inverse(self.sigma_phi) @ self.mu_phi)
        else:
            mu_phi_d, sigma_phi_d = self.mu_phi, self.sigma_phi
        return mu_phi_d, sigma_phi_d

    def log_joint_prob(self, theta, phi):
        log_prior = torch.distributions.MultivariateNormal(self.mu_theta, self.sigma_theta).log_prob(theta)
        augmented_theta = torch.cat([theta, torch.ones(theta.shape[0], 1)], dim=-1)
        temp = torch.bmm(phi.unsqueeze(-2), augmented_theta.unsqueeze(-1)).squeeze(-1)
        temp = temp.repeat(1, self.x0.shape[0])
        cov_matrix = sigma_simulateur * torch.eye(self.x0.shape[0]).unsqueeze(0).repeat(theta.shape[0], 1, 1)
        log_likelihood = torch.distributions.MultivariateNormal(temp, cov_matrix).log_prob(self.x0) if self.x0.shape[
                                                                                                           0] >= 1 else torch.zeros(
            theta.shape[0])
        log_parameter_posterior = self.parameter_posterior_distribution.log_prob(phi)
        return log_parameter_posterior + log_prior + log_likelihood

    def log_likelihood_prob(self, theta, phi):
        augmented_theta = torch.cat([theta, torch.ones(theta.shape[0], 1)], dim=-1)
        temp = torch.bmm(phi.unsqueeze(-2), augmented_theta.unsqueeze(-1)).squeeze(-1)
        temp = temp.repeat(1, self.x0.shape[0])
        cov_matrix = sigma_simulateur * torch.eye(self.x0.shape[0]).unsqueeze(0).repeat(theta.shape[0], 1, 1)
        log_likelihood = torch.distributions.MultivariateNormal(temp, cov_matrix).log_prob(self.x0) if self.x0.shape[
                                                                                                           0] >= 1 else torch.zeros(
            theta.shape[0])
        return log_likelihood

    def marginal_log_likelihood_parameters(self, x, theta):
        mu_phi_d, sigma_phi_d = self.compute_parameter_posterior_parameter()
        gamma = torch.cat([theta, torch.ones(theta.shape[0], 1)], dim=-1).unsqueeze(1).repeat(1, x.shape[0], 1)
        mean = gamma @ mu_phi_d
        cov = gamma @ sigma_phi_d.unsqueeze(0).repeat(theta.shape[0], 1, 1) @ torch.transpose(gamma, -2,
                                                                                              -1) + self.sigma_simulateur * torch.eye(
            x.shape[0]).unsqueeze(0).repeat(theta.shape[0], 1, 1)
        return mean, cov

    def marginal_log_likelihood(self, x, theta):
        mean, cov = self.marginal_log_likelihood_parameters(x, theta)
        return torch.distributions.MultivariateNormal(mean, cov).log_prob(x)

    def sample_marginal_likelihood(self, num_samples, theta):
        mean, cov = self.marginal_log_likelihood_parameters(torch.zeros(num_samples), theta)
        return torch.distributions.MultivariateNormal(mean, cov).sample()

    def compute_posterior_distribution_parameters(self, beta):
        sigma_theta_x0_phi = torch.inverse(
            torch.inverse(self.sigma_theta) + (self.x0.shape[0] * (beta[0] / self.sigma_simulateur) ** 2))
        mu_theta_x0_phi = sigma_theta_x0_phi @ (
                    torch.inverse(self.sigma_theta ** 2) @ self.mu_theta + beta[0] * torch.sum(self.x0 - beta[1]) / (
                        self.sigma_simulateur ** 2))
        return mu_theta_x0_phi, sigma_theta_x0_phi

d = 1
sigma_theta = torch.eye(d)
mu_theta = torch.zeros(d)
prior_distribution = torch.distributions.MultivariateNormal(mu_theta, sigma_theta)
prior_log_prob = lambda samples: prior_distribution.log_prob(samples)

def generate_D_x0(n_D, n_x0):
    D_theta =torch.linspace(-4,4, n_D).unsqueeze(-1)

    sigma_simulateur = .5
    f = lambda y: 1*torch.ones(d)@y.T+1
    simulateur= lambda theta: f(theta) + torch.randn(theta.shape[0])*sigma_simulateur

    D_x = simulateur(D_theta)

    theta_0 = prior_distribution.sample()
    x0 = simulateur(theta_0.unsqueeze(0).repeat(n_x0, 1))
    return D_x, D_theta, x0

def run_gibbs_chain(D_x, D_theta, x0, chain_length=1000):
    blr = bayesian_linear_regression(x0, sigma_simulateur = .5, mu_theta  = mu_theta, sigma_theta = sigma_theta)
    # initialise with phi|D
    mu_phi_D, sigma_phi_D = blr.compute_parameter_posterior_parameter(D_theta, D_x)
    current_phi = torch.distributions.MultivariateNormal(mu_phi_D, sigma_phi_D).sample()
    # sample theta|phi
    mu_theta_x0_phi, sigma_theta_x0_phi = blr.compute_posterior_distribution_parameters(current_phi)
    current_Theta = torch.distributions.MultivariateNormal(mu_theta_x0_phi, sigma_theta_x0_phi).sample([x0.shape[0]])
    list_theta_weird_gibbs = [current_Theta[0]]
    list_phi_weird_gibbs = [current_phi]
    for t in range(chain_length):
        D_theta_plus = torch.cat([D_theta, current_Theta], dim=0)
        D_x_plus = torch.cat([D_x, x0], dim=0)
        mu_phi_D_plus, sigma_phi_D_plus = blr.compute_parameter_posterior_parameter(D_theta_plus, D_x_plus)
        current_phi = torch.distributions.MultivariateNormal(mu_phi_D_plus, sigma_phi_D_plus).sample()
        mu_theta_x0_phi, sigma_theta_x0_phi = blr.compute_posterior_distribution_parameters(current_phi)
        current_Theta = torch.distributions.MultivariateNormal(mu_theta_x0_phi, sigma_theta_x0_phi).sample([x0.shape[0]])
        list_theta_weird_gibbs.append(current_Theta[0])
        list_phi_weird_gibbs.append(current_phi)
    return torch.stack(list_theta_weird_gibbs), torch.stack(list_phi_weird_gibbs)


def several_gibbs_chains(n_D,n_x0,number_tries = 10):
    ess_mean = 0
    for i in tqdm(range(number_tries)):
        D_x, D_theta, x0 = generate_D_x0(n_D,n_x0)
        theta, phi = run_gibbs_chain(D_x, D_theta, x0, chain_length=1000)
        ess_mean+=pyro.ops.stats.effective_sample_size(theta.unsqueeze(1),chain_dim = 1, sample_dim =0).item()
    return ess_mean/number_tries

def plot_ess_evolution_x0():
    list_ess = []
    for n in [2,5,10,50,100,500,1000,5000]:
        list_ess.append(several_gibbs_chains(10,n))
    return list_ess, [2,5,10,50,100,500,1000,5000]

def plot_ess_evolution_D():
    list_ess = []
    for n in [2,5,10,50,100,500,1000,5000]:
        list_ess.append(several_gibbs_chains(n,10))
    return list_ess, [2,5,10,50,100,500,1000,5000]

def plot_ess_evolution_both():
    list_ess = []
    for n in [2,5,10,50,100,500,1000,5000]:
        list_ess.append(several_gibbs_chains(n,n))
    return list_ess, [2,5,10,50,100,500,1000,5000]

def plot_ess_evolution_prop():
    list_ess = []
    for n in [2,6,10,50,100,500,1000,5000]:
        list_ess.append(several_gibbs_chains(n,int(n/2)))
    return list_ess, [2,6,10,50,100,500,1000,5000]

ess_x0, N = plot_ess_evolution_x0()
ess_D, N = plot_ess_evolution_D()
ess_both, N = plot_ess_evolution_both()
ess_prop, N = plot_ess_evolution_prop()
fig = plt.figure(1, figsize=(10,5))
ax1 = fig.add_subplot(111)
ax1.plot(range(len(N)),ess_x0, label ='ESS with N_D = 10, N_x0 = x')
ax1.plot(range(len(N)),ess_D , label ='ESS with N_D = x, N_x0 = 10')
ax1.plot(range(len(N)),ess_both,label ='ESS with N_D = N_x0 = x')
ax1.plot(range(len(N)),ess_prop,label ='ESS with N_D = 2*N_x0 = x')
ax1.xaxis.set_ticklabels(N)
plt.legend()
plt.show()
