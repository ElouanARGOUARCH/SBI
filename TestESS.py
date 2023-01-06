import torch
from tqdm import tqdm

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

def several_gibbs_chains(n_D,n_x0,number_tries = 10,):
    for i in tqdm(range(number_tries)):
        D_x, D_theta, x0 = generate_D_x0(n_D,n_x0)

