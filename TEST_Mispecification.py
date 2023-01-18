import torch
import matplotlib.pyplot as plt
torch.manual_seed(0)

#Prior on theta
sigma_theta = 1
mu_theta = 0
prior_distribution = torch.distributions.MultivariateNormal(torch.ones(1)*mu_theta, torch.eye(1)*sigma_theta)

#Define unknown simulator
sigma_simulator = 20
def simulator(theta, sigma_mispecification=0, N_simulator=100, sigma_simulator=sigma_simulator):
    u = torch.randn(theta.shape[0],N_simulator)*sigma_simulator + theta
    z = torch.randn(theta.shape[0],N_simulator)*sigma_mispecification
    return u + z
def summary_statistics(x):
    return torch.stack([torch.mean(x, dim = -1), torch.std(x, dim = -1)], dim = -1)

#Generate D_theta and D_x
n_D = 1000
D_theta = torch.linspace(-3,3,n_D).unsqueeze(-1)
D_x = summary_statistics(simulator(D_theta))

#Generate x_0 from unknown theta0
theta_0 = torch.tensor([0.])
n_x0 = 50
x0 = simulator(theta_0.unsqueeze(0).repeat(n_x0,1))

#True posterior
sigma2_n = (1 + (n_x0*x0.shape[-1])/(sigma_simulator**2))**(-1)
mu_n = sigma2_n*(torch.sum(x0)/(sigma_simulator**2))
true_posterior= torch.distributions.MultivariateNormal(torch.tensor([mu_n]), torch.tensor([[sigma2_n]]))
tt = torch.linspace(-1.5,1.5,200)
plt.plot(tt, torch.exp(true_posterior.log_prob(tt.unsqueeze(-1))))
true_posterior_samples = true_posterior.sample([1000])

#Train conditional density estimation
from conditional_density_estimators import ConditionalDIFDensityEstimator
dif = ConditionalDIFDensityEstimator(D_x,D_theta,1,[32,32,32])
epochs = 500
batch_size = 100
dif.train(epochs, batch_size, lr = 1e-3)

#Sample the posterior p(theta|x0, phi)
from samplers import IMH

list_samples = []
for sigma_mispecification in [0.,1.,2.,3.,4.]:
    x0 = summary_statistics(simulator(theta_0.unsqueeze(0).repeat(n_x0,1), sigma_mispecification=sigma_mispecification))
    posterior_log_prob = lambda theta: torch.sum(dif.log_density(x0.unsqueeze(0).repeat(theta.shape[0], 1, 1), theta.unsqueeze(1).repeat(1, n_x0, 1)),dim=1) + prior_distribution.log_prob(theta)
    sampler = IMH(posterior_log_prob, D_theta.shape[1], prior_distribution, 500)
    samples = sampler.sample(100)
    list_samples.append(samples)
    plt.hist(samples.numpy(), bins = 50, density = True, label = str(sigma_mispecification))
plt.legend()
plt.show()

from utils import compute_expected_coverage, plot_expected_coverage
for i,samples in enumerate(list_samples):
    plot_expected_coverage(true_posterior_samples, samples, label = str(i))
plt.legend()
plt.show()

current_theta = samples[:1]
gibbs_iterations = 10000
list_theta = [current_theta]
for t in range(gibbs_iterations):
    D_x_plus = torch.cat([D_x, x0], dim=0)
    D_theta_plus = torch.cat([D_theta, current_theta.repeat(n_x0, 1)], dim=0)
    dif = ConditionalDIFDensityEstimator(D_x_plus,D_theta_plus,1,[32,32,32])
    dif.train(epochs, batch_size, 1e-3)

    posterior_log_prob = lambda theta: torch.sum(dif.log_density(x0.unsqueeze(0).repeat(theta.shape[0], 1, 1), theta.unsqueeze(1).repeat(1, n_x0, 1)),dim=1) + prior_distribution.log_prob(theta)
    sampler = IMH(posterior_log_prob, 1, prior_distribution, 1)
    current_theta = sampler.sample(100)
    list_theta.append(current_theta)
torch.save(torch.cat(list_theta, dim =0), 'theta.sav')
