import torch
from targets.conditional_density_estimation_target import SymmetricalDeformedCircles
import matplotlib.pyplot as plt
from utils import compute_expected_coverage, plot_expected_coverage
torch.manual_seed(0)

#Prior on theta
sigma_theta = 1
mu_theta = 2
prior= torch.distributions.MultivariateNormal(torch.ones(1)*mu_theta, torch.eye(1)*sigma_theta)

#Generate D_theta and D_x
n_D = 1000
cond_target = SymmetricalDeformedCircles()
D_theta = torch.linspace(0,4,n_D).unsqueeze(-1)
D_x = cond_target.sample(D_theta)

#Generate x_0 from unknown theta0
theta0 =torch.tensor([1.])
n_x0 = 50
x0 = cond_target.sample(theta0.unsqueeze(0).repeat(n_x0,1))

#True posterior
from samplers import IMH
true_posterior = lambda theta: torch.sum(cond_target.log_prob(x0.unsqueeze(1).repeat(1, theta.shape[0],1), theta.unsqueeze(0).repeat(x0.shape[0],1,1)), dim = 0) + prior.log_prob(theta)
true_posterior_samples = IMH(true_posterior, 1, prior, 1000).sample(200)

#Train conditional density estimation
from conditional_density_estimators import ConditionalDIFDensityEstimator
dif = ConditionalDIFDensityEstimator(D_x,D_theta,10,[32,32,32])
epochs = 1000
batch_size = 100
dif.train(epochs,batch_size, lr = 5e-4, device = 'cudad')

#Sample the posterior p(theta|x0, phi)
posterior_log_prob = lambda theta: torch.sum(dif.log_density(x0.unsqueeze(0).repeat(theta.shape[0], 1, 1), theta.unsqueeze(1).repeat(1, n_x0, 1)),dim=1) + prior.log_prob(theta)
sampler = IMH(posterior_log_prob, D_theta.shape[1], prior,1)
current_theta = sampler.sample(200)

gibbs_iterations = 1000
list_theta = []
for _ in range(gibbs_iterations):
    D_x_plus = torch.cat([D_x, x0], dim=0)
    D_theta_plus = torch.cat([D_theta, current_theta.repeat(n_x0, 1)], dim=0)
    dif = ConditionalDIFDensityEstimator(D_x,D_theta,10,[32,32,32])
    #dif.change_dataset(D_x_plus,D_theta_plus)
    dif.train(epochs,batch_size, lr = 5e-4, device = 'cuda')
    
    posterior_log_prob = lambda theta: torch.sum(dif.log_density(x0.unsqueeze(0).repeat(theta.shape[0], 1, 1), theta.unsqueeze(1).repeat(1, n_x0, 1)),dim=1) + prior.log_prob(theta)
    sampler = IMH(posterior_log_prob, 1, prior, 1)
    current_theta = sampler.sample(1)
    list_theta.append(current_theta)
torch.save(torch.cat(list_theta, dim=0), 'theta_samples.sav')