import torch
from targets.conditional_density_estimation_target import SymmetricalDeformedCircles
import matplotlib.pyplot as plt
from utils import plot_expected_coverage, plot_1d_unormalized_function, plot_1d_unormalized_values
torch.manual_seed(0)

#Prior on theta
sigma_theta = 1
mu_theta = 2
prior= torch.distributions.MultivariateNormal(torch.ones(1)*mu_theta, torch.eye(1)*sigma_theta)


#Generate D_theta and D_x
n_D = 5000
cond_target = SymmetricalDeformedCircles()
D_theta = torch.linspace(0,4,n_D).unsqueeze(-1)
D_x = cond_target.sample(D_theta)

#Generate x_0 from unknown theta0
theta0 =torch.tensor([1.5])
n_x0 = 10
x0 = cond_target.sample(theta0.unsqueeze(0).repeat(n_x0,1))

#True posterior
from samplers import IMH
true_posterior = lambda theta: torch.sum(cond_target.log_prob(x0.unsqueeze(1).repeat(1, theta.shape[0],1), theta.unsqueeze(0).repeat(x0.shape[0],1,1)), dim = 0) + prior.log_prob(theta)
true_posterior_samples = IMH(true_posterior, 1, prior, 1000).sample(500)
fig = plt.figure()
plt.hist(true_posterior_samples.numpy(), bins = 50, density = True)
plot_1d_unormalized_function(lambda theta: torch.exp(true_posterior(theta.unsqueeze(-1))))
plt.xlim([0, 5])
plt.show()

#Train conditional density estimation
from conditional_density_estimators import ConditionalDIFDensityEstimator
dif = ConditionalDIFDensityEstimator(D_x,D_theta,10,[32,32,32])
epochs = 100
batch_size = 500
dif.train(epochs,batch_size, lr = 1e-3, device = 'cpu')

#Sample the posterior p(theta|x0, phi)
list_samples = []
list_true_samples = []
list_log_prob = []
tt = torch.linspace(-3,5,200)
for i,sigma_mispecification in enumerate([0.,.5,1.,1.5,2.]):
    x0_ = x0 + torch.randn_like(x0)*sigma_mispecification
    true_posterior_log_prob = lambda theta: torch.sum(cond_target.log_prob(x0_.unsqueeze(1).repeat(1, theta.shape[0],1), theta.unsqueeze(0).repeat(x0_.shape[0],1,1)), dim = 0) + prior.log_prob(theta)
    sampler = IMH(true_posterior_log_prob, D_theta.shape[1], prior,200)
    true_samples = sampler.sample(200)
    list_true_samples.append(true_samples)
    posterior_log_prob = lambda theta: torch.sum(dif.log_density(x0_.unsqueeze(0).repeat(theta.shape[0], 1, 1), theta.unsqueeze(1).repeat(1, n_x0, 1)),dim=1) + prior.log_prob(theta)
    sampler = IMH(posterior_log_prob, D_theta.shape[1], prior,200)
    samples = sampler.sample(200)
    list_samples.append(samples)
    list_log_prob.append(true_posterior_log_prob(tt.unsqueeze(-1)))

for i, samples in enumerate(list_samples):
    plt.figure()
    plt.hist(list_true_samples[i].numpy(), bins = 50, density = True, label = 'Model :' + str(sigma_mispecification), color = 'C'+str(i))
    plt.hist(samples.numpy(), bins = 50, density = True, label = 'True :' + str(sigma_mispecification), color='grey')
    plot_1d_unormalized_values(torch.exp(list_log_prob[i]), tt)
    plt.show()

for i, samples in enumerate(list_samples):
    plot_expected_coverage(list_true_samples[i], samples, label = str(i))
plt.legend()
plt.show()

