import torch
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
torch.manual_seed(0)

#Prior on theta
sigma_theta = 1
mu_theta = 3
prior_distribution = torch.distributions.MultivariateNormal(torch.ones(1)*mu_theta, torch.eye(1)*sigma_theta)


def simulator(thetas,sigma_mispecification=0):
    num_samples = thetas.shape[0]
    X, y = datasets.make_circles(num_samples, factor=.5, noise=0.05)
    X = StandardScaler().fit_transform(X)
    temp = torch.cat(
        [torch.tensor(X[torch.randperm(X.shape[0])][0]).unsqueeze(0).float() * torch.abs(theta) for theta in
         thetas], dim=0)
    return temp + sigma_mispecification*torch.randn_like(temp)

#Generate D_theta and D_x
n_D = 5000
D_theta = torch.linspace(-3,3,n_D).unsqueeze(-1)
D_x = simulator(D_theta)

#Generate x_0 from unknown theta0
theta_0 = torch.tensor([1.])
n_x0 = 50
x0 = simulator(theta_0.unsqueeze(0).repeat(n_x0,1))
plt.scatter(x0[:,0].numpy(), x0[:,1].numpy())
plt.show()

#Train conditional density estimation
from neural_ratio import ConditionalNeuralDensityRatio
ratio = ConditionalNeuralDensityRatio(D_x,D_theta,[32,32,32])
epochs = 500
batch_size = 500
ratio.train(epochs,batch_size, lr = 1e-3)

#Sample the posterior p(theta|x0, phi)
from samplers import IMH

list_samples = []
for sigma_mispecification in [0.,1.,2.,3.,4.]:
    x0 = simulator(theta_0.unsqueeze(0).repeat(n_x0,1), sigma_mispecification=sigma_mispecification)
    posterior_log_prob = lambda theta: torch.sum(ratio.log_ratio(x0.unsqueeze(0).repeat(theta.shape[0], 1, 1), theta.unsqueeze(1).repeat(1, n_x0, 1)),dim=1) + prior_distribution.log_prob(theta)
    sampler = IMH(posterior_log_prob, D_theta.shape[1], prior_distribution, 500)
    samples = sampler.sample(100)
    list_samples.append(samples)
    plt.hist(samples.numpy(), bins = 50, density = True, label = str(sigma_mispecification))
plt.legend()
plt.show()

current_theta = samples[:1]
gibbs_iterations = 10000
list_theta = [current_theta]
for t in range(gibbs_iterations):
    D_x_plus = torch.cat([D_x, x0], dim=0)
    D_theta_plus = torch.cat([D_theta, current_theta.repeat(n_x0, 1)], dim=0)
    ratio = ConditionalNeuralDensityRatio(D_x,D_theta,[32,32,32])
    ratio.train(epochs, batch_size, 1e-3)

    posterior_log_prob = lambda theta: torch.sum(ratio.log_ratio(x0.unsqueeze(0).repeat(theta.shape[0], 1, 1), theta.unsqueeze(1).repeat(1, n_x0, 1)),dim=1) + prior_distribution.log_prob(theta)
    sampler = IMH(posterior_log_prob, 1, prior_distribution, 1)
    current_theta = sampler.sample(100)
    if t%10==0:
        sampler = IMH(posterior_log_prob, 1, prior_distribution, 500)
        samples = sampler.sample(1000)
        plt.hist(samples.numpy())
        plt.show()
    list_theta.append(current_theta)
torch.save(torch.cat(list_theta, dim =0), 'theta.sav')
