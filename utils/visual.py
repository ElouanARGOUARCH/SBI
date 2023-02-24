import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy

def plot_1d_unormalized_function(f,x_min=-10, x_max=10, bins=200):
    tt =torch.linspace(x_min, x_max, 200)
    values = f(tt)
    plt.plot(tt, values*bins/(torch.sum(values)*(x_max - x_min)))

def plot_1d_unormalized_values(values,tt):
    x_min, x_max, bins = tt[0], tt[-1], tt.shape[0]
    plt.plot(tt, values*bins/(torch.sum(values)*(x_max - x_min)))

def plot_2d_function(f, x_min = -10,x_max = 10, y_min = -10, y_max = 10, delta = 50, alpha = 0.7, new_figure = True):
    with torch.no_grad():
        if new_figure :
            plt.figure(figsize = (10,10))
        tt_x = torch.linspace(x_min, x_max, delta)
        tt_y = torch.linspace(y_min,y_max, delta)
        mesh = torch.cartesian_prod(tt_x, tt_y)
        with torch.no_grad():
            plt.pcolormesh(tt_x,tt_y,f(mesh).numpy().reshape(delta,delta).T, cmap = matplotlib.cm.get_cmap('viridis'), alpha = alpha, lw = 0)

def plot_likelihood_function(log_likelihood, x_min = -10,x_max = 10, y_min = -10, y_max = 10, delta_x = 100,delta_y=100, levels = 2 , alpha = 0.7, new_figure = True):
    with torch.no_grad():
        if new_figure :
            plt.figure(figsize = (10,10))
        tt_x = torch.linspace(x_min, x_max, delta_x)
        tt_y = torch.linspace(y_min,y_max, delta_y)
        tt_x_plus = tt_x.unsqueeze(0).unsqueeze(-1).repeat(tt_y.shape[0],1,1)
        tt_y_plus = tt_y.unsqueeze(1).unsqueeze(-1).repeat(1,tt_x.shape[0], 1)
        with torch.no_grad():
            plt.contourf(tt_x,tt_y,torch.exp(log_likelihood(tt_y_plus, tt_x_plus)), levels = levels, cmap = matplotlib.cm.get_cmap('viridis'), alpha = alpha, lw = 0)

def plot_2d_points(samples):
    assert samples.shape[-1]==2,'Requires 2-dimensional points'
    plt.scatter(samples[:,0], samples[:,1])

def plot_image_2d_points(samples, bins = (200,200),range=None,figsize = (12,8)):
    assert samples.shape[-1]==2, 'Requires 2-dimensional points'
    fig = plt.figure(figsize)
    hist_accepted_samples, x_edges, y_edges = numpy.histogram2d(samples[:, 1].numpy(), samples[:, 0].numpy(),bins)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.imshow(torch.flip(torch.flip(torch.tensor(hist_accepted_samples).T, [0, 1]), [0, 1]),extent=[0, bins[1], 0, bins[0]])