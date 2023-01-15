import torch
import numpy

def simulate_Lotka_Volterra(max_time,initial_state, params):
    state = initial_state
    predator = [state[0]]
    prey = [state[1]]
    t=0
    time_record = [t]
    while t<max_time:
        rates = torch.tensor([state[0]*state[1],state[0],state[1],state[0]*state[1]])*torch.exp(params)
        if torch.sum(rates) == 0:
            break
        t += torch.distributions.Exponential(torch.sum(rates)).sample().item()
        time_record.append(t)
        reaction = torch.distributions.Categorical(rates/torch.sum(rates)).sample().item()

        if reaction == 0:
            state[0] += 1
        elif reaction == 1:
            state[0] -= 1
        elif reaction == 2:
            state[1] += 1
        elif reaction == 3:
            state[1] -= 1
        predator.append(state[0])
        prey.append(state[1])
        if len(time_record)>10000:
            break
    return time_record, predator, prey

def sample_signal(time, signal, dt):
    sampled_signal = [signal[0]]
    t=0
    i=1
    while i<len(signal):
        t+=time[i]-time[i-1]
        if t>dt:
            t=0
            sampled_signal.append(signal[i-1])
        i+=1
    return sampled_signal

def acf(x, length):
    return numpy.array([1]+[numpy.corrcoef(x[:-i], x[i:])[0,1]  \
        for i in range(1, length)])

def compute_statistics(time,predator,prey,dt):
    sampled_predator, sampled_prey = numpy.array(sample_signal(time, predator,dt)), numpy.array(sample_signal(time, prey, dt))
    mean_predator, mean_prey = sampled_predator.mean(),sampled_prey.mean()
    log_var_predator, log_var_prey = numpy.log(sampled_predator.var()), numpy.log(sampled_prey.var())
    autocorrelation_predator_lag_1,autocorrelation_predator_lag_2 = acf(sampled_predator, length = 3)[1:]
    autocorrelation_prey_lag_1,autocorrelation_prey_lag_2 = acf(sampled_prey, length = 3)[1:]
    cross_correlation = numpy.correlate(sampled_prey, sampled_predator)
    return [mean_predator,mean_prey,log_var_predator, log_var_prey,autocorrelation_predator_lag_1,autocorrelation_predator_lag_2,autocorrelation_prey_lag_1,autocorrelation_prey_lag_2, cross_correlation]

def simulate(theta = [0.01, 0.5, 1, 0.01], max_time = 30,initial_state= [100,50], dt = .2):
    time, predator, prey = simulate_Lotka_Volterra(max_time,initial_state,theta)
    return torch.tensor(compute_statistics(time, predator,prey,dt)), time, predator, prey