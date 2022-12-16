import sys, os
import numpy as np
import scipy.special as sp
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
from numba import njit,int64,float64,prange
import torch
import pyro
#%%
@njit(float64(float64[::1]), cache = True,)
def iat(x):
    C=10.
    n=len(x)
    m=np.mean(x)
    y=x-m
    corr0=np.sum(y**2)/(n)
    tau=1.
    for lag in range(1,n):
        corr=np.sum(y[lag:]*y[:n-lag])/(n-lag)/corr0
        tau+=2*corr
        if C*tau<lag:
            break
    # print('tau',tau,l)
    return tau
#%% prior
mu0=0
sigma0=1
a0=2
b0=2
V0=10
#%% dataset
n=30
X=np.random.standard_normal(size=n)*sigma0+mu0
#%% exact posterior
xs=np.linspace(0,4,1024)
xm=np.linspace(-2,2,1024)
s1=np.sum(X)
s2=np.sum(X**2)
Vn=1/(1/V0+n)
mun=Vn*(mu0/V0+s1)
an=a0+.5*n
bn=b0+.5*(mu0**2/V0+s2-mun**2/Vn)
print('exact marg means',mun,bn/(an-1))
fig=plt.figure()
ax=fig.add_subplot(121)
psig=st.invgamma.pdf(xs, an, 0, bn)
ax.plot(xs,psig,label=r'$\sigma^2$ marg post')
pmu=st.t.pdf(xm,2*an,loc=mun,scale=np.sqrt(Vn*bn/an))
bx=fig.add_subplot(122)
bx.plot(xm,pmu,label=r'$\mu$ marg post')
#%% Standard Gibbs
niter=100000
m=np.random.standard_normal()*np.sqrt(V0)*sigma0+mu0
#s=b0/np.random.gamma(a0)
trace=np.empty((2,niter))
for it in range(niter):
    b=b0+.5*(s2+n*(s1/n-m)**2)#np.sum((X-m)**2)#
    s=b/np.random.gamma(an)
    m=np.random.standard_normal()*np.sqrt(Vn*s)+mun
    trace[0,it]=m
    trace[1,it]=s
#print(pyro.ops.stats.effective_sample_size(torch.tensor(trace[0]).unsqueeze(0), chain_dim = 0, sample_dim = 1))
print('Gibbs marg means',np.mean(trace[0,niter//2:]),np.mean(trace[1,niter//2:]))
ax.hist(trace[1,niter//2:],bins=100, histtype='step', label='Gibbs',density=True)#,alpha=0.33)
bx.hist(trace[0,niter//2:],bins=100, histtype='step', label='Gibbs',density=True)#,alpha=0.33)
print("iat",iat(trace[0,niter//2:]),iat(trace[1,niter//2:]))
#%% Multiple imputations mean
M=np.random.standard_normal(size=n)*np.sqrt(V0)*sigma0+mu0
trace=np.empty((2,niter))
for it in range(niter):
    b=b0+.5*np.sum((X-M)**2)#
    s=b/np.random.gamma(an)
    M=np.random.standard_normal(size=n)*np.sqrt(Vn*s)+mun
    trace[0,it]=M[0]
    trace[1,it]=s
print('Gibbs X means marg means',np.mean(trace[0,niter//2:]),np.mean(trace[1,niter//2:]))
ax.hist(trace[1,niter//2:],bins=100, histtype='step', label='Gibbs X mean',density=True)#,alpha=0.33)
bx.hist(trace[0,niter//2:],bins=100, histtype='step', label='Gibbs X mean',density=True)#,alpha=0.33)
print("iat",iat(trace[0,niter//2:]),iat(trace[1,niter//2:]))

#%% Multiple imputations variance
m=np.random.standard_normal()*np.sqrt(V0)*sigma0+mu0
trace=np.empty((2,niter))
for it in range(niter):
    b=b0+.5*(s2+n*(s1/n-m)**2)#
    S=b/np.random.gamma(an,size=n)
    seff=n/np.sum(1/S)
    mueff = Vn*(mu0/V0+np.sum(X/S)*seff)
    m=np.random.standard_normal()*np.sqrt(Vn*seff)+mueff
    trace[0,it]=m
    trace[1,it]=S[0]
print('Gibbs X var marg means',np.mean(trace[0,niter//2:]),np.mean(trace[1,niter//2:]))
ax.hist(trace[1,niter//2:],bins=100, histtype='step', label='Gibbs X var',density=True)#,alpha=0.33)
bx.hist(trace[0,niter//2:],bins=100, histtype='step', label='Gibbs X var',density=True)#,alpha=0.33)
print("iat",iat(trace[0,niter//2:]),iat(trace[1,niter//2:]))
#%% Multiple imputations mean/variance
M=np.random.standard_normal(size=n)*np.sqrt(V0)*sigma0+mu0
trace=np.empty((2,niter))
for it in range(niter):
    b=b0+.5*np.sum((X-M)**2)#
    S=b/np.random.gamma(an,size=n)
    seff=n/np.sum(1/S)
    mueff = Vn*(mu0/V0+np.sum(X/S)*seff)
    M=np.random.standard_normal(size=n)*np.sqrt(Vn*seff)+mueff
    trace[0,it]=M[0]
    trace[1,it]=S[0]
print('Gibbs X means-var marg means',np.mean(trace[0,niter//2:]),np.mean(trace[1,niter//2:]))
ax.hist(trace[1,niter//2:],bins=100, histtype='step', label='Gibbs X mean-var',density=True)#,alpha=0.33)
bx.hist(trace[0,niter//2:],bins=100, histtype='step', label='Gibbs X mean-var',density=True)#,alpha=0.33)
print("iat",iat(trace[0,niter//2:]),iat(trace[1,niter//2:]))
#%% Show
ax.legend(loc='upper right')
bx.legend(loc='upper right')
plt.tight_layout()
plt.show()