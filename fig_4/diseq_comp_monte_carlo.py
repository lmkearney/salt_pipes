import numpy as np
from scipy.stats import truncnorm

yr_to_s = 3.154e+7
g = 9.81

# function for taking a random sample from a truncated
# normal distribution
def truncnorm_sample(mean, sd, low, upp):
    return truncnorm.rvs((low-mean)/sd, (upp-mean)/sd, loc=mean, scale=sd)

# function for calculating recharge rate from
# disequilibrium compaction
def dpdt(rho, h, t):
    return rho * g * h / t

# post-salt sediment density (kg/m3)
rho = np.array([2.0, 0.1, 1.8, 2.5]) * 1e3

# post-salt sediment thickness (m)
h = np.array([355, 25, 300, 400])

# sedimentation duration (Myr)
t = np.array([5, 0.5, 4, 6])

# N Monte Carlo samples
N = 20000

diseq_pressure_rate = np.zeros(N)
rhos = np.zeros(N)
hs = np.zeros(N)
ts = np.zeros(N)
for i in range(N):
    rhos = truncnorm_sample(*rho)
    hs = truncnorm_sample(*h)
    ts = truncnorm_sample(*t)
    
    diseq_pressure_rate[i] = dpdt(rhos, hs, ts)/1e6
