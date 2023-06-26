import numpy as np
from scipy.stats import truncnorm

yr_to_s = 3.154e+7
g = 9.81

# function for taking a random sample from a truncated
# normal distribution
def truncnorm_sample(mean, sd, low, upp):
    return truncnorm.rvs((low-mean)/sd, (upp-mean)/sd, loc=mean, scale=sd)

# function for calculating recharge rate from
# flow focusing due to folding
def dpdt(rho_m, rho_l, h, t):
    return 2./3. * (rho_m - rho_l) * g * h / t

# mudstone density (kg/m3)
rho_m = np.array([2.35, 0.1, 1.8, 2.5]) * 1e3

# water density (kg/m3)
rho_l = np.array([1.06, 0.1, 0.9, 1.2]) * 1e3

# sandstone thickness (m)
h = np.array([200, 25, 100, 300])

# folding duration (Myr)
t = np.array([5, 0.5, 4, 6])

# N Monte Carlo samples
N = 20000

focus_pressure_rate = np.zeros(N)
rhoms = np.zeros(N)
rhols = np.zeros(N)
hs = np.zeros(N)
ts = np.zeros(N)
for i in range(N):
    rhoms = truncnorm_sample(*rho_m)
    rhols = truncnorm_sample(*rho_l)
    hs = truncnorm_sample(*h)
    ts = truncnorm_sample(*t)
    
    focus_pressure_rate[i] = dpdt(rhoms, rhols, hs, ts)/1e6
