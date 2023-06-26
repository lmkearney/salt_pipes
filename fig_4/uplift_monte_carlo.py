import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

yr_to_s = 3.154e+7
g = 9.81

def tn_rvs(mean, sd, low, upp):
    return truncnorm.rvs((low-mean)/sd, (upp-mean)/sd, loc=mean, scale=sd)

def dpdt(theta, rho_m, rho_l, L, t):
    return 1./2. * (rho_m - rho_l) * g * np.sin(theta * np.pi/180.) * L / t

theta = np.array([3, 1, 0, 5])
rho_m = np.array([2.35, 0.1, 1.8, 2.5]) * 1e3
rho_l = np.array([1.06, 0.1, 0.9, 1.2]) * 1e3
L = np.array([1000, 1000, 100, 10000])
t = np.array([5, 0.5, 4, 6])

N = 20000
uplift_pressure_rate = np.zeros(N)
rhoms = np.zeros(N)
rhols = np.zeros(N)
Ls = np.zeros(N)
ts = np.zeros(N)    

for i in range(N):
    thetas = tn_rvs(*theta)
    rhoms = tn_rvs(*rho_m)
    rhols = tn_rvs(*rho_l)
    Ls = tn_rvs(*L)
    ts = tn_rvs(*t)
    
    uplift_pressure_rate[i] = dpdt(thetas, rhoms, rhols, Ls, ts)/1e6
