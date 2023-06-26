import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

yr_to_s = 3.154e+7

def tn_rvs(mean, sd, low, upp):
    return truncnorm.rvs((low-mean)/sd, (upp-mean)/sd, loc=mean, scale=sd)

def dpdt(p_s, s_r):
    return p_s*s_r


pressure_per_strain = np.array([1.2, 0.1, 0.5, 2.0]) #*1e6
strain = np.array([1., 0.2, 0.2, 3]) 
duration = np.array([5.5, 0.5, 5, 6])


dt = 80e3 * yr_to_s

N = 20000
tect_pressure_rate = np.zeros(N)
p_ss = np.zeros(N)
s_rs = np.zeros(N)

for i in range(N):
    p_s = tn_rvs(*pressure_per_strain)
    s_r = tn_rvs(*strain_rate)
    
    tect_pressure_rate[i] = dpdt(p_s, s_r)
    p_ss[i] = p_s
    s_rs[i] = s_r

#plt.hist(s_rs, density=True, bins=20)
#plt.hist(pressure_rate, density=True, bins=20)
#plt.show()
