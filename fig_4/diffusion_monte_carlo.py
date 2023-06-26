import numpy as np
from scipy.stats import truncnorm

yr_to_s = 3.154e+7

# function for taking a random sample from a truncated
# normal distribution
def truncnorm_sample(mean, sd, low, upp):
    return truncnorm.rvs((low-mean)/sd, (upp-mean)/sd, loc=mean, scale=sd)

# function for calculating the value of nu/gamma
def nu_gamma(h_s, h_m, alpha_s, alpha_m, poiss_s, poiss_m):
    return h_m/h_s * alpha_m/alpha_s * (1.-2*poiss_m)/(1.-2*poiss_s)

# function for calculating recharge rate from
# tectonic compression alone
def dpdt(p_s, s_r):
    return p_s*s_r

# overpressure per % strain (MPa)
pressure_per_strain = np.array([1.2, 0.2, 0.5, 2.0])

# horizontal strain-rate (%/Myr)
strain_rate = np.array([1., 0.6, 0.2, 3])

# tensile strength (MPa)
sigma_T = np.array([2, 1, 0.5, 4])*1e6

# sandstone thickness (m)
h_s = np.array([150, 50, 50, 200])

# mudstone thickness (m)
h_m = np.array([2500, 250, 2000, 3000])

# sandstone Biot coefficient
alpha_s = np.array([0.62, 0.17, 0.38, 0.83])

# mudstone Biot coefficient
alpha_m = np.array([0.68, 0.35, 0.30, 0.98])

# sandstone Poisson ratio
poiss_s = np.array([0.24, 0.04, 0.20, 0.30])

# mudstone Poisson ratio
poiss_m = np.array([0.25, 0.05, 0.15, 0.30])

# sandstone bulk modulus (Pa)
K_s = np.array([18, 8, 8, 30])*1e9

# mudstone bulk modulus (Pa)
K_m = np.array([15, 17, 5, 33])*1e9

# sandstone porosity
phi_s = np.array([0.22, 0.01, 0.19, 0.24])

# water compressibility (1/Pa)
c_ell = np.array([4.0, 0.1, 3.7, 4.3])*1e-11


# N Monte Carlo samples
N = 20000

tect_pressure_rate = np.zeros(N)
diff_pressure_rate = np.zeros(N)
p_ss = np.zeros(N)
s_rs = np.zeros(N)
nu_over_gamma = np.zeros(N)
for i in range(N):
    p_s = truncnorm_sample(*pressure_per_strain)
    s_r = truncnorm_sample(*strain_rate)
    sigmaT = truncnorm_sample(*sigma_T)
    hs = truncnorm_sample(*h_s)
    hm = truncnorm_sample(*h_m)
    alphas = truncnorm_sample(*alpha_s)
    alpham = truncnorm_sample(*alpha_m)
    poisss = truncnorm_sample(*poiss_s)
    poissm = truncnorm_sample(*poiss_m)
    Ks = truncnorm_sample(*K_s)
    Km = truncnorm_sample(*K_m)
    phis = truncnorm_sample(*phi_s)
    cell = truncnorm_sample(*c_ell)

    nu_over_gamma[i] = nu_gamma(hs, hm, alphas, alpham, poisss, poissm)
    tect_pressure_rate[i] = dpdt(p_s, s_r)
    diff_pressure_rate[i] = dpdt(p_s, s_r)*nu_over_gamma[i]
