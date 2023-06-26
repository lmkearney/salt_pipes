import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from scipy.special import erf
from matplotlib.gridspec import GridSpec
from random import choice

def find_last(t, t1):
    return t1[t1 < t].max()

def cond_dist(low, upp, theta, sd):
    return truncnorm.rvs((low-theta)/sd, (upp-theta)/sd, loc=theta, scale=sd)

def pdf_analytical_trunc(t, gamma, s):
    val = 2 * gamma/(s*np.sqrt(2*np.pi)) \
          * np.exp(-((gamma*t-1)/(s*np.sqrt(2)))**2) \
          / (1 + erf(1/(s*np.sqrt(2))))
    return val

def pressure_v_time2(t, ts, g):
    p = np.zeros(len(t))
    p0 = 0.
    for i in range(len(t)):
        if (t[i] <= ts[0]):
            p[i] = p0 + g*(t[i])
        else:
            t_last = find_last(t[i], ts)
            p[i] = p0 + g*(t[i] - t_last)
    return p

def simulate(N, gamma, mean_sigma, s):
    ts = np.zeros(N)
    for i in range(1,N):
        dt = cond_dist(0, 100, mean_sigma/gamma, s/gamma)
        ts[i] = ts[i-1] + dt
    return ts

# import data
ts = np.loadtxt('./Levant_data/Oceanus_data.csv', \
                          skiprows=1, delimiter=',', dtype=float).T

thetas_filt = np.load('./Levant_MCMC_results/oceanus_thetas_filt_priors.npy', allow_pickle=True).T

gamma = thetas_filt[:,0]
s = thetas_filt[:,1]
sigma = thetas_filt[:,2]

# colours
c1 = 'k'
c2 = 'forestgreen'
c3 = 'royalblue'
c4 = 'tab:green'
c5 = 'tab:green'

fig = plt.figure(figsize=(8,8./3.))
gs = GridSpec(nrows=2, ncols=2, width_ratios=[2,1], height_ratios=[2, 1])
ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[1,0])
ax2 = fig.add_subplot(gs[:,1])

# simulate model instances
mean_gamma = np.mean(gamma)
mean_s = np.mean(s)
mean_sigma = np.mean(sigma)

TS_mean = simulate(40, mean_gamma, mean_sigma, mean_s)
TS_sample1 = simulate(40, choice(gamma), choice(sigma), choice(s))
TS_sample2 = simulate(40, choice(gamma), choice(sigma), choice(s))

# cut out venting times greater than 1.7 Myr
TS_mean = TS_mean[TS_mean < 1.7]
TS_sample1 = TS_sample1[TS_sample1 < 1.7]
TS_sample2 = TS_sample2[TS_sample2 < 1.7]

t = np.linspace(0., 1.8, 10000)
p0 = 85 # sigma_min, taken from Cartwright et al. (2021)

# plot linearised pressure evolution
ax0.axhline(p0+mean_sigma, c='k', linewidth=0.75, ls=(0,(10,2)))
ax0.axhspan(p0+mean_sigma+mean_s, p0+mean_sigma-mean_s, \
            alpha=0.1, color='gray', lw=0)
ax0.plot(t, p0+pressure_v_time2(t, TS_mean, mean_gamma), \
         c=c2, label='posterior\n  mean', lw=1.5)

# plot Oceanus data and model instances
ax1.scatter(ts, np.full_like(ts, 0.175), color='royalblue')
ax1.scatter(1.7-TS_mean, np.full_like(TS_mean, 1-0.175), \
            color='forestgreen')
ax1.scatter(1.7-TS_sample1, np.full_like(TS_sample1, 1-0.175-0.195), \
            color='#CAE5CA')
ax1.scatter(1.7-TS_sample2, np.full_like(TS_sample2, 1-0.175-2*0.195), \
            color='#CAE5CA')

# plot Oceanus time intervals
dts = -(ts[1:]-ts[:-1])
ax2.hist(dts, bins=6, density=True, color='white', \
         edgecolor='black', linewidth=1., label='Oceanus data')

# plot 100 posterior time interval distributions
tt = np.linspace(0, 0.2, 1000)
g = np.divide(gamma, sigma)
ss = np.divide(s, sigma)
for i in range(100):
    ax2.plot(tt, pdf_analytical_trunc(tt, g[i*10], ss[i*10]), \
             c=c4, alpha=0.01, lw=6)

# plot mean posterior time interval distribution
ax2.plot(tt, pdf_analytical_trunc(tt, mean_gamma/mean_sigma, mean_s/mean_sigma), \
         c=c2, lw=1.5, label='posterior')

ax0.set_ylabel('pressure (MPa)')
ax0.set_xlim([-0.1, 1.7])
ax0.set_ylim([p0, p0+5])
ax0.set_yticks(p0+np.arange(0,6,1))
ax0.set_xticks(np.arange(-0.1, 1.9, 0.2))
ax0.set_xticklabels([], [])

ax1.set_xlabel('time (Ma)')
ax1.set_yticks([0.175, 1-0.175, 1-0.175-0.195, 1-0.175-2*0.195])
ax1.set_yticklabels(['data', '', 'model  \ninstances', ''], linespacing=1)
ax1.set_xlim([1.8, 0])
ax1.set_ylim([0,1])

ax2.set_xlabel('time interval (Myr)')
ax2.set_ylabel('probability density')
ax2.set_xlim([0, 0.2])
ax2.set_ylim([0, 12])
ax2.set_aspect(1.0/ax2.get_data_ratio(), adjustable='box')

ax3 = ax0.twinx()
ticks = [p0+mean_sigma-mean_s, p0+mean_sigma, p0+mean_sigma+mean_s]
ticklabels = ['$\overline{\sigma_T}-s_T$', \
              '$\overline{\sigma_T}$', '$\overline{\sigma_T}+s_T$']
ax3.set_yticks(ticks)
ax3.set_yticklabels(ticklabels)
ax3.set_frame_on(False)
ax3.set_xlim([-0.1, 1.7])
ax3.set_ylim([p0, p0+5])

plt.subplots_adjust(wspace=0.35)
plt.tight_layout()
plt.show()
