import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

# function for finding most recent venting time before time t
# from an array of venting times t1
def find_last(t, t1):
    return t1[t1 < t].max()

# function for taking a random sample from a truncated
# normal distribution
def truncnorm_sample(low, upp, mean, sd):
    return truncnorm.rvs((low-mean)/sd, (upp-mean)/sd, loc=mean, scale=sd)

# function for simulating a sequence of n venting times
# produced by two coupled pipes
def c_sim(low, upp, gamma, s, n):
    K = len(gamma)
    ts = np.zeros(n)
    dts = np.zeros(n)
    marks = np.zeros(n)
    dt_temp = np.zeros(K)
    sigma = np.zeros(K)
    for i in range(n):
        for j in range(K):
            sigma[j] = truncnorm_sample(low[j], upp[j], SIGMA, s[j])
            dt_temp[j] = sigma[j]/gamma[j]
        mks = np.arange(1, K+1)
        mks = mks[dt_temp.argsort()]
        dt_temp = np.sort(dt_temp)
        dts[i] = dt_temp[0]
        marks[i] = mks[0]
        if (i == 0):
            ts[0] = dts[0]
        else:
            ts[i] = ts[i-1] + dts[i]
    return ts, marks

# function for generating the linearised pressure evolution
# from a set of venting times ts
def pressure_v_time(t, gamma, s):
    p = np.zeros(len(t))
    p0 = 0.
    for i in range(len(t)):
        if (t[i] < ts[0]):
            p[i] = p0 + gamma*(t[i])
        else:
            t_last = find_last(t[i], ts)
            p[i] = p0 + gamma*(t[i] - t_last)
    return p

# parameters for simulation
n = 20
gamma = [20., 20.] # MPa/Myr
s = [0.5, 0.5] # MPa
SIGMA = 2. # MPa

# minimum and maximum tensile strengths
low = np.zeros(2)
upp = low+1e10

# simulate
ts, marks = c_sim(low, upp, gamma, s, n)

# generate linearised pressure evolution from venting times
t = np.linspace(0, max(ts), 10000)
p1 = pressure_v_time(t, gamma[0], s[0])
p2 = pressure_v_time(t, gamma[1], s[1])

fig, axs = plt.subplots(2, 1)

axs[0].plot(t, np.full_like(t, SIGMA), 'k--', linewidth=1)
axs[0].axhspan(SIGMA+s[1], SIGMA-s[1], alpha=0.2, color='gray', lw=0)
axs[0].axhspan(SIGMA+s[0], SIGMA-s[0], alpha=0.2, color='gray', lw=0)

axs[0].plot(t, p1, c='royalblue', alpha=1)
axs[0].set_xticks([], [])
axs[0].set_ylabel('$\Delta p$ (MPa)')
axs[0].set_xlim([0, 1.0])
axs[0].set_ylim([0, SIGMA+3*max(s)])

axs[1].scatter(ts[marks==1], np.full_like(ts[marks==1], 0.35), \
               color='royalblue')
axs[1].scatter(ts[marks==2], np.full_like(ts[marks==2], 0.65), \
               color='tab:orange')
axs[1].set_xlabel('$t$ (Myr)')
axs[1].set_yticks([], [])
axs[1].set_xlim([0, 1.0])
axs[1].set_ylim([0,1])
fig.set_size_inches(4, 2.5)
plt.tight_layout()
plt.show()
