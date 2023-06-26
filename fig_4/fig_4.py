import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from uplift_monte_carlo import *
from focusing_monte_carlo import *
from diseq_comp_monte_carlo import *
from diffusion_monte_carlo import *
from data_histogram import *

# function to estimate pdf
# using Gaussian kernel density
def g_kde(x, data):
    gkde = gaussian_kde(np.log10(data), bw_method=0.2)
    gkde = gkde.pdf(x)
    gkde[gkde < 0.02] = np.nan
    return gkde

# colours
c1 = '#648FFF'
c2 = '#785EF0'
c3 = '#DC267F'
c4 = '#FE6100'
c5 = '#FFB000'

# plot estimated recharge rates from various hypotheses
tt = np.linspace(-2, 2, 1001)

plt.plot(tt, g_kde(tt, uplift_pressure_rate), c=c2, label='flow focusing: marginal uplift', ls=(0, (3,1,1,1)))
plt.plot(tt, g_kde(tt, focus_pressure_rate), c=c3, label='flow focusing: folding', ls=(0,(2,1)))
plt.plot(tt, g_kde(tt, diseq_pressure_rate), c='orangered', label='disequilibrium compaction', ls=(0,(1,0.5)))
plt.plot(tt, g_kde(tt, tect_pressure_rate), c='gold', label='tectonic compression',ls=(0,(4,1)))
plt.plot(tt, g_kde(tt, diff_pressure_rate), c='black', label='pressure diffusion', ls=(0,(6,1)))

# plot recharge rates inferred from Levant data
plt.plot(tt, g_kde(tt, data_hist), c='forestgreen', label='Levant data (inferred)')

fig = plt.gcf()
fig.set_size_inches(5,2)

ax = plt.gca()
ax.set_xticks([-2, -1, 0, 1, 2])
ax.set_xticklabels(['10$^{-2}$', '10$^{-1}$', '10$^{0}$', '10$^{1}$', '10$^{2}$'])

plt.xlabel('recharge rate, $\Gamma$ (MPa/Myr)')
plt.ylabel('probability density')
plt.xlim([-2, 2])
plt.ylim([0, 8])

plt.tight_layout()
plt.show()
