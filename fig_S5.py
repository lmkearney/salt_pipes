import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, truncnorm, gaussian_kde
from scipy.special import erf, erfc
from matplotlib.gridspec import GridSpec

ts = np.loadtxt('./Levant_data/Oceanus_data.csv', \
                          skiprows=1, delimiter=',', dtype=float).T
dts = -(ts[1:]-ts[:-1])

# colours
c1 = 'k'
c2 = 'forestgreen'
c3 = 'royalblue'
c4 = 'tab:green'
c5 = 'tab:green'

def pdf_truncnorm(t, mu, sig):
    val = 2./(sig*np.sqrt(2*np.pi)) \
          * np.exp(-((t-mu)/(sig*np.sqrt(2)))**2) \
          / erfc(-mu/(sig*np.sqrt(2)))
    return val

# load results of Bayesian inference for Oceanus trail
thetas_filt = np.load('./Levant_MCMC_results/oceanus_thetas_filt_priors.npy', allow_pickle=True).T
gamma = thetas_filt[:,0]
s = thetas_filt[:,1]
sigma = thetas_filt[:,2]

mean_gamma = np.mean(gamma)
mean_s = np.mean(s)
mean_sigma = np.mean(sigma)

# estimate posterior pdf of \overline{\sigma_T}
# using Gaussian kernel density
tt_sig = np.linspace(0, 5, 5000)
gkde_sig = gaussian_kde(sigma, bw_method=0.3)
f_kde_sig = gkde_sig.pdf(tt_sig)

# estimate posterior pdf of \Gamma
# using Gaussian kernel density
tt_g = np.linspace(0, 60., 1000)
gkde_g = gaussian_kde(gamma, bw_method=0.3)
f_kde_g = gkde_g.pdf(tt_g)

# estimate posterior pdf of s_T
# using Gaussian kernel density
tt_s = np.linspace(0, 3, 1000)
gkde_s = gaussian_kde(s, bw_method=0.3)
f_kde_s = gkde_s.pdf(tt_s)

# parameters for generating prior distributions
prior_mean_sigma = 2.
prior_sd_sigma = 1.

prior_mean_gamma = prior_mean_sigma/np.median(dts)
prior_sd_gamma = prior_mean_sigma/np.median(dts)/2.

prior_mean_s = prior_mean_sigma * np.std(dts)/np.median(dts)
prior_sd_s = prior_mean_sigma * np.std(dts)/np.median(dts)/2.

fig = plt.figure(figsize=(9,3))
gs = GridSpec(nrows=1, ncols=3, width_ratios=[1,1,1])

ax3 = fig.add_subplot(gs[0,0])
ax4 = fig.add_subplot(gs[0,1])
ax5 = fig.add_subplot(gs[0,2])

# plot prior and posterior distributions for \Gamma
ax3.plot(tt_g, pdf_truncnorm(tt_g, prior_mean_gamma, \
                        prior_sd_gamma), \
         c='k', ls=(0, (1,1)))
ax3.plot(tt_g, f_kde_g, c=c1)
ax3.scatter(mean_gamma, gkde_g(mean_gamma), c=c5, zorder=10, s=15.)

# plot prior and posterior distributions for \overline{\sigma_T}
ax4.plot(tt_sig, pdf_truncnorm(tt_sig, prior_mean_sigma, \
                               prior_sd_sigma), \
         c='k', ls=(0, (1,1)))
ax4.plot(tt_sig, f_kde_sig, c=c1)
ax4.scatter(mean_sigma, gkde_sig(mean_sigma), c=c5, zorder=10, s=15.)

# plot prior and posterior distributions for s_T
ax5.plot(tt_s, pdf_truncnorm(tt_s, prior_mean_s, \
                        prior_sd_s), \
         c='k', ls=(0, (1,1)), label='prior')
ax5.plot(tt_s, f_kde_s, c=c1, label='posterior')
ax5.scatter(mean_s, gkde_s(mean_s), c=c5, zorder=10, label='post. mean', s=15.)

ax3.set_xlabel('recharge rate, \n $\Gamma$ (MPa/Myr)')
ax3.set_ylabel('probability density') 
ax3.set_xlim([0, 60])
ax3.set_ylim([0, 0.06])
ax3.set_aspect(1.0/ax3.get_data_ratio(), adjustable='box')

ax4.set_xlabel('mean tensile strength, \n $\overline{\sigma_T}$ (MPa)')
ax4.set_xlim([0, 5])
ax4.set_ylim([0, 0.6])
ax4.set_aspect(1.0/ax4.get_data_ratio(), adjustable='box')

ax5.set_xlabel('tensile strength standard deviation, \n $s_T$ (MPa)')
ax5.set_xlim([0, 3])
ax5.set_ylim([0, 1.2])
ax5.set_aspect(1.0/ax5.get_data_ratio(), adjustable='box')

plt.subplots_adjust(wspace=0.35)
plt.tight_layout()
plt.show()
