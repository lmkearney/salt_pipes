import numpy as np
import matplotlib.pyplot as plt
import os

means_gamma = []
means_s = []

trues_gamma = []
trues_s = []

MCMC_dir = './synthetic_MCMC_results/sim_MCMC_one_pipe/'
for fname in os.listdir(MCMC_dir):
    thetas_filt = np.load(MCMC_dir + fname, allow_pickle=True)
    true_values = fname.split('_')

    # use MCMC file with samples filtered
    # filtering procedure: first 8000 samples removed (burn-in)
    # then every 10th sample is taken (to remove autocorrelation)
    if (true_values[1] == 'filt'):
        # parse file name for true parameter values
        true_values = np.array([true_values[3], true_values[5][:-4]])
        true_values = true_values.astype(float)

        # add true parameter values to lists
        trues_gamma = np.append(trues_gamma, true_values[0])
        trues_s = np.append(trues_s, true_values[1])

        # separate posterior samples into individual parameters
        gamma, s = thetas_filt

        # add mean posterior parameter values to lists
        means_gamma = np.append(means_gamma, np.mean(gamma))
        means_s = np.append(means_s, np.mean(s))

        
fig, axs = plt.subplots(1, 2, figsize=(7,2.5), \
                        gridspec_kw={'width_ratios': [1, 1]})

# separate predicted parameter values based on
# if the true value of s is greater than 1.
cond = trues_s > 1

# values greater than 1
means_gamma_greater = means_gamma[cond]
trues_gamma_greater = trues_gamma[cond]

means_s_greater = means_s[cond]
trues_s_greater = trues_s[cond]

# values less than 1
means_gamma_less = means_gamma[~cond]
trues_gamma_less = trues_gamma[~cond]

trues_s_less = trues_s[~cond]
means_s_less = means_s[~cond]

lims = [0.,10.]

# plot predicted vs true recharge rates
axs[0].plot(lims, lims, 'grey', lw=0.75, zorder=0)
axs[0].scatter(trues_gamma_greater, means_gamma_greater, c='red', s=2.)
axs[0].scatter(trues_gamma_less, means_gamma_less, c='k', s=2.)


# plot predicted vs true tensile strength standard deviations
axs[1].plot(lims, lims, 'grey', lw=0.75, zorder=0)
axs[1].scatter(trues_s_greater, means_s_greater, c='red', s=2., alpha=1)
axs[1].scatter(trues_s_less, means_s_less, c='k', s=2., alpha=1)

axs[0].set_aspect('equal')
axs[0].set_xlim(lims)
axs[0].set_ylim(lims)
axs[0].set_xticks(np.arange(0, 12, 2))
axs[0].set_xlabel('true $\Gamma^*$')
axs[0].set_ylabel('predicted $\Gamma^*$')

axs[1].set_aspect('equal')
axs[1].set_xlim([0,10.])
axs[1].set_ylim(lims)
axs[1].set_xticks(np.arange(0, 12, 2))
axs[1].set_xlabel('true $s^*$')
axs[1].set_ylabel('predicted $s^*$')

plt.tight_layout()
plt.show()
