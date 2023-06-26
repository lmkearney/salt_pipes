import numpy as np
import matplotlib.pyplot as plt
import os


trues_gamma1 = []
trues_s1 = []
trues_gamma2 = []
trues_s2 = []

means_gamma1 = []
means_s1 = []
means_gamma2 = []
means_s2 = []

MCMC_dir = './synthetic_MCMC_results/sim_MCMC_coupled/'
for fname in os.listdir(MCMC_dir):
    thetas_filt = np.load(MCMC_dir + fname, allow_pickle=True).T  
    true_values = fname.split('_')

    # use MCMC file with samples filtered
    # filtering procedure: first 8000 samples removed (burn-in)
    # then every 10th sample is taken (to remove autocorrelation)
    if (true_values[1] == 'filt'):
        thetas_filt = thetas_filt.T

        # parse file name for true parameter values
        true_values = np.array([true_values[5], true_values[8], \
                            true_values[11], true_values[14][:-4]])
        true_values = true_values.astype(float)

        # add true parameter values to lists
        trues_gamma1 = np.append(trues_gamma1, true_values[0])
        trues_s1 = np.append(trues_s1, true_values[2])

        trues_gamma2 = np.append(trues_gamma2, true_values[1])
        trues_s2 = np.append(trues_s2, true_values[3])
        
        # separate posterior samples into individual parameters
        gamma1, gamma2, s1, s2, _ = thetas_filt

        # add mean posterior parameter values to lists
        means_gamma1 = np.append(means_gamma1, np.mean(gamma1))
        means_s1 = np.append(means_s1, np.mean(s1))

        means_gamma2 = np.append(means_gamma2, np.mean(gamma2))
        means_s2 = np.append(means_s2, np.mean(s2))

fig, axs = plt.subplots(1, 2, figsize=(7,2.5), gridspec_kw={'width_ratios': [1, 1]})

axs[0].plot([0.,5.], [0.,5.], 'grey', lw=0.75, zorder=0)
axs[0].scatter(trues_gamma1, means_gamma1, c='royalblue', s=2.)
axs[0].scatter(trues_gamma2, means_gamma2, c='tab:orange', s=2.)

axs[1].plot([0.,1.], [0.,1.], 'grey', lw=0.75, zorder=0)
axs[1].scatter(trues_s1, means_s1, c='royalblue', s=2., alpha=1)
axs[1].scatter(trues_s2, means_s2, c='tab:orange', s=2., alpha=1)

axs[0].set_aspect('equal')
axs[0].set_xlim([0.,5.])
axs[0].set_ylim([0.,5.])
axs[0].set_xticks(np.arange(0, 6, 1))
axs[0].set_xlabel('true $\Gamma^*$')
axs[0].set_ylabel('predicted $\Gamma^*$')

axs[1].set_aspect('equal')
axs[1].set_xlim([0,1.])
axs[1].set_ylim([0,1.])
axs[1].set_xticks(np.arange(0, 1.2, 0.2))
axs[1].set_xlabel('true $s^*$')
axs[1].set_ylabel('predicted $s^*$')

plt.tight_layout()
plt.show()
