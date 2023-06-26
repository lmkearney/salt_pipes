import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from import_data import *


# function for taking a random sample from a truncated
# normal distribution
def truncnorm_sample(low, upp, mean, sd):
    return truncnorm.rvs((low-mean)/sd, (upp-mean)/sd, loc=mean, scale=sd)

thetas_dir = './Levant_MCMC_results/MCMC_triple-wise/'
thetas12 = np.load(thetas_dir + 'thetas_filt_1_2_3.npy', allow_pickle=True)
thetas23 = np.load(thetas_dir + 'thetas_filt_2_3_4.npy', allow_pickle=True)
thetas34 = np.load(thetas_dir + 'thetas_filt_3_4_5.npy', allow_pickle=True)
thetas45 = np.load(thetas_dir + 'thetas_filt_4_5_6.npy', allow_pickle=True)
thetas56 = np.load(thetas_dir + 'thetas_filt_5_6_7.npy', allow_pickle=True)
thetas67 = np.load(thetas_dir + 'thetas_filt_6_7_8.npy', allow_pickle=True)
thetas78 = np.load(thetas_dir + 'thetas_filt_7_8_9.npy', allow_pickle=True)
thetas89 = np.load(thetas_dir + 'thetas_filt_8_9_10.npy', allow_pickle=True)
thetas910 = np.load(thetas_dir + 'thetas_filt_9_10_11.npy', allow_pickle=True)
thetas1011 = np.load(thetas_dir + 'thetas_filt_10_11_12.npy', allow_pickle=True)

thetas = [thetas12, thetas23, thetas34, thetas45,
          thetas56, thetas67, thetas78, thetas89,
          thetas910, thetas1011]
gammas = []
ss = []
prob_phi_1_pair_1 = []
prob_phi_1_pair_2 = []

for i in range(len(thetas)):   
    gammas.append(thetas[i][0])
    gammas.append(thetas[i][1])
    gammas.append(thetas[i][2])

    ss.append(thetas[i][3])
    ss.append(thetas[i][4])
    ss.append(thetas[i][5])

    # calculate probability that phi = 1 for each pair,
    # equal to the mean value of the posterior phi distribution
    prob_phi_1_pair_1.append(np.mean(thetas[i][-2]))
    prob_phi_1_pair_2.append(np.mean(thetas[i][-1]))
    
# combine samples of gamma obtained from each triple-wise inference
# e.g., samples for trail 3 from inferences of (1,2,3), (2,3,4) and (3,4,5)
gammas_combine = []
for i in range(len(thetas)):
    if (i == 0):
        gammas_combine.append(thetas[0][0]) 
        gammas_combine.append(thetas[0][1])
        gammas_combine.append(thetas[0][2]) 
    elif (i == 1):
        gammas_combine[1] = np.concatenate((gammas_combine[1], thetas[1][0]))
        gammas_combine[2] = np.concatenate((gammas_combine[2], thetas[1][1]))
        gammas_combine.append(thetas[1][2])
    else:
        gammas_combine[i] = np.concatenate((gammas_combine[i], thetas[i][0]))
        gammas_combine[i+1] = np.concatenate((gammas_combine[i+1], thetas[i][1])) 
        gammas_combine.append(thetas[i][2])

# convert distribution of \Gamma/\overline{\sigma}_T to
# distribution of \Gamma using prior distribution of \overline{\sigma}_T
for i in range(len(gammas_combine)):
    for j in range(len(gammas_combine[i])):
        gammas_combine[i][j] *= truncnorm_sample(0, 100, 2, 1)

# calculate Bayes' factors, denoted as B, for each pair of pipes
# we do this using the probabilites that phi = 1, denoted as P,
# with B = P/(1-P)
prob_phi_1_pair_1 = np.array(prob_phi_1_pair_1)
prob_phi_1_pair_2 = np.array(prob_phi_1_pair_2)

logB1s = -np.log10((1-prob_phi_1_pair_1)/prob_phi_1_pair_1)
logB1s[logB1s == np.inf] = 3
logB1s[logB1s == -np.inf] = -3

logB2s = -np.log10((1-prob_phi_1_pair_2)/prob_phi_1_pair_2)
logB2s[logB2s == np.inf] = 3
logB2s[logB2s == -np.inf] = -3

# average the two log-Bayes factors calculated for each pair of pipes
logBs = np.zeros(11)
logBs[0] = logB1s[0]
logBs[-1] = logB2s[-1]
logBs[1:-1] = 0.5*(logB1s[1:]+logB2s[:-1])

fig, axs = plt.subplots(1, 3, figsize=(7, 2.5), gridspec_kw={'width_ratios': [4, 1, 1]})

# plot Levant pipe data
for i in range(1,13):
    pd = abs(pipedict[i])    
    axs[0].scatter(pd, np.full_like(pd, i), c='royalblue')
axs[0].invert_xaxis()

# plot posterior distributions of recharge rates
parts = axs[1].violinplot(gammas_combine, positions=range(1, 13), \
                        showmeans=False, showextrema=False, vert=False)
for pc in parts['bodies']:
    pc.set_facecolor('royalblue')
    pc.set_alpha(1)
for i in range(len(parts['bodies'])):
    pc = parts['bodies'][i]
    pc.set_facecolor('royalblue')

# plot Bayes' factors on a log-scale
axs[2].barh(np.arange(0,len(logBs))+1.5, logBs, color='royalblue', zorder=10)

axs[0].axhline(6.5, c='k', ls='--', lw=0.5, zorder=0)
axs[0].axhline(8.5, c='k', ls='--', lw=0.5, zorder=0)
axs[0].axhline(9.5, c='k', ls='--', lw=0.5, zorder=0)

axs[1].axhline(6.5, c='k', ls='--', lw=0.5, zorder=0)
axs[1].axhline(8.5, c='k', ls='--', lw=0.5, zorder=0)
axs[1].axhline(9.5, c='k', ls='--', lw=0.5, zorder=0)

axs[2].axvline(0, c='k', ls='--', lw=0.5)

axs[0].set_xlim([2.25, 0])
axs[0].set_xlabel('time (Ma)')
axs[0].set_ylabel('#(pipe)')

axs[1].set_xlim([0, 100])
axs[1].set_xlabel('recharge rate, \n $\Gamma$ (MPa/Myr)')

axs[2].set_xlabel('Bayes factor') 
axs[2].set_xlim([-3, 3])
axs[2].set_xticks([-3, -1, 1, 3])
axs[2].set_xticklabels(['$10^{-3}$', '$10^{-1}$', '$10^{1}$', '$10^{3}$'])

axs[0].set_yticks(np.arange(1, 13))
axs[1].set_yticks(np.arange(1, 13))
axs[2].set_yticks(np.arange(1, 13))

axs[0].set_ylim([0.5, 12.5])
axs[1].set_ylim([0.5, 12.5])
axs[2].set_ylim([0.5, 12.5])

plt.tight_layout()
plt.show()
