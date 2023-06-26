import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from import_data import *

# function for taking a random sample from a truncated
# normal distribution
def truncnorm_sample(low, upp, mean, sd):
    return truncnorm.rvs((low-mean)/sd, (upp-mean)/sd, loc=mean, scale=sd)

thetas_dir = './Levant_MCMC_results/MCMC_pair-wise/'
thetas12 = np.load(thetas_dir + 'thetas_filt_1_2.npy', allow_pickle=True)
thetas23 = np.load(thetas_dir + 'thetas_filt_2_3.npy', allow_pickle=True)
thetas34 = np.load(thetas_dir + 'thetas_filt_3_4.npy', allow_pickle=True)
thetas45 = np.load(thetas_dir + 'thetas_filt_4_5.npy', allow_pickle=True)
thetas56 = np.load(thetas_dir + 'thetas_filt_5_6.npy', allow_pickle=True)
thetas67 = np.load(thetas_dir + 'thetas_filt_6_7.npy', allow_pickle=True)
thetas78 = np.load(thetas_dir + 'thetas_filt_7_8.npy', allow_pickle=True)
thetas89 = np.load(thetas_dir + 'thetas_filt_8_9.npy', allow_pickle=True)
thetas910 = np.load(thetas_dir + 'thetas_filt_9_10.npy', allow_pickle=True)
thetas1011 = np.load(thetas_dir + 'thetas_filt_10_11.npy', allow_pickle=True)
thetas1112 = np.load(thetas_dir + 'thetas_filt_11_12.npy', allow_pickle=True)

thetas = [thetas12, thetas23, thetas34, thetas45,
          thetas56, thetas67, thetas78, thetas89,
          thetas910, thetas1011, thetas1112]

gammas = []
ss = []
prob_phi_1 = []
for i in range(len(thetas)):   
    gammas.append(thetas[i][0])
    gammas.append(thetas[i][1])

    ss.append(thetas[i][2])
    ss.append(thetas[i][3])

    # calculate probability that phi = 1,
    # equal to the mean value of the posterior phi distribution
    prob_phi_1.append(np.mean(thetas[i][-1]))

# combine samples of gamma obtained from each pair-wise inference
# e.g., samples for trail 3 from inferences of (2,3) and (3,4)
gammas_combine = []
gammas_combine.append(gammas[0])
for i in range(1,len(gammas)-1, 2):
    gammas_combine.append(np.concatenate((gammas[i], gammas[i+1])))
gammas_combine.append(gammas[-1])

# convert distribution of \Gamma/\overline{\sigma}_T to
# distribution of \Gamma using prior distribution of \overline{\sigma}_T
for i in range(len(gammas_combine)):
    for j in range(len(gammas_combine[i])):
        gammas_combine[i][j] *= truncnorm_sample(0, 100, 2, 1)

# calculate Bayes' factors, denoted as B, for each pair of pipes
# we do this using the probabilites that phi = 1, denoted as P,
# with B = P/(1-P)
prob_phi_1 = np.array(prob_phi_1)
logBs = -np.log10((1-prob_phi_1)/prob_phi_1)
logBs[logBs == np.inf] = 3
logBs[logBs == -np.inf] = -3

# colour data and results according to values of Bayes factors
# using the interpretation of Kass & Raftery (1995)
pos = 'forestgreen'
mid = '#c3d9c3'
neg = '#c83737'

# Bayes' factors colours
clrs = []
for i in range(len(logBs)):
    if (logBs[i]<-1):
        clrs.append(neg)
    elif (logBs[i]>1):
        clrs.append(pos)
    elif (abs(logBs[i])<1):
        clrs.append(mid)

# recharge rates colours
clrs2 = np.copy(clrs)
clrs2 = np.append(clrs, neg)
clrs2[4] = pos # trail 5 is decisively coupled to trail 4
clrs2[5] = mid # trail 6 may/may not be coupled to trail 5
clrs2[7] = pos # trail 8 is decisively coupled to trail 7

# venting data colours
clrs3 = np.copy(clrs2)
clrs3[clrs3 == pos] = 'royalblue'
clrs3[clrs3 == mid] = '#bfc8e1'

fig, axs = plt.subplots(1, 3, figsize=(7, 2.5), gridspec_kw={'width_ratios': [4, 1, 1]})

# plot Levant pipe data
for i in range(1,13):
    pd = abs(pipedict[i])    
    axs[0].scatter(pd, np.full_like(pd, i), c=clrs3[i-1])

# plot posterior distributions of recharge rates
parts = axs[1].violinplot(gammas_combine, positions=range(1, 13), \
                        showmeans=False, showextrema=False, vert=False)
for pc in parts['bodies']:
    pc.set_facecolor('forestgreen')
    pc.set_alpha(1)
for i in range(len(parts['bodies'])):
    pc = parts['bodies'][i]
    pc.set_facecolor(clrs2[i])
    
# plot Bayes' factors on a log-scale
axs[2].barh(np.arange(0,len(logBs))+1.5, logBs, color=clrs, zorder=10)

axs[0].axhline(6.5, c='k', ls='--', lw=0.5, zorder=0)
axs[0].axhline(8.5, c='k', ls='--', lw=0.5, zorder=0)
axs[0].axhline(9.5, c='k', ls='--', lw=0.5, zorder=0)

axs[1].axhline(6.5, c='k', ls='--', lw=0.5, zorder=0)
axs[1].axhline(8.5, c='k', ls='--', lw=0.5, zorder=0)
axs[1].axhline(9.5, c='k', ls='--', lw=0.5, zorder=0)

axs[2].axvline(0, c='k', ls='--', lw=0.5)

axs[0].set_xlim([0.0, 2.25])
axs[0].invert_xaxis()
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

axs[2].set_ylim([0.5, 12.5])
axs[1].set_ylim([0.5, 12.5])
axs[0].set_ylim([0.5, 12.5])

plt.tight_layout()
plt.show()
