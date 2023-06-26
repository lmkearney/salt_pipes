import numpy as np

# import data
thetas_dir = '../Levant_MCMC_results/MCMC_pair-wise/'
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

# combine all samples of gamma together into one array
data_hist = np.array([])
for i in range(len(gammas_combine)):
    data_hist = np.concatenate((data_hist, gammas_combine[i]))
