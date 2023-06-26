import numpy as np
from scipy.stats import truncnorm, bernoulli
from scipy.special import erf

fname = 'ts_gamma_1_1.2651_gamma_2_2.0386_s_1_0.3511_s_2_0.3593.npy'
true_values = fname.split('_')
true_values = np.array([true_values[3], true_values[6], \
                        true_values[9], true_values[12][:-4]])
true_values = true_values.astype(float)

ts = np.load('../synthetic_data/sim_ts_coupled/' + fname, allow_pickle=True)
marks = np.load('../synthetic_data/sim_marks_coupled/' + 'marks' + fname[2:], allow_pickle=True)

# separate a set of K pipes into clusters,
# where a cluster is a set of adjacent coupled pipes
# e.g. for a set of pipes labelled [1,2,3,4],
# with phi = [1,0,1], implies [1,2] and [3,4] are coupled clusters.
def get_clusters(phi):
    K = len(phi)+1
    pipes = range(1,K+1)
    n_clusters = K - np.count_nonzero(phi)
    C_last=0
    clusters = []
    for i in range(len(phi)):
        if (i == len(phi)-1):
            if (phi[i]==1):
                phi_cluster = phi[C_last:i+1]
                cluster = range(1+C_last, 2+C_last+len(phi_cluster))
                clusters.append(cluster)
            elif (phi[i]==0):
                phi_cluster = phi[C_last:i+1]
                cluster = range(1+C_last, 1+C_last+len(phi_cluster))
                clusters.append(cluster)
                clusters.append([pipes[-1]])

        elif (phi[i]==0):
            phi_cluster = phi[C_last:i+1]
            cluster = range(1+C_last, 1+C_last+len(phi_cluster))
            C_last = i+1
            clusters.append(cluster)
    return clusters

# probability density function of inter-event times
# for a single, uncoupled pipe
def pdf_analytical_trunc(t, gamma, s):
    val = 2 * gamma/(s*np.sqrt(2*np.pi)) \
          * np.exp(-((gamma*t-1)/(s*np.sqrt(2)))**2) \
          / (1 + erf(1/(s*np.sqrt(2))))
    return val

# cumulative density function of inter-event times
# for a single, uncoupled pipe
def cdf_analytical_trunc(t, gamma, s):
    val = (erf((gamma*t - 1)/(s*np.sqrt(2))) + erf(1/(s*np.sqrt(2)))) \
        / (1 + erf(1/(s*np.sqrt(2))))
    return val

# calculate log likelihood for a set of pipes
def log_likelihood(theta):
    gammas, ss, phi = theta
    LL = 0
    clusters = get_clusters(phi)
    
    for i in range(len(clusters)):
        if (len(clusters[i])>1):
            LL += coupled_log_likelihood(gammas, ss, clusters[i])
        else:
            LL += single_log_likelihood(gammas, ss, clusters[i])
    return LL

# calculate log likelihood for a set of uncoupled pipes
def single_log_likelihood(gammas, ss, cluster):
    idx = cluster[0]-1
    ts_c = ts[marks==cluster[0]]
    dts_c = ts_c[1:]-ts_c[:-1]
    
    # calculate probability of pipe with mark=idx
    P_k = float(gammas[idx])/sum(gammas)
    LL=0
    for i in range(len(dts_c)):
        LL += np.log(pdf_analytical_trunc(dts_c[i], gammas[idx], ss[idx]) * P_k)
    return LL

# calculate log likelihood for a set of coupled pipes
def coupled_log_likelihood(gammas, ss, cluster):
    ts_c, marks_c = [],[]
    for i in range(len(ts)):
        if (marks[i] in cluster):
            ts_c.append(ts[i])
            marks_c.append(marks[i])
    ts_c = np.array(ts_c)
    marks_c = np.array(marks_c)
    
    dts_c = ts_c[1:]-ts_c[:-1]
    c0 = cluster[0]-1
    c1 = cluster[-1]-1
    LL=0
    for i in range(len(dts_c)):
        LL += np.log(pdf_joint(dts_c[i], gammas, ss, marks_c[i+1]))
    return LL

# calculate probability density from joint distribution of couple pipes
def pdf_joint(t, gammas, ss, mark):
    idx = int(mark)-1
    val = pdf_analytical_trunc(t, gammas[idx], ss[idx])
    for i in range(len(gammas)):
        if (i != idx):
            val *= (1-cdf_analytical_trunc(t, gammas[i], ss[i]))
    return val 

# calculate probability density of a sample
# from a truncated normal distribution
def cond_pd(x, low, upp, theta, sd):
    return truncnorm.pdf(x, (low-theta)/sd, (upp-theta)/sd, loc=theta, scale=sd)

# function for taking a random sample from a truncated
# normal distribution
def truncnorm_sample(low, upp, mean, sd):
    return truncnorm.rvs((low-mean)/sd, (upp-mean)/sd, loc=mean, scale=sd)

# sampling step-size adaptation, taken from PyMC3
def tune(scale, acc_rate):
    """
    Tunes the scaling parameter for the proposal distribution
    according to the acceptance rate over the last tune_interval:
    Rate    Variance adaptation
    ----    -------------------
    <0.001        x 0.1
    <0.05         x 0.5
    <0.2          x 0.9
    >0.5          x 1.1
    >0.75         x 2
    >0.95         x 10
    """
    
    return scale * np.where(
        acc_rate < 0.001,
        # reduce by 90 percent
        0.1,
        np.where(
            acc_rate < 0.05,
            # reduce by 50 percent
            0.5,
            np.where(
                acc_rate < 0.2,
                # reduce by ten percent
                0.9,
                np.where(
                    acc_rate > 0.95,
                    # increase by factor of ten
                    10.0,
                    np.where(
                        acc_rate > 0.75,
                        # increase by double
                        2.0,
                        np.where(
                            acc_rate > 0.5,
                            # increase by ten percent
                            1.1,
                            # Do not change
                            1.0,
                        ),
                    ),
                ),
            ),
        ),
    )

# function for performing Metropolis-Hastings
# algorithm to sample from posterior distribution
def MetHast(n=100):
    K = 2
    params = 5
    thetas = np.zeros((n, params))
    accept = np.zeros(n)
    LL = np.zeros(n)
    theta = prior
    thetas[0] = theta
    accept[0] = 1.
    new_walk = walk
    accept_ratio = 0
    # draw n samples
    for i in range(1,n):
        # generate candidate from random walk
        theta_cand = np.zeros(params)

        # phi is sampled from a Bernoulli distribution
        theta_cand[-1] = bernoulli.rvs(0.5)

        # the rest are sampled from truncated normal distributions
        for j in range(params-1):
            theta_cand[j] = truncnorm_sample(lows[j], upps[j], theta[j], new_walk[j])

        if (i%50 == 0 and i>=50):
            print("ITERATION %d\n\nacceptance rate = %f\nlog likelihood  = %f\n" % (i, accept_ratio, LL_t))
            print("parameter = [gamma_1,  gamma_2,  s_1,      s_2,      phi]")
            print("original  = [%f, %f, %f, %f, %d]" % tuple(theta))
            print("candidate = [%f, %f, %f, %f, %d]\n" % tuple(theta_cand))
            print("-----------------------------------------------------------------\n")
            
            # calculate acceptance rate
            accept_ratio = np.count_nonzero(accept[max(0,i-1000):i])/float(len(accept[-1000:]))

            # adjust random walk step-size according to acceptance rate
            new_walk *= tune(1, accept_ratio)

        # calculate prior probabilities
        prior_cand, prior_t = 1., 1.
        for j in range(params-1):
            prior_cand *= cond_pd(theta_cand[j], lows[j], upps[j], prior[j], sds[j])
            prior_t *= cond_pd(theta[j], lows[j], upps[j], prior[j], sds[j])

        # calculate log likelihoods of previous sample and candidate sample
        LL_cand = log_likelihood((theta_cand[:K], theta_cand[K:2*K], theta_cand[2*K:]))
        LL_t = log_likelihood((theta[:K], theta[K:2*K], theta[2*K:]))

        # calculate acceptance ratio
        alpha = np.exp(LL_cand-LL_t)*prior_cand/prior_t
        
        # correct for the asymmetry of proposal function
        for j in range(params-1):
            alpha /= cond_pd(theta_cand[j], lows[j], upps[j], theta[j], sds[j])
            alpha *= cond_pd(theta[j], lows[j], upps[j], theta_cand[j], sds[j])

        # generate uniform random number between 0 and 1
        u = np.random.uniform(low=0.0, high=1.0)
        if (u <= alpha): # accept
            thetas[i] = theta_cand
            theta = theta_cand
            accept[i] = 1.
            LL[i] = LL_cand
        else: # reject
            thetas[i] = theta
            LL[i] = LL_t
    return thetas, accept, LL


dts = ts[1:]-ts[:-1]
mks = marks[1:]
dt1 = dts[mks==1]
dt2 = dts[mks==2]

# prior expected value of each parameter
prior = [1/np.median(dt1),  \
         1/np.median(dt2), \
         np.std(dt1)*np.median(dt1), \
         np.std(dt2)*np.median(dt2), \
         0.]

# prior standard deviation of each parameter
sds = [1., \
       1., \
       0.1, \
       0.1, \
       0.]

# perform inference using ~uniform prior distributions
# instead of normal prior distributions
# (comment out the following line for the latter option)
sds = np.array(sds)*1000.

# upper limit of truncation
upps = [100., \
        100., \
        0.8, \
        0.8, \
        1.]

# lower limit of truncation
lows = [0., \
        0., \
        0., \
        0., \
        0.]

# initial random walk step-size
walk = np.array(sds)/10.
           
if __name__ == '__main__':
    n = 51000 # total number of samples
    K = 2 # number of pipes
    params = 2*K + (K-1) # number of parameters
    burn = 1000 # burn-in duration
    itv = 100 # interval to remove autocorrelation
    thetas, accept, LL = MetHast(n)
    thetas_filt = thetas[burn::itv]
