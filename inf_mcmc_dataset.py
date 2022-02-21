import os
import numpy as np
import h5py
from IPython import embed
from scipy.interpolate import griddata

rseed = 87654
rand = np.random.RandomState(rseed)

nsteps_burn = 500
nsteps_main = 4000
nwalkers = 40
ndim = 2

Npargrid = 100
version = 'mcmc_2D_Nyx_v3.0_sed' + str(rseed) + '_walker' + str(nwalkers) + '_step' + str(nsteps_main)
N_dataset = 1
data_dz = N_dataset * 2.136

ILLTNG_theta = [3.609, 1.53]
oldILL_theta = [3.63, 1.56]
Nyx_theta = [3.60, 1.59]
true_theta = Nyx_theta


chain_arr0=[]
logprob_arr0=[]
l_arr=[]

Nrun=10
for i_run in range(0,Nrun):

    filename = '2D' + str(nsteps_main) + 'steps' + str(nwalkers) + 'walkers_T0_' + str(
        true_theta[0]) + '_gamma_' + str(true_theta[1]) + '_' + str(i_run) + 'posterior_' + version + '.hdf5'

    with h5py.File(filename, 'r') as f:
        chain = f['chain'].value
        log_prob = f['log_prob'].value

        chain_l = len(chain[:, 0, 0])
        print(str(i_run), "th run finish restoring mcmc posterior")

    l_arr.append(chain_l)
    chain_arr0.append(chain)
    logprob_arr0.append(log_prob)


lmin = np.min(l_arr)

chain_arr = []
logprob_arr = []
logPtrue_arr = []
Ptrue_arr = []

for i_run in range(0,Nrun):
    flatchain = chain_arr0[i_run][-1*lmin:].flatten().reshape(lmin*nwalkers,ndim)
    flatprob = logprob_arr0[i_run][-1*lmin:].flatten()

    print("calculate L(P_true)")
    L_Ptrue = griddata(flatchain, flatprob, true_theta, method='nearest')

    chain_arr.append(flatchain)
    logprob_arr.append(flatprob)
    logPtrue_arr.append(L_Ptrue)
    Ptrue_arr.append(true_theta)


filedataset = '2D_' + str(Nrun) + 'run' + str(nwalkers) + 'walkers_'+ version + 'dataset.hdf5'

with h5py.File(filedataset, 'w') as f2:
    f2.create_dataset('Nrun', data=Nrun)
    f2.create_dataset('mcmc_nsteps_tot', data=lmin)
    f2.create_dataset('ln_probs', data=logprob_arr)
    f2.create_dataset('samples', data=chain_arr)
    f2.create_dataset('ln_prob_true', data=logPtrue_arr)
    f2.create_dataset('theta_true', data=Ptrue_arr)

embed()
