import os
import numpy as np
import h5py
from IPython import embed
from scipy.interpolate import griddata

rseed = 12345
rand = np.random.RandomState(rseed)

nsteps_burn = 500
nsteps_main = 4000
nwalkers = 40
ndim = 3

Npargrid = 100
version = 'Nyx_3D_v1.0_sed' + str(rseed) + '_walker' + str(nwalkers) + '_step' + str(nsteps_main)
N_dataset = 1
data_dz = N_dataset * 2.136

#ILLTNG_theta = [3.609, 1.53]
#oldILL_theta = [3.63, 1.56]
#Nyx_theta = [3.60, 1.59]

ILLTNG_theta = [3.609, 1.53, -13.28]
oldILL_theta = [3.63, 1.56, -13.67]
Nyx_theta = [3.60, 1.59, -13.308]
true_theta = Nyx_theta

chain_arr0=[]
logprob_arr0=[]
LPtrue_arr=[]
l_arr=[]

Nrun=10
for i_run in range(0,Nrun):

    filename = 'dz'+str(data_dz)+'_MCMCfile_mcmctest_'+version + '_' + str(i_run) + '.hdf5'

    with h5py.File(filename, 'r') as f:
        chain = f['chain'].value
        log_prob = f['log_prob'].value
        L_Ptrue = f['LP_true'].value

        chain_l = len(chain[:, 0, 0])
        print(str(i_run), "th run finish restoring mcmc posterior")

    l_arr.append(chain_l)
    chain_arr0.append(chain)
    logprob_arr0.append(log_prob)
    LPtrue_arr.append(L_Ptrue)


lmin = np.min(l_arr)

chain_arr = []
logprob_arr = []
Ptrue_arr = []

for i_run in range(0,Nrun):
    flatchain = chain_arr0[i_run][-1*lmin:].flatten().reshape(lmin*nwalkers,ndim)
    flatprob = logprob_arr0[i_run][-1*lmin:].flatten()

    #print("calculate L(P_true)")
    #L_Ptrue = griddata(flatchain, flatprob, true_theta, method='nearest')

    chain_arr.append(flatchain)
    logprob_arr.append(flatprob)
    Ptrue_arr.append(true_theta)


filedataset = '3D_' + str(Nrun) + 'run'+ version + 'dataset.hdf5'

with h5py.File(filedataset, 'w') as f2:
    f2.create_dataset('Nrun', data=Nrun)
    f2.create_dataset('mcmc_nsteps_tot', data=lmin)
    f2.create_dataset('ln_probs', data=logprob_arr)
    f2.create_dataset('samples', data=chain_arr)
    f2.create_dataset('ln_prob_true', data=LPtrue_arr)
    f2.create_dataset('theta_true', data=Ptrue_arr)

embed()