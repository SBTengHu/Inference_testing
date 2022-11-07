import os
import numpy as np
import h5py
from IPython import embed
from scipy.interpolate import griddata
import glob

rseed = 12345
rand = np.random.RandomState(rseed)

nsteps_burn = 500
nsteps_main = 3000
nwalkers = 30
ndim = 2

Npargrid = 100
version = 'oldILL_2D_TP_paraAB_v1_sed' + str(rseed) + '_walker' + str(nwalkers) + '_step' + str(nsteps_main)+'_'
N_dataset = 1
data_dz = N_dataset * 2.136

ILLTNG_theta = [1.0, 0.0]
#ILLTNG_theta = [3.609, 1.53]
oldILL_theta = [1.5, -0.25]
#Nyx_theta = [3.60, 1.59]
Nyx_theta = [1.0, 0.0]

#ILLTNG_theta = [3.62, 1.60, -13.28]
#oldILL_theta = [3.63, 1.58, -13.67]
#Nyx_theta = [3.65, 1.61, -13.308]
true_theta = oldILL_theta

chain_arr0=[]
logprob_arr0=[]
LPtrue_arr=[]
l_arr=[]

filenamelist = raw_skewers = sorted([os.path.basename(x) for x in glob.glob('*posterior_mcmc_2D_oldILL_TP_paraAB_v1_sed*.hdf5')])

Nrun=len(filenamelist)
for i_run in range(0,Nrun):

    filename = filenamelist[i_run]

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
#lmin = nsteps_main

chain_arr = []
logprob_arr = []
Ptrue_arr = []


for i_run in range(0,Nrun):
    flatchain = chain_arr0[i_run][-1*lmin:].flatten().reshape(lmin*nwalkers,ndim)
    flatprob = logprob_arr0[i_run][-1*lmin:].flatten()

    print("calculate L(P_true)")
    L_Ptrue = griddata(flatchain, flatprob, true_theta, method='nearest')

    chain_arr.append(flatchain)
    logprob_arr.append(flatprob)
    Ptrue_arr.append(true_theta)

filedataset = version +str(Nrun) + 'run_dataset.hdf5'

with h5py.File(filedataset, 'w') as f2:
    f2.create_dataset('Nrun', data=Nrun)
    f2.create_dataset('mcmc_nsteps_tot', data=lmin)
    f2.create_dataset('ln_probs', data=logprob_arr)
    f2.create_dataset('samples', data=chain_arr)
    f2.create_dataset('ln_prob_true', data=LPtrue_arr)
    f2.create_dataset('theta_true', data=Ptrue_arr)

embed()