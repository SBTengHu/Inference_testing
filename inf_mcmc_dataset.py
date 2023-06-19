import os
import numpy as np
import h5py
from IPython import embed
import glob
from scipy.interpolate import griddata

rseed = 12345
rand = np.random.RandomState(rseed)

nsteps_burn = 500
nsteps_main = 3000
nwalkers = 30
ndim = 3

Npargrid = 100
version = 'Nyx_3D_paraTP_fixGUV_v1.0_sed' + str(rseed) + '_walker' + str(nwalkers) + '_step' + str(nsteps_main)
N_dataset = 1
data_dz = N_dataset * 2.136

#optimized
ILLTNG_theta = [1.18, 0.03, -13.338]
oldILL_theta = [1.69, -0.35, -13.378]
#Nyx_theta =    [1.0, 0.0, -13.308]

#fit
#ILLTNG_theta = [3.61, 1.52, -13.30]
#oldILL_theta = [3.63, 1.56, -13.30]
Nyx_theta =   [1.0, 0.0, -13.20]

#test
#ILLTNG_theta = [3.61, 1.60, -13.35]
true_theta = Nyx_theta


chain_arr0=[]
logprob_arr0=[]
LPtrue_arr=[]
Ptrue_arr=[]
l_arr=[]

filenames = sorted([os.path.basename(x) for x in glob.glob('2D3000steps30walkers_*posterior_mcmc_goodness_TP_v1_sed87654_walker30_step3000.hdf5')])
Nrun=len(filenames)
for i_run in range(0,Nrun):

    filename = filenames[i_run]
    with h5py.File(filename, 'r') as f:
        chain = f['chain'].value
        log_prob = f['log_prob'].value
        L_Ptrue = f['LP_true'].value
        Ptrue = f['P_true'].value

        chain_l = len(chain[:, 0, 0])
        print(str(i_run), "th run finish restoring mcmc posterior")

    l_arr.append(chain_l)
    chain_arr0.append(chain)
    logprob_arr0.append(log_prob)
    LPtrue_arr.append(L_Ptrue)
    Ptrue_arr.append(Ptrue)

lmin = np.min(l_arr)
chain_arr = []
logprob_arr = []

for i_run in range(0,Nrun):
    flatchain = chain_arr0[i_run][-1*lmin:].flatten().reshape(lmin*nwalkers,ndim)
    flatprob = logprob_arr0[i_run][-1*lmin:].flatten()

    #print("calculate L(P_true)")
    #L_Ptrue = griddata(flatchain, flatprob, true_theta, method='nearest')
    #LPtrue_arr.append(L_Ptrue)

    chain_arr.append(flatchain)
    logprob_arr.append(flatprob)
    #Ptrue_arr.append(true_theta)


filedataset = '3D_' + str(Nrun) + 'run'+ version + 'dataset.hdf5'

with h5py.File(filedataset, 'w') as f2:
    f2.create_dataset('Nrun', data=Nrun)
    f2.create_dataset('mcmc_nsteps_tot', data=lmin)
    f2.create_dataset('ln_probs', data=logprob_arr)
    f2.create_dataset('samples', data=chain_arr)
    f2.create_dataset('ln_prob_true', data=LPtrue_arr)
    f2.create_dataset('theta_true', data=Ptrue_arr)

