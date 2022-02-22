
import os
import numpy as np
import h5py
from Inference_testing.inference_testing.inftest_mcmc_tool import run_inference_test
from Inference_testing.inference_testing.inftest_mcmc_tool import compute_importance_weights, assign_importance_weights
from qso_fitting.utils.get_paths import get_HI_DW_path
from qso_fitting.fitting.jax.dw_base import DampingWingBase
from IPython import embed


#from dw_inference.inference.utils import corner_plot



def Inference_test(mcmcfile, reweight=False, marginalize=False, astro_params_ngauss=8, seed_or_rng=None, cornerprefix=None,savename='convergence_test.pdf'):

    alpha_vec = np.concatenate((np.linspace(0.00, 0.994, num=100), np.linspace(0.995, 1.0, num=51)))

    #_alpha_vec = np.linspace(0.01,0.99,num=29) if alpha_vec is None else alpha_vec
    rng = np.random.default_rng(seed_or_rng) if seed_or_rng is not None else np.random.default_rng(42)

    mcmc_results = h5py.File(mcmcfile, 'r')

    Nrun = mcmc_results['Nrun'].value
    #nastro = mcmc_results['mcmc'].attrs['nastro']
    mcmc_nsteps_tot = mcmc_results['mcmc_nsteps_tot'].value

    lnProb =  np.array(mcmc_results['ln_probs'].value)           # lnProb has shape (nqsos, nchain)
    lnProb_true =  np.array(mcmc_results['ln_prob_true'].value)  # lnProb_true has shape (nqsos,)
    samples = np.array(mcmc_results['samples'].value)            # samples has shape (nqsos, nchain, nparams)
    theta_true = np.array(mcmc_results['theta_true'].value)      # theta_true has shape (nqsos, nparams)

    # Some code for the inference test on the marginalized distribution
   #theta_astro_samp = samples[:,:,0:nastro]
    #theta_astro_true = theta_true[:,0:nastro]

    coverage, coverage_lo, coverage_hi = run_inference_test(
        lnProb, lnProb_true, alpha_vec, title='Full Coverage', show=True, verbose=True,savename=savename)

    if reweight:
        importance_weights, C_ge_lnP_true, n_eff = compute_importance_weights(mcmc_nsteps_tot, alpha_vec, coverage, show=True)

        # Perform the weighted coverage test
        coverage_rw, coverage_rw_lo, coverage_rw_hi = run_inference_test(
            lnProb, lnProb_true, alpha_vec, C_ge_lnP_true=C_ge_lnP_true, title='Full Coverage Reweighted', show=True,
            verbose=True,savename='convergence_test_reweight.pdf')

        embed()
        # Show a reweighted posterior
        imock =  rng.choice(np.arange(Nrun))
        importance_weights_chain = assign_importance_weights(lnProb[imock,:], importance_weights)

        # Make an example corner plot
        # These are hacks for backward compatibility
        #try:
        #    dv_is_param = mcmc_results['mcmc'].attrs['dv_is_param']
        #except:
        #    dv_is_param = (mcmc_results['mcmc'].attrs['latent_dim'] + 1 != mcmc_results['mcmc'].attrs['ndim_qso'])
        #try:
        #    astro_only = mcmc_results['mcmc'].attrs['dv_is_param']
        #except:
        #    astro_only = mcmc_results['mcmc'].attrs['nastro'] == mcmc_results['mcmc'].attrs['ndim']
        #var_label = DampingWingBase.get_var_label(astro_only, mcmc_results['mcmc'].attrs['latent_dim'], dv_is_param)


        var_label = [r'$T_0$', r'$\gamma$']
        # Original corner plot
        corner_orig = os.path.join(cornerprefix, 'original_corner.pdf')
        corner_reweight = os.path.join(cornerprefix, 'reweighted_corner.pdf')

        #corner_plot(samples[imock, ...], var_label, theta_true=theta_true[imock, :], weights=importance_weights_chain,cornerfile=corner_reweight)
        #corner_plot(samples[imock, ...], var_label, theta_true=theta_true[imock, :], cornerfile=corner_orig)

    # Marginal inference test
    #if marginalize:
    #  coverage_marg, coverage_marg_lo, coverage_marg_hi = run_inference_test_GMM(
     #       theta_astro_samp, theta_astro_true, astro_params_ngauss, alpha_vec, seed_or_rng=rng,
    #      title='Marginal Coverage', show=True, verbose=True)


if __name__ == "__main__":

    #
    inf_path = '/mnt/quasar2/teng/NYX_v4/MCMC_2D'

    #mcmc_path = 'DW_sims_gauss_hand_fwhm_100_latent_10_seed_334455'
    #mcmc_path = 'DW_sims_hand_fwhm_100_latent_10_seed_334455'

    mcmc_outfile = os.path.join(inf_path, '2D_10run40walkers_mcmc_2D_Nyx_v3.0_sed87654_walker40_step4000dataset.hdf5')
    cornerprefix =  os.path.join(inf_path,'inftest_mcmc_figures')
    seed_or_rng = np.random.default_rng(7777)
    Inference_test(mcmc_outfile, reweight=False, cornerprefix=cornerprefix, seed_or_rng=seed_or_rng)