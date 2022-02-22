import numpy as np
from matplotlib import pyplot as plt
import scipy
from sklearn import mixture
#from IPython import embed
import corner


alpha_vec_1_2_sigma = np.array([scipy.stats.norm.cdf(1.0) - scipy.stats.norm.cdf(-1.0), scipy.stats.norm.cdf(2.0) - scipy.stats.norm.cdf(-2.0)])


def inference_test_plot(alpha_vec, prob, prob_lo, prob_hi, title=None,savename='convergence_test.pdf'):

    plt.plot(alpha_vec, prob, color='black', linestyle='solid', label='inference test points', zorder=10)
    plt.fill_between(alpha_vec, prob_lo, prob_hi, facecolor='grey', alpha=0.8, zorder=3)
    x_vec = np.linspace(0.0, 1.0, 11)
    plt.plot(x_vec, x_vec, linewidth=1.5, color='red', linestyle=(0,(5,10)), zorder=20, label='inferred model')
    plt.xlim((-0.01, 1.01))
    plt.ylim((-0.01, 1.01))
    plt.xlabel(r'$P_{{\rm true}}$', fontsize=16)
    plt.ylabel(r'$P_{{\rm inf}}$', fontsize=16)
    plt.title(title)
    plt.savefig(savename)
    plt.close()

def C_ge(nsamp):

    return (np.arange(nsamp)[::-1] + 1)/nsamp



def calc_percentiles(binary_inference_test_results):
    """
    Given the results of inference_test on a set of models, compute the binomial errors
    Args:
        inf_test_full (ndarray):
            binary vector of inference test results.  1 if the true model is within the contour set, 0 otherwise.
    Returns:
        probability (float):
        lower_error (float):
        upper_error (float):
    """
    num_tests = binary_inference_test_results.shape[0]

    # number of tests that passed
    n_test_full = np.sum(binary_inference_test_results, axis=0)
    # fraction of tests that passed
    probability = n_test_full / num_tests

    lower_error = (
        scipy.stats.binom.isf(scipy.stats.norm.cdf(1.0), num_tests, probability) / num_tests
    )
    upper_error = (
        scipy.stats.binom.isf(scipy.stats.norm.cdf(-1.0), num_tests, probability) / num_tests
    )

    return probability, lower_error, upper_error



def get_C_ge_x(x):
    """
    Returns cumulative probability C(>= x) of a set of samples from a distribution x
    Args:
        x:
    Returns:
    """
    nsamp = x.shape[0]

    C_ge_x = C_ge(nsamp)
    x_sort = x[x.argsort()]

    return x_sort, C_ge_x


def inference_test(lnP, lnP_true, alpha_vec=None, C_ge_lnP_true=None):
    """
    Given probabilities lnP evaluated at samples from a Markov chain and lnP_true, the lnP evaluated at the true
    model, perform an inference test to see if the true model is within the contours specified by the probabilities
    in alpha_vec
    Args:
        lnP (ndarray):
            Probabilities evaluated at samples from a Markov chain, shape = (nsamp,)
        lnP_true (float):
            Probability evaluated at the location of the true model
        alpha_vec (float or ndarray):
           Probability threhsolds for infernence test.  Optional, defaults to give the 68% and 95% contour levels.
        C_ge_lnP_true(ndarray):
            Cumlative probability lnP rank relationship determined from importance weights, shape = (nsamp,). Optional. If not set
            a linear relation will be used as returned by  C_ge(nsamp), which assumes uniform weighting of the mcmc samples
            to generate the final posterior.
    Returns:
        passed (bool ndarray):
           Boolean array which is true if the true model is within the contour set.  shape = (nalpha,)
        alpha_vec (ndarray):
            The values of alpha that were used.
    """

    _alpha_vec = alpha_vec_1_2_sigma if alpha_vec is None else np.atleast_1d(alpha_vec)
    nsamp = lnP.shape[0]
    C_ge_lnP = C_ge(nsamp) if C_ge_lnP_true is None else C_ge_lnP_true

    lnP_sort = np.sort(lnP)

    isort = C_ge_lnP.argsort()
    lnP_alpha = np.interp(_alpha_vec, C_ge_lnP[isort], lnP_sort[isort])
    #lnP_alpha = interpolate.interp1d(C_gt_lnP_cov, lnP_sort, bounds_error=False, kind='cubic', fill_value='extrapolate', assume_sorted=False)(_alpha_vec)
    return lnP_true >= lnP_alpha, _alpha_vec
    # TODO:
    # Should this be > or >=? It's a bit subtle and definitional

def print_inference_report(nmock, alpha_vec, coverage, coverage_lo, coverage_hi):

    print('-------------------------------------------------------------------------')
    print('Full Inference test results for nqsos={:d}'.format(nmock))
    print('-------------------------------------------------------------------------')
    for cov, cov_lo, cov_hi, alpha in zip(coverage, coverage_lo, coverage_hi, alpha_vec):
        print('Full    : {:3.1f} + {:3.1f} - {:3.1f} % for alpha={:5.3f} %'.format(
            100.0 * cov, 100.0 * (cov - cov_lo), 100.0 * (cov_hi - cov), 100.0 * alpha))



def run_inference_test(lnP_mock, lnP_true, alpha_vec, C_ge_lnP_true=None, show=True, verbose=True, title=None,savename='convergence.pdf'):
    """
    Args:
        lnL_mock (ndarray):
            Log-Probability evaluated at every sample in a Markov Chain for a set of mock inference realizations.
            Shape = (nmock, nmcmc)
        lnL_true (ndarray:
            Log-Probability evaluate at the true model that underlies each mock. Shape = (nmock,)
        alpha_vec (ndarray):
           Probability threhsolds for infernence test. This is the x-axis of the coverage plot. Shape = (nalpha,)
        C_ge_lnP_true(ndarray):
            Cumlative probability lnP rank relationship determined from importance weights, shape = (nsamp,). Optional. If not set
            a linear relation will be used as returned by  C_ge(nsamp), which assumes uniform weighting of the mcmc samples
            to generate the final posterior.
        show (bool):
            Make and show the coverage plot (coverage vs alpha_vec). Optional, default=True
        verbose (bool):
            Print the coverage report to the screen. Optional, default = True
        title (str):
            Title of the plot if showing the plot is requested.
    Returns:
        coverage, coverage_lo, coverage_hi
        coverage (ndarray):
            Coverage of the inference at each probability threshold in alpha_vec. shape = (nalpha,)
        coverage_lo (ndarray):
            Lower 1-sigma error bar based on the binomial distribution. shape = (nalpha,)
        coverage_hi (ndarray):
            Upper 1-sigma error bar based on the binmoial distribution. shape = (nalpha,)
    """

    nmock, nmcmc = lnP_mock.shape
    # Full inference test
    for imock in range(nmock):
        # Full inference test
        inf_test_full0, _ = inference_test(lnP_mock[imock, :], lnP_true[imock], C_ge_lnP_true=C_ge_lnP_true,
                                           alpha_vec=alpha_vec)
        if imock == 0:
            inf_test_full = np.zeros((nmock,) + inf_test_full0.shape, dtype=bool)

        inf_test_full[imock, :] = inf_test_full0

    coverage, coverage_lo, coverage_hi = calc_percentiles(inf_test_full)

    if verbose:
        print_inference_report(nmock, alpha_vec, coverage, coverage_lo, coverage_hi)

    # Make a plot of the Full inference test results
    if show:
        inference_test_plot(alpha_vec, coverage, coverage_lo, coverage_hi, title=title,savename=savename)

    return coverage, coverage_lo, coverage_hi


def run_inference_test_GMM(samples, truths, ngauss, alpha_vec, nGM_samp=10000, seed_or_rng=None, C_ge_lnP_true=None,
                           show=True, verbose=True, title=None):
    """
    Args:
        samples (ndarray):
            MCMC samples from the posterior distribution, shape = (nmock, nmcmc, nparams)
        truths (ndarray):
            The true value of the model parameters that gave rise to the posterior samples, shape =(nmock, nparams,)
        ngauss (int):
            Then nubmer of Gaussian components used to fit the gaussianize mixture model (GMM) to the samples
        alpha_vec (ndarray):
           Probability threhsolds for infernence test. This is the x-axis of the coverage plot. Shape = (nalpha,)
        nGM_samp (int):
            Number of samples from the GMM to use for performing the inference test, optional, default = 10000
        seed_or_rng (inr or np.random.generator):
            Seed or random number generator
        C_ge_lnP_true(ndarray):
            NOT YET IMPLEMENTED
            Cumlative probability lnP rank relationship determined from importance weights, shape = (nsamp,). Optional. If not set
            a linear relation will be used as returned by  C_ge(nsamp), which assumes uniform weighting of the mcmc samples
            to generate the final posterior.
        show (bool):
            Make and show the coverage plot (coverage vs alpha_vec). Optional, default=True
        verbose (bool):
            Print the coverage report to the screen. Optional, default = True
        title (str):
            Title of the plot if showing the plot is requested.
    Returns:
        coverage, coverage_lo, coverage_hi
        coverage (ndarray):
            Coverage of the inference at each probability threshold in alpha_vec. shape = (nalpha,)
        coverage_lo (ndarray):
            Lower 1-sigma error bar based on the binomial distribution. shape = (nalpha,)
        coverage_hi (ndarray):
            Upper 1-sigma error bar based on the binmoial distribution. shape = (nalpha,)
    """

    nmock, nmcmc, nparams = samples.shape
    # Full inference test
    for imock in range(nmock):
        # Full inference test
        inf_test_marg0, _ = inference_test_GMM(samples[imock, :, :], truths[imock, :],
                                               ngauss, alpha_vec=alpha_vec, nGM_samp=nGM_samp,
                                               seed_or_rng=seed_or_rng)
        if imock == 0:
            inf_test_marg = np.zeros((nmock,) + inf_test_marg.shape, dtype=bool)

        inf_test_marg[imock, :] = inf_test_marg0

    coverage, coverage_lo, coverage_hi = calc_percentiles(inf_test_marg)

    if verbose:
        print_inference_report(nmock, alpha_vec, coverage, coverage_lo, coverage_hi)

    # Make a plot of the Full inference test results
    if show:
        inference_test_plot(alpha_vec, coverage, coverage_lo, coverage_hi, title=title)

    return coverage, coverage_lo, coverage_hi


def inference_test_GMM(samples, true_value, ngauss, nGM_samp=10000, seed_or_rng=None, alpha_vec=None,
                       debug1d=False, debug2d=False):
    """
    Given Markov chain samples from a posterior distribution test whether the true_value is within the contour levels
    set by alpha_vec
    Args:
        samples (ndarray):
            MCMC samples from the posterior distribution, shape = (nsamples, nparams)
        true_value (ndarray):
            The true value of the model parameters that gave rise to the posterior samples, shape =(nparams,) or (1,nparams)
        ngauss (int):
            Then nubmer of Gaussian components used to fit the gaussianize mixture model (GMM) to the samples
        nGM_samp (int):
            Number of samples from the GMM to use for performing the inference test, optional, default = 10000
        seed_or_rng (inr or np.random.generator):
            Seed or random number generator
        alpha_vec (float or ndarray):
            Probability threhsolds for infernence test.  Defaults to give the 68% and 95% contour levels.
    Returns:
    """

    _alpha_vec = alpha_vec_1_2_sigma if alpha_vec is None else np.atleast_1d(alpha_vec)

    nsamples, nparams = samples.shape
    rng = np.random.default_rng(seed_or_rng)
    sk_learn_ran_state = np.random.RandomState(rng.integers(2 ** 32))
    GMM = mixture.GaussianMixture(n_components=ngauss, random_state=sk_learn_ran_state).fit(samples)
    samples_GMM, _ = GMM.sample(nGM_samp)
    lnP_true = GMM.score_samples(np.atleast_2d(true_value))
    lnP_samp = GMM.score_samples(samples_GMM)

    inf_test_marg, alpha_vec_out = inference_test(lnP_samp, lnP_true, alpha_vec=alpha_vec)

    if debug2d:
        # Debugging of Gaussian mixture by making a cornder plot.
        # TODO: Note that the histograms are not normalized correctly in the corner plot, but they are fine in 1d below
        fig = corner.corner(samples, truths=true_value, levels=(0.68,0.95),color='k', truth_color='red', show_titles=True,
                            title_kwargs={"fontsize": 15}, label_kwargs={'fontsize': 20},
                            data_kwargs={'ms': 1.0, 'alpha': 0.1})
        corner.corner(samples_GMM, fig=fig, color='g', weights=np.ones(len(samples_GMM))*len(samples)/len(samples_GMM))
        plt.show()

    if debug1d:
        for ii in range(nparams):
            plt.hist(samples[:, ii], bins=40, alpha=0.5, density=True, label='input samples')
            plt.hist(samples_GMM[:, ii], bins=40, alpha=0.5, density=True, label='GMM samples')
            plt.legend()
            plt.show()

    return inf_test_marg, alpha_vec_out


def compute_importance_weights(nmcmc, alpha_vec, coverage, show=True):
    """
    Compute importance sampling weights from a coverage plot to reweight posterior samples to deliver
    the correct posterior.
    Args:
        nmcmc (int):
           Number of samples in the Markov chain
        alpha_vec (ndarray):
           Probability threhsolds for infernence test. This is the x-axis of the coverage plot. Shape = (nalpha,)
        coverage:
        show:
    Returns:
        importance_weights (ndarray):
           Importance weights to synthesize the correct posterior from Markov chain samples from the approximate
           posterior. The weights are assigned to the samples according to the lnP rank, and as such the array
           returned here is the weight to assign to each sample in increasing order of lnP
        C_ge_lnP_true(ndarray):
           Cumlative probability lnP rank relationship determined from importance weights, shape = (nmcmc,). Optional.
        n_eff (float):
            Effective number of Markov chain samples after reweighting. This is the Kish effective sample size defined by
                   n_eff \equiv (\Sum_i w_i)^2/\Sum_i w_i^2
            See https://en.wikipedia.org/wiki/Effective_sample_size for more details
        show (bool):
            Make two diagnostic plots, namely the "Cumulative Probability - Probability Density Rank Relation" and a
            plot showing the importance weights as a function of lnP rank. Optional, default=True
    """

    # C_ge_lnP_approx is the cumulative distribution function as a funch of lnP rank
    C_ge_lnP_approx = C_ge(nmcmc)
    # Interpolate the coverage results (coverage vs alpha_vec) onto this cumulative distribution to remap
    # (approximate) posterior probabilities (alpha_vec) to true probabilities
    C_ge_lnP_true = np.interp(C_ge_lnP_approx, alpha_vec, coverage)

    # Trim the zeros since these will cause problems (i.e. zero weight or negative weight)
    unit_normalized_rank = np.log10((np.arange(nmcmc) + 1) / nmcmc)

    print('Number of unique C_cov(>= lnP) before fix = {:d}'.format(np.unique(C_ge_lnP_true).shape[0]))
    itrim = C_ge_lnP_true > 0.0
    C_unique, indx = np.unique(C_ge_lnP_true[itrim], return_index=True)
    unit_norm_rank_unique = unit_normalized_rank[itrim][indx]
    isort = np.argsort(unit_norm_rank_unique)
    unit_norm_rank_unique = unit_norm_rank_unique[isort]
    C_unique = C_unique[isort]
    C_interp = np.interp(unit_normalized_rank, unit_norm_rank_unique, C_unique)
    print('Number of unique C_cov(>= lnP) after interpolating over non-unique values = {:d}'.format(C_interp.shape[0]))

    # Solve the linear system for the importance weights
    A_matrix = np.triu(np.ones((nmcmc, nmcmc)))
    importance_weights = np.abs(scipy.linalg.solve_triangular(A_matrix, C_interp))

    C_ge_lnP_true_from_weights = np.cumsum(importance_weights[::-1])[::-1]
    n_eff = np.square(np.sum(importance_weights)) / np.sum(np.square(importance_weights))

    if show:
        # Show the new Cum Prob - Rank relation
        ms = 1.5
        plt.title('Cumulative Probability - Probability Density Rank Relation')
        plt.plot(unit_normalized_rank, C_ge_lnP_approx, color='k', marker='o', ms=ms, linestyle="", label='Original')
        plt.plot(unit_normalized_rank, C_ge_lnP_true, color='blue', marker='o', ms=ms, alpha=0.5, linestyle="",
                 label='Inference Test')
        plt.plot(unit_normalized_rank, C_ge_lnP_true_from_weights, color='orange', marker='o', ms=ms, alpha=0.5,
                 linestyle="", label='Recovered from weights')
        plt.legend()
        plt.xlabel('ln(unit normalized lnP rank)')
        plt.ylabel('C(>= lnP)')
        plt.show()

        # Show the importance weights
        plt.title('Importance Weights, n_eff={:5.3f}'.format(n_eff))
        plt.plot(unit_normalized_rank, importance_weights, 'k.')
        # plt.xscale('log')
        plt.xlabel('ln(unit normalized lnP rank)')
        plt.ylabel('Importance Weights')
        plt.show()

    return importance_weights, C_ge_lnP_true_from_weights, n_eff

def assign_importance_weights(lnP_chain, importance_weights):
    """
    Importance weights are determined from a global coverage analysis of a large ensemble of mocks. This routine
    will assign this set of weights to Markov chain samples from an approximate posterior
    (i.e. that didn't pass a coverage test). The resulting weighted posterior will then pass a coverage test.  Since
    importance weights determined from compute_importance_weights above) are ordered by the lnP rank, this routine
    simply sorts the lnP for the samples and then assigns weights to the chain based on these sorted lnP values.
    Args:
        lnP_chain (ndarray):
            The lnP values associated with each sample from a Markov chain, shape = (nmcmc,)
        importance_weights (ndarray):
            Importance weights determined by the routine compute_importance_weights above. Note that this
            array is ordered by lnP rank, shape =(nmcmc,)
    Returns:
        importance_weights_chain (ndarray):
            The importance weights to be used for each sample in the chain based on its lnP value, shape = (nmcmc,)
    """

    nmcmc = lnP_chain.shape[0]
    isort = np.argsort(lnP_chain)
    importance_weights_chain = np.zeros(nmcmc)
    importance_weights_chain[isort] = importance_weights

    return importance_weights_chain

#
#
#
# def inference_test_old(lnP, lnP_true, alpha_vec=None):
#     """
#     Given probabilities lnP evaluated at samples from a Markov chain and lnP_true, the lnP evaluated at the true
#     model, perform an inference test to see if the true model is within the contours specified by the probabilities
#     in alpha_vec
#
#     Args:
#         lnP (ndarray):
#             Probabilities evaluated at samples from a Markov chain, shape = (nchain,)
#
#         lnP_true (float):
#             Probability evaluated at the location of the true model
#
#         alpha_vec (float or ndarray):
#            Probability threhsolds for infernence test.  Defaults to give the 68% and 95% contour levels.
#
#     Returns:
#         passed (bool ndarray):
#            Boolean array which is true if the true model is within the contour set.  shape = (nalpha,)
#
#         alpha_vec (ndarray):
#             The values of alpha that were used.
#
#     """
#
#     _alpha_vec = alpha_vec_1_2_sigma if alpha_vec is None else np.atleast_1d(alpha_vec)
#     lnP_sort, C_ge_lnP = get_C_ge_x(lnP)
#
#     #lnP_alpha = np.interp(_alpha_vec, C_gt_lnP[::-1], lnP_sort[::-1])
#     lnP_alpha = interpolate.interp1d(C_ge_lnP, lnP_sort, bounds_error=False, kind='cubic', fill_value='extrapolate', assume_sorted=False)(_alpha_vec)
#     return lnP_true >= lnP_alpha, _alpha_vec