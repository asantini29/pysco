import sys, os
import numpy as np
import pysco
import matplotlib as mpl
from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt

from eryn.utils import get_integrated_act, psrf, Stopping


def get_clean_chain(coords, ndim, temp=0):
    """Simple utility function to extract the squeezed chains for all the parameters
    """
    naninds = np.logical_not(np.isnan(coords[:, temp, :, :, 0].flatten()))
    samples_in = np.zeros((coords[:, temp, :, :, 0].flatten()[naninds].shape[0], ndim))  # init the chains to plot
    # get the samples to plot
    for d in range(ndim):
        givenparam = coords[:, temp, :, :, d].flatten()
        samples_in[:, d] = givenparam[
            np.logical_not(np.isnan(givenparam))
        ]  # Discard the NaNs, each time they change the shape of the samples_in
    return samples_in

def adjust_covariance(samp, ndim, svd=False, idxs=1):
    """
    Adjusts the covariance matrix for each branch in the given sample.

    Parameters:
    - samp: The sample object containing the chains.
    - ndim: A dictionary containing the number of dimensions for each branch.
    - svd: A boolean indicating whether to perform singular value decomposition (SVD) on the covariance matrix. Default is False.
    - idxs: The indeces of the move in the sample object. Default is 1.

    Returns:
    None
    """
    discard = int(0.8 * samp.iteration)

    if not isinstance(idxs, list):
        idxs = [idxs]
    
    for idx in idxs:
        move = samp.moves[idx]
        branches = move.all_proposal.keys()

        for key in branches:
            item_samp = samp.get_chain(discard=discard)[key][:, 0][samp.get_inds(discard=discard)[key][:, 0]]
        
            cov = np.cov(item_samp, rowvar=False) * 2.38**2 / ndim[key]
            if svd:
                svd = np.linalg.svd(cov)
                move.all_proposal[key].svd = svd
            else:
                move.all_proposal[key].scale = cov
        
def plot_diagnostics(samp, path, ndim, truths, labels, transform_all, acceptance_all, true_logl=None, trace_color=None, acceptance_moves=None, rj_acceptance_all=None, rj_acceptance_moves=None, rj_branches=[], nleaves_min=None, nleaves_max=None, moves_names=None, rj_moves_names=None, use_chainconsumer=False, suffix='', **kwargs):
    
    """
    Plot various diagnostics for the given samples.

    Parameters:
    - samp: The samples object containing the chains.
    - path: The path to save the diagnostic plots.
    - ndim: A dictionary mapping the keys to the number of dimensions for each chain.
    - truths: A dictionary mapping the keys to the true values for each parameter.
    - labels: A dictionary mapping the keys to the labels for each parameter.
    - transform_all: A dictionary mapping the keys to the transformation functions to apply to the chains.
    - acceptance_all: The acceptance fraction for all moves.
    - trace_color: The color to use for the trace plots.
    - acceptance_moves: The acceptance fraction for each move.
    - rj_acceptance_all: The acceptance fraction for all RJ moves.
    - rj_acceptance_moves: The acceptance fraction for each RJ move.
    - rj_branches: The keys of the branches to plot RJ diagnostics for.
    - nleaves_min: The minimum number of leaves for the RJ diagnostics.
    - nleaves_max: The maximum number of leaves for the RJ diagnostics.
    - moves_names: The names of the moves for the acceptance fraction plot.
    - rj_moves_names: The names of the RJ moves for the acceptance fraction plot.
    - use_chainconsumer: Whether to use ChainConsumer for plotting.
    - suffix: The suffix to add to the plot filenames.
    - kwargs: Additional keyword arguments to pass to the plotting functions.

    Returns:
    None
    """

    nwalkers = samp.nwalkers
    ntemps = samp.ntemps
    steps = np.arange(samp.iteration)
    myred = '#9A0202' 
    tempcolors = pysco.plot.get_colors_from_cmap(ntemps, cmap='inferno', reverse=False)    
                        
    for key in samp.branch_names:
        #check_latex()
        chain = get_clean_chain(samp.get_chain(discard=int(samp.iteration*0.3), thin=1)[key], ndim=ndim[key])
        chain = transform_all[key].transform_base_parameters(chain)

        inds = samp.get_inds(discard=int(samp.iteration*0.3))
        logP = samp.get_log_posterior(discard=int(samp.iteration*0.3), thin=1)[:,0]

        if use_chainconsumer:
            try:
                fig = pysco.plot.chainplot(samples=chain, 
                                        truths=truths[key], 
                                        labels=labels[key], 
                                        names=key,
                                        #logP=logP[inds_mask].flatten(),
                                        filename=path + 'diagnostic/' + key + suffix,
                                        return_obj=False, plot_walks=False,
                                        **kwargs,
                                        )
                        
                plt.close()

            except:
                print('ChainConsumer failed to plot the chains. Falling back to `pysco.plot.corner`.')
                fig = pysco.plot.corner(chain, 
                                        truths=truths[key], 
                                        labels=labels[key], 
                                        save=True, 
                                        custom_whspace= 0.15, 
                                        filename=path + 'diagnostic/' + key + '_cornerplot' + suffix, 
                                        dpi=150
                                        )
                plt.close()

        else:
            fig = pysco.plot.corner(chain, 
                                    truths=truths[key], 
                                    labels=labels[key], 
                                    save=True, 
                                    custom_whspace= 0.15, 
                                    filename=path + 'diagnostic/' + key + '_cornerplot' + suffix, 
                                    dpi=150,
                                    linestyle='-',
                                    )
            plt.close()

        #check_latex()
        fig, axs = plt.subplots(ndim[key], 1)
        fig.set_size_inches(20, 20)
        for i in range(ndim[key]):
            for walk in range(nwalkers):
                chain = transform_all[key].transform_base_parameters(samp.get_chain(discard=int(samp.iteration*0.3), thin=1)[key])
                if trace_color is not None:
                    axs[i].plot(chain[:, 0, walk, :, i], color = trace_color, ls='-', alpha = 0.2)
                else:
                    axs[i].plot(chain[:, 0, walk, :, i], ls='-', alpha = 1)
                axs[i].set_ylabel(labels[key][i])

                if truths[key][i] is not None:
                    axs[i].axhline(truths[key][i], color=myred)

        fig.savefig(path + 'diagnostic/' + key + '_traceplot' + suffix, dpi=150)
        plt.close()

        #* Plotting RJ diagnostics
        if key in rj_branches:
            plot_leaves_hist(samp, key, path, tempcolors, nleaves_min, nleaves_max, suffix)

    #* Plotting logL evolution with the number of steps
    plot_logl(samp, path, nwalkers, suffix, true_logl=true_logl)

    #* Plotting acceptance fraction evolution with the number of steps
    plot_acceptance(steps, path, acceptance_all, acceptance_moves, moves_names=moves_names, suffix=suffix)

    if rj_acceptance_all is not None:
        plot_acceptance(steps, path, rj_acceptance_all, rj_acceptance_moves, moves_names=rj_moves_names, suffix='_rj'+suffix)


def plot_leaves_hist(samp, key, path, tempcolors, nleaves_min, nleaves_max, suffix=''):
    """
    Plots a histogram of the number of leaves for each temperature in the given sample.

    Parameters:
    - samp (Sample): The sample object containing the data.
    - key (str): The key representing the specific data to plot.
    - path (str): The path to save the generated plot.
    - tempcolors (list): A list of colors for each temperature.

    Returns:
    - None
    """
    nleaves_rj = samp.get_nleaves()[key] 
    bns = (np.arange(nleaves_min[key], nleaves_max[key] + 2) - 0.5)

    fig = plt.figure()

    for temp, tempcolor in enumerate(tempcolors):
        plt.hist(nleaves_rj[:, temp].flatten(), bins=bns, histtype="stepfilled", edgecolor=tempcolor, facecolor=to_rgba(tempcolor, 0.2), density=True, ls='-', zorder=100-temp)
    
    plt.xlabel('Number of leaves')
    plt.ylabel('Density')
    # Create legend entry for temperature colors
    legend_colors = [to_rgba(tempcolor, 0.2) for tempcolor in tempcolors[::-1]]
    legend_labels = ['' for _ in range(len(tempcolors))]
    legend_labels[0] = r' $T_i$'
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in legend_colors]
    legend_entry = plt.legend(legend_handles, legend_labels, loc='upper right', ncol=len(tempcolors), mode='expand', frameon=False, framealpha=0, bbox_to_anchor=(0.8, 0.95))

    # Add legend entry to the plot
    plt.gca().add_artist(legend_entry)

    fig.text(0.07, 0.083, f"Step: {samp.iteration}", ha='left', va='top', fontfamily='serif', c='red')
    #plt.title(key)
    fig.savefig(path + 'diagnostic/leaves_' + key + suffix, dpi=150)
    plt.close()

def plot_logl(samp, path, nwalkers, suffix='', true_logl=None):
    """
    Plot the log likelihood of the samples.

    Parameters:
    samp (object): The samples object.
    path (str): The path to save the plot.
    nwalkers (int): The number of walkers.

    Returns:
    None
    """
    fig = plt.figure()
    logl = samp.get_log_like(discard=int(samp.iteration*0.3), thin=1)
    for walk in range(nwalkers):
        #plt.plot(logl[:, 0, walk] - maxlogl[walk], color='k', ls='-', alpha=0.2, lw=1)
        plt.plot(logl[:, 0, walk], color='k', ls='-', alpha=0.2, lw=1)

    if true_logl is not None:
        plt.axhline(true_logl, color='r', ls='--', lw=1)
    plt.ylabel(r'$\log{\mathcal{L}}$')
    fig.savefig(path + 'diagnostic/loglike_evolution'+suffix, dpi=150)
    plt.close()

    # plot an histogram of the log likelihood at the current step for the T=1 chain
    logl_here = logl[-1, 0, :].flatten()
    
    fig = plt.figure()
    plt.hist(logl_here, bins=int(nwalkers/3), histtype='step', color='k', alpha=1, lw=2, density=True, label=f"Step: {samp.iteration}")
    if true_logl is not None:
        plt.axvline(true_logl, color='k', ls='--', lw=2)
    plt.xlabel(r'$\log{\mathcal{L}}$')

    plt.legend()

    fig.savefig(path + 'diagnostic/loglike_hist'+suffix, dpi=150)


def plot_acceptance(steps, path, acceptance_all, acceptance_moves, moves_names=None, suffix=''):
    """
    Plot the acceptance fraction for a given number of steps.
    
    Parameters:
    - steps (array-like): The steps for which the acceptance fraction is calculated.
    - path (str): The path where the plot will be saved.
    - acceptance_all (array-like): The acceptance fraction for all proposals.
    - acceptance_moves (list of array-like): The acceptance fraction for each proposal move.
    
    Returns:
    None
    """
    fig = plt.figure()                    
    plt.semilogy(steps, acceptance_all, color='k', label='Total')
    
    if acceptance_moves is not None:
        for i, move in enumerate(acceptance_moves):
            move = np.array(move)
            nanmask = np.isnan(move)

            if moves_names is not None:
                label = moves_names[i]
            else:
                label = f'Move {i}'

            plt.semilogy(steps[~nanmask], move[~nanmask], label=label)
    
    plt.ylabel(r'Acceptance fraction')
    plt.legend()
    fig.savefig(path + 'diagnostic/acceptance'+suffix, dpi=150)
    plt.close()


def general_act(sampler, discard=None, all_T=False, return_max=True, act_kwargs={}):
    """
    Compute the generalised auto-correlation time for the given sampler.

    Parameters:
    - sampler (object): The sampler object.
    - discard (int): The number of samples to discard. Default is None.
    - all_T (bool): Whether to compute the auto-correlation time for all temperatures. Default is False.
    - return_max (bool): Whether to return the maximum auto-correlation time. Default is True.
    - act_kwargs (dict): Additional keyword arguments to pass to the auto-correlation time function.

    Returns:
    if return_max:
    - tau_all (float): The maximum auto-correlation time across all the parameters.

    else:
    - tau_all (array-like): The auto-correlation time for all the parameters.
    """
    if discard is None:
        discard = 0.3 * sampler.iteration

    samples = sampler.get_chain(discard=int(discard), thin=1)

    tau_all_dict = get_integrated_act(samples, **act_kwargs)
    tau_all = []

    for values in tau_all_dict.values:
        values = np.atleast_2d(values)
        nw, nt = values.shape
        if not all_T:
            act = values[:, 0]
        
        tau_all.append(act)
    
    tau_all = np.array(tau_all)

    if return_max:
        tau_all.flatten()
        return np.max(tau_all)
    
    return tau_all


class GelmanRubinStopping(Stopping):
    def __init__(self, threshold=1.1, per_walker=False, transform_fn=None, sort_fn=None, discard=0.3, thin=True, start_iteration=0, verbose=False):
        """Stopping criterion based on the Gelman-Rubin convergence diagnostic.

        Args:
            threshold (float): Threshold value for the Gelman-Rubin diagnostic. Default is 1.1.
            per_walker (bool): Whether to compute the Gelman-Rubin diagnostic per walker. Default is False.
            transform_fn (callable): Function to transform the chains before computing the diagnostic. Default is None.
            sort_fn (callable): Function to sort the chains before computing the diagnostic. Default is None.
            discard (float): Fraction of the samples to discard before computing the diagnostic. Default is 0.3.
            thin (bool): Whether to thin the chains before computing the diagnostic. Default is True.
            start_iteration (int): Iteration to start applying the stopping criterion. Default is 0.
            verbose (bool): Whether to print the diagnostic values. Default is False.
        """

        self.threshold = threshold
        self.per_walker = per_walker
        self.transform_fn = transform_fn
        self.sort_fn = sort_fn
        self.discard = discard
        self.thin = thin
        self.start_iteration = start_iteration
        self.verbose = verbose

    def __call__(self, iter, sample, sampler):
        """Call update function.

        Args:
            iter (int): Iteration of the sampler.
            last_sample (obj): Last state of sampler (:class:`eryn.state.State`).
            sampler (obj): Full sampler oject (:class:`eryn.ensemble.EnsembleSampler`).

        Returns:
            bool: Value of ``stop``. If ``True``, stop sampling.
            
        """
        if iter < self.start_iteration:
            return False
        
        tau = general_act(sampler, discard=self.discard) if self.thin else 1
        
        chains = sampler.get_chain(discard=int(self.discard), thin=tau)

        if self.transform_fn is not None:
            chains = self.transform_fn(chains)

            ndims = chains.shape[-1]
            Rhat = psrf(chains, ndims, self.per_walker)

            if self.verbose:
                print(f'Gelman-Rubin Rhat: {Rhat}')

        else:
            Rhat = []

            for name, chain in chains.items:
                chain = chain[:, 0, :, :, :]
                nsteps, nwalkers, nleaves, ndims = chain.shape

                #TODO: how to handle the case when I have multiple leaves?
                #TODO: test also on the logl and number of leaves
                if nleaves > 1:
                    chain = self.sort_fn(chain)
                chain = chain.reshape(nsteps, nwalkers, -1)

                Rhat_tmp = psrf(chains, ndims, self.per_walker)
                Rhat += [Rhat_tmp]

                if self.verbose:
                    print(f'Gelman-Rubin Rhat in the branch {name}: {Rhat_tmp}')

        Rhat = np.array(Rhat)
        
        return np.all(Rhat < self.threshold)
    

class AutoCorrelationStopping(Stopping):
    
    def __init__(self, autocorr_multiplier=50, verbose=False, n_iter=1000):
        self.autocorr_multiplier = autocorr_multiplier
        self.verbose = verbose
        self.time = 0
        self.n_iter = n_iter
        self.when_to_stop = np.inf

    def __call__(self, iter, last_sample, sampler):
        samples = sampler.get_chain(discard=int(sampler.iteration*0.3), thin=1)

        tau = get_integrated_act(samples)

        if self.time > 0:
            iteration = sampler.iteration

            if iteration >= self.when_to_stop:
                return True
            
            elif iteration < self.when_to_stop and np.isfinite(self.when_to_stop):
                return False

            finish = []

            for name, values in tau.items():
                converged = np.all(tau[name] * self.autocorr_multiplier < iteration)
                converged &= np.all(
                    np.abs(self.old_tau[name] - tau[name]) / tau[name] < 0.01
                )

                finish.append(converged)

            if np.all(finish):
                stop = True

                if self.when_to_stop == np.inf:
                    self.when_to_stop = iteration + self.n_iter
            
            else:
                stop = False
                
            
            if self.verbose:
                print(
                    "\ntau:",
                    tau,
                    "\nIteration:",
                    iteration,
                    "\nAutocorrelation multiplier:",
                    self.autocorr_multiplier,
                    "\nStopping:",
                    stop,
                    "\n",
                )

        else:
            stop = False

        self.old_tau = tau
        self.time += 1
        return stop



"""
    def __init__(self, autocorr_multiplier=50, verbose=False):
            self.autocorr_multiplier = autocorr_multiplier
            self.verbose = verbose

            self.time = 0

        def __call__(self, iter, last_sample, sampler):

            tau = sampler.backend.get_autocorr_time(multiply_thin=False)

            if self.time > 0:
                # backend iteration
                iteration = sampler.backend.iteration

                finish = []

                for name, values in tau.items():
                    converged = np.all(tau[name] * self.autocorr_multiplier < iteration)
                    converged &= np.all(
                        np.abs(self.old_tau[name] - tau[name]) / tau[name] < 0.01
                    )

                    finish.append(converged)

                stop = True if np.all(finish) else False
                if self.verbose:
                    print(
                        "\ntau:",
                        tau,
                        "\nIteration:",
                        iteration,
                        "\nAutocorrelation multiplier:",
                        self.autocorr_multiplier,
                        "\nStopping:",
                        stop,
                        "\n",
                    )

            else:
                stop = False

            self.old_tau = tau
            self.time += 1
            return stop
"""
    