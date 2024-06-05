import sys, os
import numpy as np
import pysco
import matplotlib as mpl
from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt

import warnings

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

def adjust_covariance(samp, discard=0.8, svd=False, skip_idxs=[]):
    """
    Adjusts the covariance matrix for each branch in the given sample.

    Parameters:
    samp (Sampler): The sampler object.
    discard (float, optional): The fraction of iterations to discard. Defaults to 0.8.
    svd (bool, optional): Whether to perform singular value decomposition on the covariance matrix. Defaults to False.
    skip_idxs (list, optional): List of indices to skip. Defaults to an empty list.

    Returns:
    None
    """

    discard = int(discard * samp.iteration)
    ndims = samp.ndims
    accept_update = ['GaussianMove'] #moves which have a covariance matrix to adjust

    for i, move in enumerate(samp.moves):
        #if hasattr(move, 'all_proposal') and i not in skip_idxs:
        if move.__name__ in accept_update and i not in skip_idxs:
            branches = move.all_proposal.keys()

            for key in branches:
                item_samp = samp.get_chain(discard=discard)[key][:, 0][samp.get_inds(discard=discard)[key][:, 0]]
            
                cov = np.cov(item_samp, rowvar=False) * 2.38**2 / ndims[key]
                if svd:
                    svd = np.linalg.svd(cov)
                    move.all_proposal[key].svd = svd
                
                move.all_proposal[key].scale = cov


class DiagnosticPlotter:
    """
    A class for generating diagnostic plots based on the state of the sampler.

    Attributes:
    - sampler: The sampler object to use for the diagnostic plots.
    - path: The path to save the diagnostic plots.
    - truths: A dictionary containing the true values for each branch.
    - labels: A dictionary containing the labels for each parameter in each branch.
    - transform_all: A ``transform_container`` object containing the transformation functions for each branch.
    - true_logl: The true log likelihood value.
    - discard: The fraction of iterations to discard before plotting.
    - suffix: A suffix to append to the filenames of the diagnostic plots.

    Methods:
    - setup(sampler): Set up the diagnostic plotter with the given sampler.
    - __call__(**kwargs): Perform various plotting operations based on the current state of the sampler.
    - plot_corners(samples, logl, trace=True, **kwargs): Plot corner plots and trace plots for the given samples.
    - plot_acceptance(): Plot the acceptance fraction for each update step in the MCMC sampling process.
    - plot_leaves_hist(): Plot the histogram of the number of leaves for each temperature.
    """

    def __init__(self, sampler, path, truths, labels, transform_all=None, true_logl=None, discard=0.3, suffix='') -> None:
        self.sampler = sampler
        self.path = path
        self.truths = truths
        self.labels = labels
        self.transform_all = transform_all
        self.discard = discard
        self.true_logl = true_logl
        self.suffix = suffix
        self.setup(sampler)

    def setup(self, sampler):
        """
        Set up the diagnostic plotter with the given sampler.

        Parameters:
        - sampler: The sampler object to use for the diagnostic plots.

        Returns:
        None
        """
        if self.sampler is None and sampler is not None:
            # Retrieve information from the sampler
            print('Retrieving information from the sampler.')
            self.sampler = sampler
            self.nw = sampler.nwalkers
            self.nt = sampler.ntemps
            self.nleaves_max = sampler.nleaves_max

            if hasattr(sampler, 'nleaves_min'):
                self.nleaves_min = sampler.nleaves_min
            else:
                self.nleaves_min = self.nleaves_max

            self.rj_branches = [key for key in sampler.branch_names if self.nleaves_min[key] != self.nleaves_max[key]]

            self.has_rj = len(self.rj_branches) > 0

            self.tempcolors = pysco.plot.get_colors_from_cmap(self.nt, cmap='inferno', reverse=False)

            if not isinstance(self.truths, dict):
                self.truths = np.atleast_2d(self.truths)
                self.truths = {key: self.truths[i] for i, key in enumerate(sampler.branch_names)}
            
            if not isinstance(self.labels, dict):
                if not isinstance(self.labels[0], list):
                    self.labels = [self.labels]
                self.labels = {key: self.labels[i] for i, key in enumerate(sampler.branch_names)}

            # Acceptance fraction diagnostics
            self.steps = []
            self.acceptance_all = []
            self.acceptance_moves = {}

            for i, move in enumerate(sampler.moves):
                try:
                    label = move.label
                except:
                    label = f'Move {i}'
                self.acceptance_moves[label] = []

            if self.has_rj:
                self.rj_acceptance_all = []
                self.rj_acceptance_moves = {}

                for i, move in enumerate(sampler.rj_moves):
                    try:
                        label = move.label
                    except:
                        label = f'Move {i}'
                    self.rj_acceptance_moves[label] = []

        else:
            # The sampler is already set up or a valid sampler was not provided
            pass

    def __call__(self, return_samples=False, **kwargs):
        """
        Perform various plotting operations based on the current state of the sampler.

        Parameters:
        - kwargs: Additional keyword arguments to be passed to the plotting functions.

        Returns:
        None
        """
        iteration = self.sampler.iteration
        samples = self.sampler.get_chain(discard=int(self.discard * iteration), thin=1)
        logl = self.sampler.get_log_like(discard=int(self.discard * iteration), thin=1)

        # Plot the corner and trace plots
        self.plot_corners(samples, logl[:, 0].flatten(), **kwargs)

        # Plot the acceptance fraction evolution
        self.plot_acceptance()

        # Plot the RJ diagnostics
        if self.has_rj:
            self.plot_leaves_hist()

        # Plot the integrated autocorrelation time evolution
        self.plot_act_evolution(N=10, all_T=False, **kwargs)

        # Plot the log likelihood evolution
        self.plot_logl_evolution(logl)

        if return_samples:
            return samples, logl

    def plot_corners(self, samples, logl, trace=True, **kwargs):
        """
        Plot corner plots and trace plots for the given samples.

        Args:
            samples (dict): A dictionary containing the samples for each branch.
            logl (array-like): The log likelihood values.
            trace (bool, optional): Whether to plot trace plots. Defaults to True.
            **kwargs: Additional keyword arguments to be passed to the corner plot function.

        Returns:
        None
        """
        for key in self.sampler.branch_names:
            ndims = self.sampler.ndims[key]

            if self.transform_all is not None:
                    samples_here = self.transform_all[key].both_transforms(samples[key])
            else:
                samples_here = samples[key]

            chain = get_clean_chain(samples_here, ndim=ndims)

            truths_here = self.truths[key]
            labels_here = self.labels[key]

            if self.nleaves_min[key] == self.nleaves_max[key]:
                truths_here = np.append(truths_here, self.true_logl)
                labels_here = np.append(labels_here, r'$\log{\mathcal{L}}$')

                chain = np.column_stack((chain, logl))

                fig = pysco.plot.corner(chain,
                                        truths=truths_here,
                                        labels=labels_here,
                                        save=True,
                                        custom_whspace=0.15,
                                        filename=self.path + key + '_cornerplot' + self.suffix,
                                        dpi=150,
                                        linestyle='-',
                                        **kwargs
                                        )
            plt.close()

            if trace:
                fig, axs = plt.subplots(self.sampler.ndims[key], 1, sharex=True)
                fig.set_size_inches(20, 20)

                for i in range(ndims):
                    for walk in range(self.nw):
                        axs[i].plot(samples_here[:, 0, walk, :, i], ls='-', alpha=0.7, lw=1)

                    axs[i].set_ylabel(self.labels[key][i])

                    if truths_here[i] is not None:
                        axs[i].axhline(truths_here[i], color='k', ls='--', lw=2)

                fig.savefig(self.path + key + '_traceplot' + self.suffix, dpi=150)
                plt.close()

    def plot_acceptance(self):
        """
        Plot the acceptance fraction for each update step in the MCMC sampling process.
        If the `has_rj` attribute is True, it also plots the RJ acceptance fraction.

        Returns:
        None
        """
        fig = plt.figure()
        self.steps.append(self.sampler.iteration)
        steps = np.array(self.steps)
        self.acceptance_all.append(np.mean(self.sampler.acceptance_fraction[0]))

        plt.plot(self.steps, self.acceptance_all, color='k', label='Total', marker='x', lw=1)

        for i, move in enumerate(self.sampler.moves):
            try:
                label = move.label
            except:
                label = f'Move {i}'
            self.acceptance_moves[label].append(np.mean(move.acceptance_fraction[0]))

            acceptance = np.array(self.acceptance_moves[label])

            nanmask = np.isnan(acceptance)

            plt.plot(steps[~nanmask], acceptance[~nanmask], label=label, marker='x', lw=1)

        plt.axhline(0.234, color='k', ls='--', lw=2, label=r'$\alpha=23.4\%$')

        plt.xlabel(r'$N_{\rm steps}$')
        plt.ylabel(r'Acceptance fraction')
        plt.legend()
        fig.savefig(self.path + 'acceptance' + self.suffix, dpi=150)
        plt.close()

        if self.has_rj:
            fig = plt.figure()
            self.rj_acceptance_all.append(np.mean(self.sampler.rj_acceptance_fraction[0]))

            plt.plot(self.steps, self.rj_acceptance_all, color='k', label='Total', marker='x')

            for i, move in enumerate(self.sampler.rj_moves):
                try:
                    label = move.label
                except:
                    label = f'Move {i}'
                self.rj_acceptance_moves[label].append(np.mean(move.acceptance_fraction[0]))

                nanmask = np.isnan(self.rj_acceptance_moves[label])

                plt.plot(self.steps[~nanmask], self.rj_acceptance_moves[label][~nanmask], label=label, marker='x')

            plt.xlabel(r'$N_{\rm steps}$')
            plt.ylabel(r'RJ Acceptance fraction')
            plt.legend()
            fig.savefig(self.path + 'acceptance_rj' + self.suffix, dpi=150)
            plt.close()

    def plot_leaves_hist(self):
        """
        Plot the histogram of the number of leaves for each temperature.

        This method plots a histogram of the number of leaves for each temperature in the `rj_branches` dictionary.
        It uses the `sampler` object to get the number of leaves for each temperature.
        The histogram is plotted using the `plt.hist` function from the `matplotlib.pyplot` module.
        The plot includes temperature-specific colors and a legend for the colors.

        Returns:
        None
        """
        for key in self.rj_branches:
            nleaves_rj = self.sampler.get_nleaves()[key]
            bns = (np.arange(self.nleaves_min[key], self.nleaves_max[key] + 2) - 0.5)

            fig = plt.figure()

            for temp, tempcolor in enumerate(self.tempcolors):
                plt.hist(nleaves_rj[:, temp].flatten(), bins=bns, histtype="stepfilled", edgecolor=tempcolor,
                         facecolor=to_rgba(tempcolor, 0.2), density=True, ls='-', zorder=100 - temp)

            plt.xlabel('Number of leaves')
            plt.ylabel('Density')
            # Create legend entry for temperature colors
            legend_colors = [to_rgba(tempcolor, 0.2) for tempcolor in self.tempcolors[::-1]]
            legend_labels = ['' for _ in range(len(self.tempcolors))]
            legend_labels[0] = r' $T_i$'
            legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in legend_colors]
            legend_entry = plt.legend(legend_handles, legend_labels, loc='upper right', ncol=len(self.tempcolors), mode='expand', frameon=False, framealpha=0, bbox_to_anchor=(0.8, 0.95))

            # Add legend entry to the plot
            plt.gca().add_artist(legend_entry)

            fig.text(0.07, 0.083, f"Step: {self.sampler.iteration}", ha='left', va='top', fontfamily='serif', c='k')
            #plt.title(key)
            fig.savefig(self.path + 'leaves_' + key + self.suffix, dpi=150)
            plt.close()
    
    def plot_act_evolution(self, N=10, all_T=False, **kwargs):
        """
        Plot the auto-correlation time evolution.

        Parameters:
        - N (int): Number of points to plot.
        - all_T (bool): Whether to plot for all temperatures or just one.
        - **kwargs: Additional keyword arguments to pass to `get_integrated_act` function.

        Returns:
        - None

        Raises:
        - None
        """

        fig = plt.figure()
        toplim = 0

        samples = self.sampler.get_chain(discard=0, thin=1)
        Npoints = np.exp(np.linspace(np.log(min(100, self.sampler.iteration)), np.log(self.sampler.iteration), N)).astype(int)

        for key, color in zip(samples.keys(), pysco.plot.get_colorslist(colors='colors10')):

            chain = samples[key]
            nsteps, nt, nw, nleaves, ndim = chain.shape

            chain = chain.reshape(nsteps, nt, nw, nleaves * ndim)

            ntemps = chain.shape[1] if all_T else 1
            tau = np.empty(shape=(len(Npoints), ntemps, chain.shape[-1]))

            try:
                for i, N in enumerate(Npoints):
                    tau[i] = get_integrated_act(chain[:N, :ntemps], average=True, **kwargs)

                for temp in range(ntemps):
                    where_max = np.argmax(tau[:, temp], axis=-1)

                    to_plot =np.max(tau[:, temp], axis=-1)

                    plt.loglog(Npoints, to_plot, color=color, marker='o', alpha=0.9, label=key + fr' - $T_{temp}$')

                    toplim = max(toplim, 5 * max(tau[-1, temp, where_max]))
                
            except:
                warnings.warn(f"Could not compute the auto-correlation times for the branch {key}.")

        plt.loglog(Npoints, Npoints / 50, label=r'$\tau = N/50$', linestyle='--', color='black')

        plt.xlabel(r'$N_{\rm steps}$')
        plt.ylabel(r'$\tau$')

        plt.ylim(0, toplim)
        plt.legend()

        fig.savefig(self.path + 'act_evolution' + self.suffix, dpi=150)

    
    def plot_logl_evolution(self, logl):
        """
        Plots the evolution of log-likelihood values.

        Args:
            logl (numpy.ndarray): Array of log-likelihood values.

        Returns:
            None
        """
        fig = plt.figure()
        for walk in range(self.nw):
            plt.plot(logl[:, 0, walk], ls='-', alpha=0.7, lw=1)

        if self.true_logl is not None:
            plt.axhline(self.true_logl, color='k', ls='--', lw=2)
        plt.ylabel(r'$\log{\mathcal{L}}$')
        fig.savefig(self.path + 'loglike_evolution' + self.suffix, dpi=150)

        plt.close()

        
def plot_diagnostics(samp, path, ndim, truths, labels, transform_all, acceptance_all=None, true_logl=None, trace_color=None, acceptance_moves=None, rj_acceptance_all=None, rj_acceptance_moves=None, rj_branches=[], nleaves_min=None, nleaves_max=None, moves_names=None, rj_moves_names=None, use_chainconsumer=False, suffix='', **kwargs):
    
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
    myred = '#9A0202' 
    tempcolors = pysco.plot.get_colors_from_cmap(ntemps, cmap='inferno', reverse=False)    
                        
    for key in samp.branch_names:
        #check_latex()
        chain = get_clean_chain(samp.get_chain(discard=int(samp.iteration*0.3), thin=1)[key], ndim=ndim[key])
        chain = transform_all[key].transform_base_parameters(chain)

        inds = samp.get_inds(discard=int(samp.iteration*0.3))
        logP = samp.get_log_posterior(discard=int(samp.iteration*0.3), thin=1)[:,0]

        truths_here = truths[key]
        labels_here = labels[key]

        if nleaves_min[key] == nleaves_max[key]:
            logl = samp.get_log_like(discard=int(samp.iteration*0.3), thin=1)[:, 0].flatten()
            truths_here = np.append(truths_here, true_logl)
            labels_here = np.append(labels_here, r'$\log{\mathcal{L}}$')

            chain = np.column_stack((chain, logl))

        if use_chainconsumer:
            try:
                fig = pysco.plot.chainplot(samples=chain, 
                                        truths=truths_here, 
                                        labels=labels_here, 
                                        names=key,
                                        #logP=logP[inds_mask].flatten(),
                                        filename=path + key + suffix,
                                        return_obj=False, plot_walks=False,
                                        **kwargs,
                                        )
                        
                plt.close()

            except:
                print('ChainConsumer failed to plot the chains. Falling back to `pysco.plot.corner`.')
                fig = pysco.plot.corner(chain, 
                                        truths=truths_here, 
                                        labels=labels_here, 
                                        save=True, 
                                        custom_whspace= 0.15, 
                                        filename=path + key + '_cornerplot' + suffix, 
                                        dpi=150
                                        )
                plt.close()

        else:
            fig = pysco.plot.corner(chain, 
                                    truths=truths_here, 
                                    labels=labels_here, 
                                    save=True, 
                                    custom_whspace= 0.15, 
                                    filename=path + key + '_cornerplot' + suffix, 
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

        fig.savefig(path + key + '_traceplot' + suffix, dpi=150)
        plt.close()

        #* Plotting RJ diagnostics
        if key in rj_branches:
            plot_leaves_hist(samp, key, path, tempcolors, nleaves_min, nleaves_max, suffix)

    #* Plotting logL evolution with the number of steps
    plot_logl(samp, path, suffix, true_logl=true_logl)

    #* Plotting acceptance fraction evolution with the number of steps
    if acceptance_all is not None:
        steps = np.arange(len(acceptance_all))
        plot_acceptance(steps, path, acceptance_all, acceptance_moves, moves_names=moves_names, suffix=suffix)

    if rj_acceptance_all is not None:
        steps = np.arange(len(rj_acceptance_all))
        plot_acceptance(steps, path, rj_acceptance_all, rj_acceptance_moves, moves_names=rj_moves_names, suffix='_rj'+suffix)

    #* Plotting the integrated autocorrelation time evolution
    plot_act_evolution(samp, path, N=10, discard=0, all_T=False, suffix=suffix)


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
    fig.savefig(path + 'leaves_' + key + suffix, dpi=150)
    plt.close()

def plot_logl(samp, path, suffix='', true_logl=None):
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
    nwalkers = logl.shape[2]
    for walk in range(nwalkers):
        #plt.plot(logl[:, 0, walk] - maxlogl[walk], color='k', ls='-', alpha=0.2, lw=1)
        plt.plot(logl[:, 0, walk], color='k', ls='-', alpha=0.2, lw=1)

    if true_logl is not None:
        plt.axhline(true_logl, color='r', ls='--', lw=1)
    plt.ylabel(r'$\log{\mathcal{L}}$')
    fig.savefig(path + 'loglike_evolution'+suffix, dpi=150)
    plt.close()

    # plot an histogram of the log likelihood at the current step for the T=1 chain
    logl_here = logl[-1, 0, :].flatten()
    
    fig = plt.figure()
    plt.hist(logl_here, bins=int(nwalkers/3), histtype='step', color='k', alpha=1, lw=2, density=True, label=f"Step: {samp.iteration}")
    if true_logl is not None:
        plt.axvline(true_logl, color='k', ls='--', lw=2)
    plt.xlabel(r'$\log{\mathcal{L}}$')

    plt.legend()

    fig.savefig(path + 'loglike_hist'+suffix, dpi=150)


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
    fig.savefig(path + 'acceptance'+suffix, dpi=150)
    plt.close()


def plot_act_evolution(samp, path, N=10, discard=0, all_T=False, suffix='', act_kwargs={}):


    samples = samp.get_chain(discard=int(discard), thin=1)
    Npoints = np.exp(np.linspace(np.log(min(100, samp.iteration)), np.log(samp.iteration), N)).astype(int)

    fig = plt.figure()
    toplim = 0

    for key, color in zip(samples.keys(), pysco.plot.get_colorslist(colors='colors10')):
        #try:
        chain = samples[key]
        nsteps, nt, nw, nleaves, ndim = chain.shape

        chain = chain.reshape(nsteps, nt, nw, nleaves * ndim)

        ntemps = chain.shape[1] if all_T else 1
        tau = np.empty(shape=(len(Npoints), ntemps, chain.shape[-1]))

        try:
            for i, N in enumerate(Npoints):
                tau[i] = get_integrated_act(chain[:N, :ntemps], average=True, **act_kwargs)

            for temp in range(ntemps):
                where_max = np.argmax(tau[:, temp], axis=-1)

                to_plot =np.max(tau[:, temp], axis=-1)

                plt.loglog(Npoints, to_plot, color=color, marker='o', label=key + fr' - $T_{temp}$')

                toplim = max(toplim, 5 * max(tau[-1, temp, where_max]))
        
        except:
            warnings.warn(f"Could not compute the auto-correlation times for the branch {key}.")
        
    plt.loglog(Npoints, Npoints / 50, label=r'$\tau = N/50$', linestyle='--', color='black')

    plt.xlabel('Number of samples')
    plt.ylabel(r'$\tau$')

    plt.ylim(0, toplim)
    plt.legend()

    fig.savefig(path + 'act_evolution' + suffix, dpi=150)




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

    for values in tau_all_dict.values():
        
        values = np.array(values)
        if len(values.shape) == 2: # (ntemps, ndims)
            values = values[None, : , :] # (nwalks, ntemps, ndims). Generalized to deal with individual walkers. If ``act_kwargs['average'] = `False` the first dimension is the number of walkers, otherwise it is 1. 
        
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
    
    def __init__(self, autocorr_multiplier=50, verbose=False, N=0, n_skip=0, ess=None, transform_fn=None):
        """
        Stopping criterion based on the auto-correlation time.

        Args:
            autocorr_multiplier (float): Multiplier for the auto-correlation time. Default is 50.
            verbose (bool): Whether to print the diagnostic values. Default is False.
            N (int): Number of iterations to run after convergence. Default is 0, meaning that the run is stopped as soon as the threshold is reached.
            n_skip (int): Number of iterations to skip before applying the stopping criterion. Default is 0.
            ess (float): Effective sample size. Default is None.
            transform_fn (callable): Function to transform the chains before computing the diagnostic. Default is None.
        """

        self.autocorr_multiplier = autocorr_multiplier
        self.verbose = verbose
        self.time = 0
        self.N = N
        self.n_skip = n_skip
        self.ess = ess
        self.when_to_stop = np.inf
        self.transform_fn = transform_fn

    def __call__(self, iter, last_sample, sampler):
        """
        Call update function.

        Args:
            iter (int): Iteration of the sampler.
            last_sample (obj): Last state of sampler (:class:`eryn.state.State`).
            sampler (obj): Full sampler oject (:class:`eryn.ensemble.EnsembleSampler`).
        
        Returns:
            bool: ``True`` if the stopping criterion is met, ``False`` otherwise. 

        """
        samples = sampler.get_chain(discard=0, thin=1)

        if self.transform_fn is not None:
            samples = self.transform_fn(samples)

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

                tau_max = np.max([np.max(tau[name]) for name in tau.keys()])
                if self.ess is not None:
                    nw = last_sample.log_like.shape[1]
                    self.N = self.get_N_from_ess(nw, tau_max)           

                if self.when_to_stop == np.inf:
                    self.when_to_stop = max(self.n_skip, int(iteration / 10) + self.N)
            
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
        return False
    

    def get_N_from_ess(self, nw, tau):
        """
        Get the number of samples from the effective sample size.

        Args:
            nw (int): Number of walkers.
            tau (float): Auto-correlation time.

        Returns:
            int: Target number of samples.
        """
        
        return int(np.ceil(self.ess * tau / nw))