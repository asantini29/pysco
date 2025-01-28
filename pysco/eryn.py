import sys, os
import numpy as np
import pysco
import matplotlib as mpl
from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt
import pandas as pd
from .utils import find_files

import warnings

from eryn.utils import get_integrated_act, psrf, Stopping, SearchConvergeStopping, stepping_stone_log_evidence
from eryn.backends import HDFBackend
from eryn import moves

def get_numpy(x):
    try:
        return x.get()
    except:
        return np.asarray(x)


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

def adjust_covariance(samp, discard=0.8, svd=False, lr=None, skip_idxs=[]):
    """
    Adjusts the covariance matrix for each branch in the given sample.

    Parameters:
    samp (Sampler): The sampler object.
    discard (float, optional): The fraction of iterations to discard. Defaults to 0.8.
    svd (bool, optional): Whether to perform singular value decomposition on the covariance matrix. Defaults to False.
    skip_idxs (list, optional): List of indices to skip. Defaults to an empty list.
    lr (float, optional): The learning rate for the covariance matrix adjustment. Defaults to None. 
                          If None, the covariance matrix is computed from the samples. Else, the covariance matrix is 
                          adjusted by a factor of `lr`.
                          
    Returns:
    None
    """

    discard = int(discard * samp.iteration)
    ndims = samp.ndims
    accept_update = ['GaussianMove'] #moves which have a covariance matrix to adjust

    for i, move in enumerate(samp.moves):
        #if hasattr(move, 'all_proposal') and i not in skip_idxs:
        name = move.__class__.__name__
        if name in accept_update and i not in skip_idxs:
            branches = move.all_proposal.keys()

            for key in branches:

                if lr is not None:
                    acceptance = np.mean(move.acceptance_fraction[0]) 
                    signed_lr = lr if acceptance > 0.234 else -lr
                    cov = move.all_proposal[key].scale * (1 + signed_lr) 
                else:
                    item_samp = samp.get_chain(discard=discard)[key][:, 0][samp.get_inds(discard=discard)[key][:, 0]]
                    cov = np.cov(item_samp, rowvar=False) * 2.38**2 / ndims[key]

                if svd:
                    svd = np.linalg.svd(cov)
                    move.all_proposal[key].svd = svd
                
                move.all_proposal[key].scale = cov

def adjust_gamma0(samp, skip_idxs=[]):
    """
    Adjusts the ``gamma_0`` parameter for the given sample.

    Parameters:
    samp (Sampler): The sampler object.
    skip_idxs (list, optional): List of indices to skip. Defaults to an empty list.

    Returns:
    None
    """
    
    accept_update = ['DEMove'] #moves which have a ``gamma_0`` parameter to adjust
    for i, move in enumerate(samp.moves):
        name = move.__class__.__name__
        if name in accept_update and i not in skip_idxs:

            acceptance = np.mean(move.acceptance_fraction[0])
            factor = np.sqrt(acceptance / 0.234)

            factor = 0.1 if factor < 1e-5 else factor

            if isinstance(move.g0, dict):
                for key in move.g0.keys():
                    print(f"Adjusting gamma_0 for {key}: old = {move.g0[key]}, new = {move.g0[key] * factor}")
                    move.g0[key] = move.g0[key] * factor
            
            else:
                move.g0 = move.g0 * factor

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
    - converter: A callable object to convert the samples to a different basis.

    Methods:
    - setup(sampler): Set up the diagnostic plotter with the given sampler.
    - __call__(**kwargs): Perform various plotting operations based on the current state of the sampler.
    - plot_corners(samples, logl, trace=True, **kwargs): Plot corner plots and trace plots for the given samples.
    - plot_acceptance(): Plot the acceptance fraction for each update step in the MCMC sampling process.
    - plot_leaves_hist(): Plot the histogram of the number of leaves for each temperature.
    """

    def __init__(self, sampler, path, truths, labels, plot_kwargs={}, plot_all_temps=False, transform_all=None, true_logl=None, discard=0.3, suffix='', converter=None) -> None:
        self.sampler = sampler
        self.path = path
        self.truths = truths
        self.labels = labels
        self.transform_all = transform_all
        self.discard = discard
        self.plot_kwargs = plot_kwargs
        self.plot_all_temps = plot_all_temps
        self.true_logl = true_logl
        self.suffix = suffix
        self.converter = converter

        self.mixture_here = False

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

            self.temperaturestoplot = np.arange(self.nt, dtype=np.int32) if self.plot_all_temps else [0]   
            self.nleaves_max = sampler.nleaves_max

            if hasattr(sampler, 'nleaves_min'):
                self.nleaves_min = sampler.nleaves_min
            else:
                self.nleaves_min = self.nleaves_max

            self.rj_branches = [key for key in sampler.branch_names if self.nleaves_min[key] != self.nleaves_max[key]]

            self.has_rj = len(self.rj_branches) > 0
            print('All branches:', sampler.branch_names)
            print('Branches with RJ:', self.rj_branches)

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
                self.acceptance_moves[label] = np.array([])

                if isinstance(move, moves.GaussianMixtureProposal):
                    self.mixture_here = True
                    self.mixture_idx = i

            if self.has_rj:
                self.rj_acceptance_all = []
                self.rj_acceptance_moves = {}

                for i, move in enumerate(sampler.rj_moves):
                    try:
                        label = move.label
                    except:
                        label = f'Move {i}'
                    self.rj_acceptance_moves[label] = np.array([])

        else:
            # The sampler is already set up or a valid sampler was not provided
            pass

    def __call__(self, return_samples=False, **kwargs):
        """
        Perform various plotting operations based on the current state of the sampler.

        Parameters:
        - return_samples (bool): Whether to return the samples and log likelihood values. Defaults to False.
        - kwargs: Additional keyword arguments to be passed to the plotting functions.

        Returns:
        None
        """
        iteration = self.sampler.iteration
        samples = self.sampler.get_chain(discard=int(self.discard * iteration), thin=1)
        logl = self.sampler.get_log_like(discard=int(self.discard * iteration), thin=1)
        betas = self.sampler.get_betas(discard=int(self.discard * iteration), thin=1)

        all_kwargs = {**self.plot_kwargs, **kwargs}
        # Plot the corner and trace plots
        self.plot_corners(samples, logl[:, 0].flatten(), **all_kwargs)

        # Plot the acceptance fraction evolution
        self.plot_acceptance()

        # Plot the RJ diagnostics
        if self.has_rj:
            self.plot_leaves_hist()

        # Plot the integrated autocorrelation time evolution
        self.plot_act_evolution(N=10, all_T=False, **kwargs)

        # Plot the log likelihood evolution
        self.plot_logl_evolution(logl)
        self.plot_logl_betas(betas, logl)



        if return_samples:
            return samples, logl

    def  plot_corners(self, samples, logl, trace=True, covs=True, **kwargs):
        """
        Plot corner plots and trace plots for the given samples.

        Args:
            samples (dict): A dictionary containing the samples for each branch.
            logl (array-like): The log likelihood values.
            trace (bool, optional): Whether to plot trace plots. Defaults to True.
            covs (bool, optional): Whether to plot the diagonal elements of the covariance matrix of the samples. Defaults to True.
            **kwargs: Additional keyword arguments to be passed to the corner plot function.

        Returns:
        None
        """
        if self.mixture_here:
            mix_move = self.sampler.moves[self.mixture_idx]
            dgpmm_dict = mix_move.dpgmms

        for key in self.sampler.branch_names:
            ndims = self.sampler.ndims[key]

            if self.transform_all is not None:
                    samples_here = self.transform_all[key].both_transforms(samples[key])
            else:
                samples_here = samples[key]

            nsteps = samples_here.shape[0]
            truths_here = self.truths[key]
            labels_here = self.labels[key]

            for temp in self.temperaturestoplot:
                chain = get_clean_chain(samples_here, ndim=ndims, temp=temp)

                if self.nleaves_min[key] == self.nleaves_max[key] == 1:
                    truths_here = np.append(truths_here, self.true_logl)
                    labels_here = np.append(labels_here, r'$\log{\mathcal{L}}$')

                    chain = np.column_stack((chain, logl))
                    logl_here = True

                try:
                    fig = pysco.plot.corner(chain,
                                            truths=truths_here,
                                            labels=labels_here,
                                            save=False,
                                            custom_whspace=0.15,
                                            filename=self.path + key + '_cornerplot' + self.suffix,
                                            dpi=150,
                                            linestyle='-',
                                            **kwargs
                                            )
                except:
                    print('Could not plot the corner plot for branch', key)
                    fig = None

                
                if self.mixture_here:
                    ndim = chain.shape[-1]
                    dpgmm = dgpmm_dict[key]
                    covs = dpgmm.covariances_
                    means = dpgmm.means_

                    # Plot the ellipses
                    if dpgmm.n_components > 1:
                        for i in range(dpgmm.n_components):
                            cov = covs[i]
                            mean = means[i]
                            v, w = np.linalg.eigh(cov)
                            v = 2. * np.sqrt(2.) * np.sqrt(v)

                            for j in range(cov.shape[0] // 2):
                                j = 2*j
                                u = w[j] / np.linalg.norm(w[j])
                                angle = np.arctan(u[j+1] / u[j])
                                angle = 180. * angle / np.pi

                                ell = mpl.patches.Ellipse(xy=mean, width=v[j], height=v[j+1], angle=180. + angle, color='k', facecolor='none', alpha=0.5)
                                fig.axes[ndim].add_patch(ell)
                                fig.axes[ndim].plot(mean[0], mean[1], 'x', color='k', alpha=0.5)
                if fig:
                    fig.savefig(self.path + key + '_cornerplot_T%.i' % temp  + self.suffix, dpi=150)
                    plt.close()

                if trace:
                    fig, axs = plt.subplots(self.sampler.ndims[key], 1, sharex=True)
                    fig.set_size_inches(20, 20)

                    for i in range(ndims):
                        for walk in range(self.nw):
                            axs[i].plot(samples_here[:, temp, walk, :, i], ls='-', alpha=0.7, lw=1)

                        axs[i].set_ylabel(self.labels[key][i])

                        if truths_here[i] is not None:
                            axs[i].axhline(truths_here[i], color='k', ls='--', lw=2)

                    fig.savefig(self.path + key + '_traceplot_T%.i' % temp + self.suffix, dpi=150)
                    plt.close()
                
                if covs:
                    steps = np.arange(100, nsteps, 100, dtype=int)
                    colors = pysco.plot.get_colorslist(colors='colors6')

                    fig, axs = plt.subplots(ndims, 1, figsize=(20, 5*ndims), sharex=True)

                    for n in steps:
                        cov = np.cov(np.squeeze(samples_here)[n, temp].T) 
                        for i in range(ndims):
                            axs[i].plot(n, np.sqrt(cov[i, i]), marker='x', color=colors[temp])
                    
                    for i in range(ndims):
                        axs[i].set_ylabel(labels_here[i])
        

                    plt.xlabel('Number of steps')
                    plt.savefig(self.path + key + '_covariance_T%.i' % temp + self.suffix, dpi=150)
                    plt.close()

        if self.converter is not None:
            try:
                new_samples = get_numpy(self.converter(samples)[0])
                #todo look at the shape of the samples
                nsteps, ntemps, nw = new_samples.shape[:3]
                new_samples = new_samples.reshape(nsteps, ntemps, nw, -1)
                new_samples_flat = new_samples[:, 0].reshape(nsteps * nw, -1)
                fig = pysco.plot.corner(new_samples_flat,
                                        # truths=truths_here,
                                        # labels=labels_here,
                                        save=True,
                                        custom_whspace=0.15,
                                        filename=self.path + 'converter_cornerplot' + self.suffix,
                                        dpi=150,
                                        linestyle='-',
                                        **kwargs
                                        )
                plt.close()
                
                fig, axs = plt.subplots(new_samples.shape[-1], 1, sharex=True)
                fig.set_size_inches(20, 20)

                for i in range(new_samples.shape[-1]):
                    for walk in range(nw):
                        axs[i].plot(new_samples[:, 0, walk, i], ls='-', alpha=0.7, lw=1)

                    axs[i].set_ylabel('Parameter ' + str(i))

                    
                fig.savefig(self.path + 'converter_traceplot' + self.suffix, dpi=150)
                plt.close()
            except:
                print('Could not plot the converter corner plot.')
                pass

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
            
            self.acceptance_moves[label] = np.concatenate((self.acceptance_moves[label], np.atleast_1d(np.mean(move.acceptance_fraction[0]))))

            nanmask = np.isnan(self.acceptance_moves[label])

            plt.plot(steps[~nanmask], self.acceptance_moves[label][~nanmask], label=label, marker='x', lw=1)

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
                self.rj_acceptance_moves[label] = np.concatenate((self.rj_acceptance_moves[label], np.atleast_1d(np.mean(move.acceptance_fraction[0]))))

                nanmask = np.isnan(self.rj_acceptance_moves[label])

                plt.plot(steps[~nanmask], self.rj_acceptance_moves[label][~nanmask], label=label, marker='x', lw=1)

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
        inds = self.sampler.get_inds(discard=0, thin=1)
        Npoints = np.exp(np.linspace(np.log(min(100, self.sampler.iteration)), np.log(self.sampler.iteration), N)).astype(int)

        extra_keys = ['logl']
        extra_colors = ['lightgray']
        if self.converter is not None:
            extra_keys.append('converted')
            extra_colors.append('darkgray')

        for key, color in zip(extra_keys + list(samples.keys()), extra_colors + pysco.plot.get_colorslist(colors='colors10')):

            if key == 'logl':
                chain = self.sampler.get_log_like(discard=0, thin=1)
                nsteps, nt, nw = chain.shape
                key = r'$\log{\mathcal{L}}$'   
                ls = '--'

            elif key == 'converted':
                chain = get_numpy(self.converter(samples)[0])
                nsteps, nt, nw = chain.shape[:3]
                chain = chain.reshape(nsteps, nt, nw, -1)
                key = 'converted'
                ls = '-.'

            else:
                chain = samples[key]
                nsteps, nt, nw, nleaves, ndim = chain.shape
                chain = chain.reshape(nsteps, nt, nw, nleaves * ndim)
                ls = '-'
            ntemps = chain.shape[1] if all_T else 1
            tau = np.empty(shape=(len(Npoints), ntemps, chain.shape[-1]))

            try:
                for i, N in enumerate(Npoints):
                    tau[i] = get_integrated_act(chain[:N, :ntemps], average=True, **kwargs)

                for temp in range(ntemps):
                    where_max = np.argmax(tau[:, temp], axis=-1)

                    to_plot =np.max(tau[:, temp], axis=-1)

                    if np.any(np.isnan(to_plot)):
                        warnings.warn(f"Auto-correlation time for branch {key} and temperature {temp} contains NaN values.")
                    else:
                        plt.loglog(Npoints, to_plot, color=color, marker='o', ls=ls, alpha=0.9, label=key + fr' - $T_{temp}$')

                    toplim = max(toplim, 5 * max(tau[-1, temp, where_max]))
                
            except:
                warnings.warn(f"Could not compute the auto-correlation times for the branch {key}.")

        plt.loglog(Npoints, Npoints / 50, label=r'$\tau = N/50$', linestyle='--', color='black')

        plt.xlabel(r'$N_{\rm steps}$')
        plt.ylabel(r'$\tau$')

        plt.ylim(0, toplim)
        plt.legend()

        fig.savefig(self.path + 'act_evolution' + self.suffix, dpi=150)
        plt.close()
    
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

    def plot_logl_betas(self, betas, logl):
        """
        Plots the evolution of log-likelihood values for each temperature.

        Args:
            betas (numpy.ndarray): Array of inverse temperatures.
            logl (numpy.ndarray): Array of log-likelihood values.

        Returns:
            None
        """
        fig = plt.figure()
        for temp in range(self.nt):
            plt.loglog(betas[-1, temp], np.mean(logl[:, temp]), '.', c=self.tempcolors[temp], label=f'$T_{temp}$')

        if self.true_logl is not None:
            plt.axhline(self.true_logl, color='k', ls='--', lw=2)

        logZ, dlogZ = stepping_stone_log_evidence(betas[-1], logl)
        
        plt.ylabel(r'$\log{\mathcal{L}}$')
        plt.xlabel(r'$\beta$')
        plt.title(r'$\log{\mathcal{Z}} = %.2f \pm %.2f$' % (logZ, dlogZ))
        fig.savefig(self.path + 'loglike_betas' + self.suffix, dpi=150)

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
    
    def __init__(self, autocorr_multiplier=50, verbose=False, N=0, n_skip=0, ess=None, transform_fn=None, n_iters=30, diff=0.1, start_iteration=0):
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

        #likelihood checks. Adapted from the SearchConvergenceStopping criterion
        self.n_iters = n_iters
        self.diff = diff
        self.start_iteration = start_iteration

        self.iters_consecutive = 0
        self.past_like_best = -np.inf
        

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

        # get best Likelihood so far
        like_best = sampler.get_log_like(discard=self.start_iteration).max()

        # compare to last
        # if it is less than diff change it passes
        if np.abs(like_best - self.past_like_best) < self.diff:
            self.iters_consecutive += 1

        else:
            # if it fails reset iters consecutive
            self.iters_consecutive = 0

            # store new best
            self.past_like_best = like_best

        samples = sampler.get_chain(discard=0, thin=1)

        if self.transform_fn is not None:
            samples = self.transform_fn(samples)

        tau = {}
        for name in samples.keys():
            chain = samples[name]
            nsteps, ntemps, nw, nleaves, ndims = chain.shape
            chain = chain.reshape(nsteps, ntemps, nw, nleaves * ndims)
            tau[name] = get_integrated_act(chain, average=True)

        #tau = get_integrated_act(samples)

        use_to_stop = []
        for name, values in tau.items():
            if np.all(np.isfinite(values)):
                use_to_stop.append(name)
                

        if self.time > 0:
            iteration = sampler.iteration

            if iteration >= self.when_to_stop:
                return True
            
            elif iteration < self.when_to_stop and np.isfinite(self.when_to_stop):
                return False

            finish = []

            for name, values in tau.items():
                if name not in use_to_stop:
                    continue
                converged = np.all(tau[name] * self.autocorr_multiplier < iteration)
                converged &= np.all(
                    np.abs(self.old_tau[name] - tau[name]) / tau[name] < 0.01
                )

                finish.append(converged)

            if np.all(finish) and (self.iters_consecutive >= self.n_iters):
                # if we have passes the number of iters necessary, return True and reset
                self.iters_consecutive = 0

                stop = True
                
                taus_all = []

                for name in tau.keys():
                    tau_here = np.max(tau[name])
                    if np.isfinite(tau_here):
                        taus_all.append(tau_here)
                
                tau_max = np.max(taus_all)

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
    

class SamplesLoader():
    def __init__(self, path, transform_fn=None):

        backend_path = find_files(path, 'h5')[0] if os.path.isdir(path) else path
        self.backend = backend_path
        self.transform_fn = transform_fn

    @property
    def backend(self):
        return self._backend
    
    @backend.setter
    def backend(self, backend):
        self._backend = HDFBackend(backend)

    @property
    def transform_fn(self):
        return self._transform_fn

    @transform_fn.setter
    def transform_fn(self, transform_fn):
        self._transform_fn = transform_fn


    def load(self, ess=1e4, squeeze=False, leaves_to_ndim=False):
        """
        Load the samples from the backend.

        Args:
            ess (float): Effective sample size. Default is 1e4.
            squeeze (bool): Whether to 'squeeze' the samples. Default is False.

        Returns:
            if squeeze == True and there is only one branch:
                samples_out (array): 2D array containing the samples of the single branch.
                logL (array): 1D array containing the log likelihood.
                logP (array): 1D array containing the log posterior
            else:
                samples_out (dict): Dictionary containing the samples for each branch, 
                                    the log likelihood and the log posterior.
        """

        if not hasattr(self, 'discard') or not hasattr(self, 'thin'):
            self.compute_discard_thin(ess=ess)
        
        discard, thin = self.discard, self.thin

        samples = self.backend.get_chain(discard=discard, thin=thin) #todo: this is not addressing the nans in the RJ chains

        samples_out = {}

        logP = self.backend.get_log_posterior(discard=discard, thin=thin)[:, 0].flatten()
        logL = self.backend.get_log_like(discard=discard, thin=thin)[:, 0].flatten()

        #samples_out['logP'] = logP
        #samples_out['logL'] = logL

        branches = self.backend.branch_names
        for branch in branches:
            samples_here = samples[branch]
            nsteps, ntemps, nw, nleaves, ndim = samples_here.shape
            if self.transform_fn is not None:
                if not isinstance(self.transform_fn, dict):
                    samples_here = self.transform_fn.both_transforms(samples_here)  
                else:  
                    samples_here = self.transform_fn[branch].both_transforms(samples_here)
            
            if leaves_to_ndim:
                samples_here = samples_here[:, 0].reshape(-1, nleaves*ndim) # take only the zero temperature chain
            else:
                samples_here = samples_here[:, 0].reshape(-1, ndim) # take only the zero temperature chain
            samples_out[branch] = samples_here
        
        if len(branches) == 1 and squeeze:
            return samples_out[branch], logL, logP
        else:
            return samples_out, logL, logP
        
    def get_leaves(self, ess=1e4):
        """
        Get the number of leaves for each temperature.

        Returns:
            dict: Dictionary containing the number of leaves for each temperature.
        """
        if not hasattr(self, 'discard') or not hasattr(self, 'thin'):
            self.compute_discard_thin(ess=ess)
        
        discard, thin = self.discard, self.thin
        return self.backend.get_nleaves(discard=discard, thin=thin)
    
    @pysco.utils.timeit
    def compute_discard_thin(self, ess=1e4):
        """
        Compute the number of samples to discard and thin.

        Args:
            ess (float): Effective sample size. Default is 1e4.
        Returns:    
            None
        """
        samples = self.backend.get_chain()
        tau = {}
        for name in samples.keys():
            chain = samples[name]
            nsteps, ntemps, nw, nleaves, ndims = chain.shape
            chain = chain.reshape(nsteps, ntemps, nw, nleaves * ndims)
            tau[name] = get_integrated_act(chain, average=True, fast=True)
        
        taus_all = []

        for name in tau.keys():
            tau_here = np.max(tau[name])
            if np.isfinite(tau_here):
                taus_all.append(tau_here)
        
        self.thin = int(np.max(taus_all))
        print("Number of steps: ", nsteps)

        ess = int(ess)
        N_keep = int(np.ceil(ess * self.thin / nw))
        print("Number of samples to keep: ", N_keep)
        self.discard = max(5000, self.backend.iteration - N_keep)
    
    def make_dataframe(self, labels=None, samples=None, ess=1e4):
        """
        Make a pandas DataFrame from the samples.

        Args:
            labels (dict): Dictionary containing the labels for each parameter in each branch.
            samples (dict): Dictionary containing the samples for each branch. Default is None.
            ess (float): Effective sample size. Default is 1e4.
        
        Returns:
            dict: Dictionary containing the pandas DataFrame for each branch.
        """

        labels = self.labels if hasattr(self, 'labels') else labels
        assert labels is not None, "Labels must be provided."

        if samples is None:
            samples = self.load(ess=ess)

        if not isinstance(labels, dict):
            labels = {'tmp': labels}

        if not isinstance(samples, dict):
            samples = {branch: samples for branch in labels.keys()}

        dfs = {}
        for branch, samples_here in samples.items():
            if branch in ['logP', 'logL']:
                continue

            df = pd.DataFrame(samples_here, columns=labels[branch])

            if 'logP' in samples.keys():
                df['log_posterior'] = samples['logP']
            
            if 'logL' in samples.keys():
                df['log_likelihood'] = samples['logL']

            dfs[branch] = df

        return dfs if len(dfs) > 1 else dfs[branch]
        

        

        

            




