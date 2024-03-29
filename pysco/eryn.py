import sys, os
import numpy as np
from distutils.spawn import find_executable
import pysco
import matplotlib as mpl
from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt


def check_latex():
    if find_executable('latex'):
        mpl.rcParams['text.usetex']=True
        return
    else:
         mpl.rcParams['text.usetex']=False
         return

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
        

def plot_diagnostics(samp, path, ndim, truths, labels, transform_all_back, acceptance_all, acceptance_moves=None, rj_acceptance_all=None, rj_acceptance_moves=None, rj_branches=[], nleaves_min=None, nleaves_max=None, moves_names=None, rj_moves_names=None, use_chainconsumer=False, **kwargs):
    """
    Plots the diagnostics of the given sample.

    Parameters:
    - samp: The sample object containing the chains.
    - path: The path to save the plots.
    - plot_names: A list of the names of the plots to generate.
    - ndim: A dictionary containing the number of dimensions for each branch.
    - truths: A dictionary containing the true values for each parameter.
    - labels: A dictionary containing the labels for each parameter.
    - transform_all_back: A dictionary containing the transformation functions for each branch.
    - rj: A boolean indicating whether to plot reversible jump (RJ) diagnostics. Default is False.
    - rj_branches: A dictionary containing the names of the RJ branches. Default is None.
    - nleaves_min: A dictionary containing the minimum number of leaves for each RJ branch. Default is None.
    - nleaves_max: A dictionary containing the maximum number of leaves for each RJ branch. Default is None.
    - moves_names: A list of the names of the moves. Default is None.

    Returns:
    None
    """
    nwalkers = samp.nwalkers
    ntemps = samp.ntemps
    steps = np.arange(samp.iteration)
    myred = '#9A0202' 
    tempcolors = pysco.plot.get_colors_from_cmap(ntemps, cmap='inferno', reverse=False)    
                        
    for key in samp.branch_names:
        check_latex()
        chain = get_clean_chain(samp.get_chain(discard=int(samp.iteration*0.3), thin=1)[key], ndim=ndim[key])
        chain = transform_all_back[key].transform_base_parameters(chain)

        inds = samp.get_inds(discard=int(samp.iteration*0.3))
        logP = samp.get_log_posterior(discard=int(samp.iteration*0.3), thin=1)[:,0]

        if use_chainconsumer:
            try:
                fig = pysco.plot.chainplot(samples=chain, 
                                        truths=truths[key], 
                                        labels=labels[key], 
                                        names=key,
                                        #logP=logP[inds_mask].flatten(),
                                        filename=path + 'diagnostic/' + key,
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
                                        filename=path + 'diagnostic/' + key + '_cornerplot', 
                                        dpi=150
                                        )
                plt.close()

        else:
            fig = pysco.plot.corner(chain, 
                                    truths=truths[key], 
                                    labels=labels[key], 
                                    save=True, 
                                    custom_whspace= 0.15, 
                                    filename=path + 'diagnostic/' + key + '_cornerplot', 
                                    dpi=150
                                    )
            plt.close()

        check_latex()
        fig, axs = plt.subplots(ndim[key], 1)
        fig.set_size_inches(20, 20)
        for i in range(ndim[key]):
            for walk in range(nwalkers):
                chain = transform_all_back[key].transform_base_parameters(samp.get_chain(discard=int(samp.iteration*0.3), thin=1)[key])
                axs[i].plot(chain[:, 0, walk, :, i], color = 'k', ls='-', alpha = 0.2)
                axs[i].set_ylabel(labels[key][i])

                if truths[key][i] is not None:
                    axs[i].axhline(truths[key][i], color=myred)

        fig.savefig(path + 'diagnostic/' + key + '_traceplot', dpi=150)
        plt.close()

        #* Plotting RJ diagnostics
        if key in rj_branches:
            plot_leaves_hist(samp, key, path, tempcolors, nleaves_min, nleaves_max)

    #* Plotting logL evolution with the number of steps
    plot_logl(samp, path, nwalkers)

    #* Plotting acceptance fraction evolution with the number of steps
    plot_acceptance(steps, path, acceptance_all, acceptance_moves, moves_names=moves_names)

    if rj_acceptance_all is not None:
        plot_acceptance(steps, path, rj_acceptance_all, rj_acceptance_moves, moves_names=rj_moves_names, suffix='_rj')


def plot_leaves_hist(samp, key, path, tempcolors, nleaves_min, nleaves_max):
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

    fig.text(0.07, 0.083, f"Step: {samp.iteration}", ha='left', va='top', fontfamily='fantasy', c='red')
    #plt.title(key)
    fig.savefig(path + 'diagnostic/leaves_' + key, dpi=150)
    plt.close()

def plot_logl(samp, path, nwalkers):
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
    maxlogl = np.max(logl[:,0,:], axis=0)
    for walk in range(nwalkers):
        plt.plot(logl[:, 0, walk] - maxlogl[walk], color='k', ls='-', alpha=0.2, lw=1)
    plt.ylabel(r'$\log{\mathcal{L}}$')
    fig.savefig(path + 'diagnostic/loglike', dpi=150)
    plt.close()


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