# -*- coding: utf-8 -*-

from functools import wraps
from importlib.machinery import SourceFileLoader
import os
import warnings
from distutils.spawn import find_executable
import warnings
from distutils.spawn import find_executable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib import ticker
from cycler import cycler

import numpy as np
import pandas as pd
import chainconsumer

try: 
    pysco_file = str(__file__)
    corner_file = os.getenv('CORNER_PATH') + '/src/corner/__init__.py'
    
    c = SourceFileLoader('corner', corner_file).load_module() #custom version of corner, allow to choose the color of the quantiles
except:
    import corner as c
    warnings.warn('WARNING: imported standard corner.py')

    __all__ = [
        'which_corner',
        'check_latex',
        'default_plotting',
        'reset_rc',
        'set_ticker',
        'custom_corner',
        'corner',
        'set_color_cycle_from_cmap',
        'get_colors_from_cmap',
        'custom_color_cycle',
        'get_colorslist',
        'to_pandas',
        'chainplot'
    ]


#---- Plotting Stuff ----#

def which_corner():
    print('You are using the corner package located at' + str(corner.__file__))

def check_latex():
    if find_executable('latex'):
        mpl.rcParams['text.usetex']=True
        return
    else:
         mpl.rcParams['text.usetex']=False
         return

def default_plotting(backcolor='white', frontcolor='black'):
    """
    Set the default plotting parameters for matplotlib.

    Parameters:
    - backcolor (str): The background color of the plot. Default is 'white'.
    - frontcolor (str): The foreground color of the plot. Default is 'black'.
    """

    default_rcParams = {
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': 'cmr10',
        'font.weight':'medium',
        'mathtext.fontset': 'cm',
        'text.latex.preamble': r"\usepackage{amsmath}",
        'font.size': 14,
        'figure.figsize': (7, 7),
        'figure.titlesize': 'large',
        'axes.formatter.use_mathtext': True,
        'axes.formatter.limits': [-2, 4],
        'axes.titlesize': 'large',
        'axes.labelsize': 'large',
        'xtick.top': True,
        'xtick.major.size': 5,
        'xtick.minor.size': 3,
        'xtick.major.width': 0.8,
        'xtick.minor.visible': True,
        'xtick.direction': 'in',
        'xtick.labelsize': 'medium',
        'ytick.right': True,
        'ytick.major.size': 5,
        'ytick.minor.size': 3,
        'ytick.major.width': 0.8,
        'ytick.minor.visible': True,
        'ytick.direction': 'in',
        'ytick.labelsize': 'medium',
        'legend.frameon': True,
        'legend.framealpha': 1,
        'legend.fontsize': 'medium',
        'legend.scatterpoints' : 3,
        'lines.color': 'k',
        'lines.linewidth': 2,
        'patch.linewidth': 1,
        'hatch.linewidth': 1,
        'grid.linestyle': 'dashed',
        'savefig.dpi' : 200,
        'savefig.format' : 'pdf',
        'savefig.bbox' : 'tight',
        'savefig.transparent' : True,
    }
    
    # adjust colors
    default_rcParams['text.color'] = frontcolor
    default_rcParams['axes.labelcolor'] = frontcolor
    default_rcParams['axes.edgecolor'] = frontcolor
    default_rcParams['xtick.color'] = frontcolor
    default_rcParams['ytick.color'] = frontcolor
    default_rcParams['axes.frontcolor'] = backcolor
    default_rcParams['figure.frontcolor'] = backcolor
    default_rcParams['legend.frontcolor'] = backcolor
    default_rcParams['legend.edgecolor'] = frontcolor
    default_rcParams['legend.labelcolor'] = frontcolor
    default_rcParams['grid.color'] = frontcolor
    default_rcParams['lines.color'] = frontcolor

    plt.rcParams.update(default_rcParams)

    custom_color_cycle()
    check_latex()

def reset_rc():
    mpl.rcParams.update(mpl.rcParamsDefault)

def set_ticker():
    ticker.ScalarFormatter(useMathText=plt.rcParams['text.usetex'])

def custom_corner(function):
    """
    A decorator function that customizes the corner plot generated by the input function.

    Parameters:
    - function: The function that generates the corner plot.

    Returns:
    - The customized corner plot.

    Usage:
    @custom_corner
    def generate_corner_plot(*args, **kwargs):
        # Code to generate the corner plot

    Example:
    generate_corner_plot(data, labels=['x', 'y', 'z'], color='blue', save=True, filename='corner_plot.png')
    """
    
    @wraps(function)
    def wrapper(*args, **kwargs):
        """
        A wrapper function that customizes matplotlib settings and calls the specified function with the given arguments.

        Parameters:
        *args: positional arguments to be passed to the specified function.
        **kwargs: keyword arguments to be passed to the specified function.

        Returns:
        fig: the figure object returned by the specified function.

        Raises:
        None.
        """

        corner_rcParams = {
            'xtick.top': False,
            'xtick.major.size': 3.5,
            #'xtick.major.width': 1.,
            'xtick.minor.visible': False,
            'xtick.direction': 'out',
            'ytick.right': False,
            'ytick.major.size': 3.5,
            #'ytick.major.width': 1.,
            'ytick.minor.visible': False,
            'ytick.direction': 'out',
            'ytick.labelsize': 'medium',
            'lines.linewidth': 1.5,
            }
        
        backcolor = plt.rcParams['figure.facecolor']
        frontcolor = plt.rcParams['text.color']

        # customize matplotlib
        if 'custom_rc' in kwargs.keys():
            custom_rc = kwargs.pop('custom_rc')
        else:
            custom_rc = True
        if custom_rc:
            plt.rcParams.update(corner_rcParams)

        if 'rcParams' in kwargs.keys():
            rcParams = kwargs.pop('rcParams')
            plt.rcParams.update(rcParams)

        defaults_kwargs = dict(
            bins=50, 
            smooth=0.5,
            title_kwargs=dict(fontsize=16), 
            color='#3f90da',
            truth_color=frontcolor,
            quantiles=[0.05, 0.5, 0.95],
            linestyle='-',
            plot_median=False,
            marginal_type='hist',
            quantiles_color=None, 
            show_titles = True, 
            title_fmt='.2e',
            levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
            plot_density=False, 
            plot_datapoints=False, 
            fill_contours=True,
            max_n_ticks=5, 
            use_math_text=True, 
            custom_whspace=0.05,
        )

        
        _kwargs = kwargs.copy()
        
        save = False
        if 'save' in kwargs.keys():
            #save = True
            save = kwargs.pop('save')
            try:
                filename = kwargs['filename']
                kwargs.pop('filename')
            except:
                print('"filename" not provided, defaulted to cornerplot.pdf')
                filename = './cornerplot'

        hist_kwargs = dict(
                    histtype = _kwargs['histtype'] if 'histtype' in _kwargs.keys() else 'step',
                    edgecolor = _kwargs['color'] if 'color' in _kwargs.keys() else defaults_kwargs['color'],
                    lw = 1.5,
                    density=True, 
                )
        
        keys = _kwargs.keys()
        for key in keys:
            if key in defaults_kwargs.keys():
                defaults_kwargs.pop(key)

            if key in hist_kwargs.keys():
                hist_kwargs[key] = kwargs.pop(key)
            
        alpha = _kwargs['histalpha'] if 'histalpha' in _kwargs.keys() else 0.1
        defaults_kwargs.update(kwargs)
        fc = to_rgba(defaults_kwargs['color'], alpha=alpha)
        hist_kwargs['fc'] = fc
        defaults_kwargs['hist_kwargs'] = hist_kwargs
        kwargs = defaults_kwargs

        fig = function(*args, **kwargs)

        if save:
            fig.savefig(filename)
        
        default_plotting(backcolor=backcolor, frontcolor=frontcolor)

        return fig

    return wrapper

@custom_corner
def corner(*args, **kwargs):
    fig =  c.corner(*args, **kwargs)
    return fig

def set_color_cycle_from_cmap(cmap=None):
    '''
    use custom colormap as standard colorcycle. 
    '''
    if cmap:
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", cmap.colors)
    else:
         plt.rcParams["axes.prop_cycle"] =  plt.rcParamsDefault["axes.prop_cycle"]
    
def get_colors_from_cmap(N, cmap='viridis', reverse=False):
    cmap = mpl.colormaps[cmap]
    colors = cmap(np.linspace(0.3, 1., N))
    if reverse:
        colors = colors[::-1]
    return colors

def custom_color_cycle(colors='colors6', linestyles=['-'], skip=0, lsfirst=False):
    '''
    setup a custom matplotlib color cycler.
    `colors6`, `colors8`, `colors10` refer to the results of arXiv:2107.02270.
    `fancy` contains a color palette I like (still work in progress).
    '''

    baselinestyles = ['-', '--', '-.', ':']

    if isinstance(colors, str):
        colors = get_colorslist(colors)

    elif isinstance(colors, list):
        pass
    else:
        raise ValueError('provide `colors` as a list of color IDs or a key of `basecolors`')
    
    if isinstance(linestyles, int):
        linestyles = baselinestyles[:linestyles]
    
    if colors is not None:
        if lsfirst:
            mycycler = cycler(color=colors[skip:]) * cycler(linestyle=linestyles)
        else:
            mycycler = cycler(linestyle=linestyles) * cycler(color=colors[skip:])

        plt.rcParams["axes.prop_cycle"] =  mycycler
    
    else:
        plt.rcParams["axes.prop_cycle"] =  plt.rcParamsDefault["axes.prop_cycle"]


def get_colorslist(colors='colors6'):
    basecolors = {
        'colors6': ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"],
        'colors8': ["#1845fb", "#ff5e02", "#c91f16", "#c849a9", "#adad7d", "#86c8dd", "#578dff", "#656364"],
        'colors10': ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"],
        'fancy': ['#1c161c', '#324b58', '#088395', '#8ac5ad', '#a1f5a8']
    }

    assert colors in basecolors.keys()
    return basecolors[colors]


def to_pandas(samples, labels):
    """
    Converts the samples to a pandas DataFrame.

    Parameters:
    - samples (array): The samples to convert.
    - labels (list): The labels for the samples.

    Returns:
    - df (DataFrame): The samples as a pandas DataFrame.
    """
    df = pd.DataFrame(samples, columns=labels)
    return df

def chainplot(samples, labels, names, weights=None, logP=None, truths=None, return_obj=False, colors='#1d545c', plot_walks=False, **kwargs):
    """
    Plot the samples using the ChainConsumer package. This function is a wrapper around the ChainConsumer.plot function.
    The goal is to allow plotting multiple chains on the same canvas to compare different results.

    Arguments:
    - samples (list or array-like): The samples to plot. Each element in the list should be a 
                                    2D array with shape (n_samples, n_parameters). 
                                    The arrays will be converted to pandas DataFrames by `pysco.plot.to_pandas` 
                                    to be read by ChainConsumer.
    - labels (list): The labels for the samples.
    - names (list): The names of the chains.
    - weights (array-like, optional): The weights for each sample. If None, uniform weights will be used.
    - logP (array-like, optional): The log posterior values for each sample. If provided, the samples will be plotted as points.
    - truths (dict, list or array-like, optional): The true values for the parameters. If None, no truths will be plotted.
    - return_obj (bool, optional): If True, return the ChainConsumer object. If False, plot the results.
    - colors (str or list, optional): The colors to use for the chains. If a string, it should be a key for `pysco.plot.get_colorslist`.
                                      If a list, it should be a list of colors to use for each chain. If None, the default matplotlib 
                                      color cycle will be used.
    - plot_walks (bool, optional): If True, plot the walks of the chains in addition to the corner plot.
    - kwargs: Additional keyword arguments to pass to ChainConsumer.plot.

    Returns:
    - If return_obj is False, the function plots the results and returns the figure object(s).
    - If return_obj is True, the function returns the figure object(s) and the ChainConsumer object.

    Note:
    - This function requires the ChainConsumer package to be installed.
    - The samples will be converted to pandas DataFrames using `pysco.plot.to_pandas` before being passed to ChainConsumer.
    - The ChainConsumer object is used to create the corner plot and, if plot_walks is True, the walks plot.
    - The corner plot shows the parameter distributions and correlations between parameters.
    - The walks plot shows the evolution of the chains over iterations.

    Example usage:
    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from pysco.plot import chainplot

    # Generate some random samples
    n_samples = 1000
    n_parameters = 3
    samples = [np.random.randn(n_samples, n_parameters) for _ in range(3)]
    labels = ['Param 1', 'Param 2', 'Param 3']
    names = ['Chain 1', 'Chain 2', 'Chain 3']

    # Plot the samples
    chainplot(samples, labels, names, plot_walks=True)
    plt.show()
    ```
    """
    # Create a ChainConsumer object
    c = chainconsumer.ChainConsumer()

    # Check if samples is a list, if not convert it to a list with a single element
    if not isinstance(samples, list):
        samples = [samples]
        names = [names]
    # Check if colors is provided and is a list
    if isinstance(colors, list):
        assert len(colors) == len(samples), 'Provide a color for each chain'

    # Check if colors is provided and is a string, if so get the colors from the color list
    elif isinstance(colors, str):
        try:
            colors = get_colorslist(colors)
        except:
            colors = [colors]

    # If colors is not provided, use the default matplotlib color cycle
    else:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for sample, name, color in zip(samples, names, colors):
        # Convert the sample to a pandas DataFrame
        plot_point = False
        df = to_pandas(sample, labels)

        if logP is not None:
            df['log_posterior'] = logP
            plot_point = True
        else:
            plot_point = False

        if weights is None:
            weights = np.ones(shape=sample.shape[0])
            
        df['weight'] = weights
    
        chain = chainconsumer.Chain(samples=df, name=name, color=color, plot_point=plot_point, marker_style="*", marker_size=100, **kwargs)
        c.add_chain(chain)
    

    if truths is not None:
        # Check if truths is a dictionary, if not convert it to a dictionary using labels as keys
        if not isinstance(truths, dict):
            truths = dict(zip(labels, truths))

        # Add the truth values
        c.add_truth(chainconsumer.Truth(location=truths, color='k'))
        c.add_marker(location=truths, name="Truth", color="k", marker_style="x", marker_size=30)
    
    c.set_override(chainconsumer.ChainConfig(
        sigmas=[0, 1, 2],
        shade_gradient=1,
        shade_alpha=0.5,
        statistics='cumulative',
        summary_area=0.90,

    )
    )
    # Plot the results
    c.set_plot_config(
                chainconsumer.PlotConfig(
                    summarise=True,
                    tick_font_size=12,
                    label_font_size=plt.rcParams['font.size'],
                    summary_font_size=plt.rcParams['font.size'],
                    usetex=True,
                    serif=True,
                    dpi=plt.rcParams['savefig.dpi'],
                    )   
                    )
    
    plotter_kwargs = {
        #'chains': [name],
        'columns': labels,
        'figsize': plt.rcParams['figure.figsize'],
    }

    if 'filename' in kwargs:
        plotter_kwargs['filename'] = kwargs['filename'] + '_cornerplot'

    fig = c.plotter.plot(**plotter_kwargs)

    out = (fig,)

    if plot_walks:
        plotter_kwargs['filename'] = kwargs['filename'] + '_walks'
        plotter_kwargs['figsize'] = (20, 20)
        plotter_kwargs['plot_weights'] = kwargs['plot_weights'] if 'plot_weights' in kwargs.keys() else False
        plotter_kwargs['plot_posterior'] = kwargs['plot_posterior'] if 'plot_posterior' in kwargs.keys() else False
        plotter_kwargs['convolve'] = kwargs['convolve'] if 'convolve' in kwargs.keys() else 100

        fig_walks = c.plotter.plot_walks(**plotter_kwargs)
        
        out += (fig_walks,)
    
    if return_obj:
        out += (c,)

    return out
