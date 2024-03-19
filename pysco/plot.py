
from functools import wraps
from importlib.machinery import SourceFileLoader
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib import ticker
from cycler import cycler
import numpy as np

try: 
    pysco_file = str(__file__)
    corner_file = os.getenv('CORNER_PATH') + '/src/corner/__init__.py'
    
    c = SourceFileLoader('corner', corner_file).load_module() #custom version of corner, allow to choose the color of the quantiles
except:
    import corner as c
    print('WARNING: imported standard corner.py')


#---- Plotting Stuff ----#

def which_corner():
    print('You are using the corner package located at' + str(corner.__file__))

def default_plotting():
    default_rcParams = {
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': 'cmr10',
        'font.weight':'medium',
        'mathtext.fontset': 'cm',
        'text.latex.preamble': r"\usepackage{amsmath}",
        'font.size': 14,
        'figure.figsize': (5, 5),
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

    plt.rcParams.update(default_rcParams)

def reset_rc():
    mpl.rcParams.update(mpl.rcParamsDefault)

def set_ticker():
    ticker.ScalarFormatter(useMathText=plt.rcParams['text.usetex'])

def custom_corner(function):

    @wraps(function)
    def wrapper(*args, **kwargs):
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
            bins=50, smooth=0.5,
            title_kwargs=dict(fontsize=16), color='#366c81',
            truth_color='#9A0202', #'#990000'
            quantiles=[0.05, 0.5, 0.95], 
            quantiles_color='k', 
            show_titles = True, title_fmt='.2e',
            levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
            plot_density=False, plot_datapoints=True, fill_contours=True,
            max_n_ticks=5, use_math_text=True, custom_whspace=0.05)
        

        _kwargs = kwargs.copy()
        
        save = False
        if 'save' in kwargs.keys():
            save = True
            kwargs.pop('save')
            try:
                filename = kwargs['filename']
                kwargs.pop('filename')
            except:
                print('"filename" not provided, defaulted to cornerplot.pdf')
                filename = './cornerplot'

        hist_kwargs = dict(
                    histtype='stepfilled',
                    edgecolor = 'k',
                    lw = 1.3,
                    density=True
                )
        
        keys = _kwargs.keys()
        for key in keys:
            if key in defaults_kwargs.keys():
                defaults_kwargs.pop(key)

            if key in hist_kwargs.keys():
                hist_kwargs[key] = kwargs[key]
                kwargs.pop(key)
            

        defaults_kwargs.update(kwargs)
        fc = to_rgba(defaults_kwargs['color'], alpha=0.5)
        hist_kwargs['fc'] = fc
        defaults_kwargs['hist_kwargs'] = hist_kwargs
        kwargs = defaults_kwargs

        fig = function(*args, **kwargs)

        if save:
            fig.savefig(filename)
        
        default_plotting()

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

def custom_color_cycle(colors='colors10', linestyles=['-'], skip=0, lsfirst=False):
    '''
    setup a custom matplotlib color cycler.
    `colors6`, `colors8`, `colors10` refer to the results of arXiv:2107.02270.
    `fancy` contains a color palette I like (still work in progress).
    '''
    basecolors = {
        'colors6': ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"],
        'colors8': ["#1845fb", "#ff5e02", "#c91f16", "#c849a9", "#adad7d", "#86c8dd", "#578dff", "#656364"],
        'colors10': ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"],
        'fancy': ['#1c161c', '#324b58', '#088395', '#8ac5ad', '#a1f5a8']
    }
    baselinestyles = ['-', '--', '-.', ':']

    if isinstance(colors, str):
        assert colors in basecolors.keys()
        colors = basecolors[colors]

    elif isinstance(colors, str):
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