from functools import wraps
from importlib.machinery import SourceFileLoader
import os

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib import ticker
import numpy as np

#ry: 
#print(os.getcwd())
pysco_file = str(__file__)
corner_file = pysco_file[:-19] + 'corner.py/src/corner/__init__.py'
print(corner_file)
corner = SourceFileLoader('corner', corner_file).load_module() #custom version of corner, allow to choose the color of the quantiles
print('imported custom corner.py (' + corner_file + ')')
# except:
#     import corner
#     print('imported standard corner.py')


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

        plt.rcParams.update(corner_rcParams)

        # truth_color = 
        # color = 
        # quantiles_color = '#254B5A' #'#20404D'

        defaults_kwargs = dict(
            bins=50, smooth=0.5,
            title_kwargs=dict(fontsize=16), color='#366c81',
            truth_color='#990000',
            quantiles=[0.05, 0.5, 0.95], 
            #title_quantiles=[0.05, 0.5, 0.95], 
            quantiles_color='k', 
            show_titles = True, title_fmt='.2e',
            levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
            plot_density=False, plot_datapoints=True, fill_contours=True,
            max_n_ticks=5, use_math_text=True)
        

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
                    lw = 1.3
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

    return wrapper

@custom_corner
def plot_corner(*args, **kwargs):
    fig =  corner.corner(*args, **kwargs)
    return fig

def color_cycle(cmap=None):
    '''
    use custom colormap as standard colorcycle
    '''
    if cmap:
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", cmap.colors)
    else:
         plt.rcParams["axes.prop_cycle"] =  plt.rcParamsDefault["axes.prop_cycle"]
    
