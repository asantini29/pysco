# -*- coding: utf-8 -*-

from functools import wraps
from pathlib import Path
import warnings
from shutil import which as find_executable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, LinearSegmentedColormap
from matplotlib import ticker
from cycler import cycler

import numpy as np
import pandas as pd

try:
    import chainconsumer
    chain_here = True
except ImportError:
    warnings.warn('WARNING: ChainConsumer not found. chainplot will not work.')
    chain_here = False

import corner as c

__all__ = [
    'which_corner',
    'check_latex',
    'default_plotting',
    'reset_rc',
    'set_ticker',
    'set_colors',
    'custom_corner',
    'corner',
    'set_color_cycle_from_cmap',
    'get_colors_from_cmap',
    'custom_color_cycle',
    'get_colorslist',
    'get_cmap',
    'to_pandas',
    'chainplot',
    'PALETTES',
    'CMAPS',
]


# ---- Colour palettes & colormaps ---- #

PALETTES = {
    # arXiv:2107.02270 CVD-friendly sets
    'colors6':  ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"],
    'colors8':  ["#1845fb", "#ff5e02", "#c91f16", "#c849a9", "#adad7d", "#86c8dd", "#578dff", "#656364"],
    'colors10': ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"],
    'fancy':    ['#1c161c', '#324b58', '#088395', '#8ac5ad', '#a1f5a8'],

    # CVD-safe categorical (8 colours)
    'cat': [
        "#1B6BBF",  # title blue
        "#E8832A",  # warm orange
        "#7B3FA0",  # accent purple
        "#2A9E8F",  # teal
        "#C94040",  # brick red
        "#F0B429",  # amber yellow
        "#5AAADE",  # light blue
        "#A0522D",  # sienna
    ],

    # Sequential anchor colours (deep navy → vivid turquoise)
    'seq': [
        "#033270",  # deep navy
        "#0B5896",  # dark ocean blue
        "#0A85B8",  # medium blue
        "#08ACBE",  # blue-teal
        "#06CFC2",  # teal
        "#04E8C8",  # turquoise
    ],

    # Diverging anchor colours (deep navy ↔ vivid turquoise, pale centre)
    'div': [
        "#033270",  # deep navy          ← pole
        "#0A85B8",  # medium blue
        "#A8D8EA",  # pale sky            near-centre
        "#A8EDE6",  # pale mint           near-centre
        "#06CFC2",  # teal
        "#04E8C8",  # vivid turquoise    ← pole
    ],
}
"""All built-in colour palettes, keyed by short name."""

# Build continuous colormaps from the sequential & diverging anchors
_seq_cmap   = LinearSegmentedColormap.from_list('pysco_seq', PALETTES['seq'])
_seq_cmap_r = _seq_cmap.reversed()
_div_cmap   = LinearSegmentedColormap.from_list('pysco_div', PALETTES['div'])
_div_cmap_r = _div_cmap.reversed()

CMAPS = {
    'seq':   _seq_cmap,
    'seq_r': _seq_cmap_r,
    'div':   _div_cmap,
    'div_r': _div_cmap_r,
}
"""Continuous colormaps derived from the sequential and diverging palettes."""

# Register with matplotlib so plt.get_cmap('pysco_seq') etc. work
for _cmap in CMAPS.values():
    try:
        mpl.colormaps.register(_cmap)
    except ValueError:
        pass  # already registered (e.g. module reloaded)


def get_cmap(name='seq'):
    """Return one of the pysco colormaps by short name.

    Parameters
    ----------
    name : str, optional
        One of ``'seq'``, ``'seq_r'``, ``'div'``, ``'div_r'``.
        Default is ``'seq'``.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
    """
    if name not in CMAPS:
        raise KeyError(
            f"Unknown colourmap {name!r}. "
            f"Available: {list(CMAPS.keys())}"
        )
    return CMAPS[name]


# ---- Adaptive scaling for corner plots ---- #

_CORNER_REF_PARAMS = 5
"""Reference parameter count at which the corner-plot scale factor equals 1."""

_FONT_SCALE_MAP = {
    'xx-small': 0.579, 'x-small': 0.694, 'small': 0.833,
    'medium': 1.0, 'large': 1.2, 'x-large': 1.44, 'xx-large': 1.728,
}


def _resolve_size(key):
    """Resolve an rcParam value to a numeric point size.

    Handles both numeric values (returned as-is) and relative font-size
    strings (``'medium'``, ``'large'``, …) which are resolved against the
    current ``font.size``.
    """
    val = mpl.rcParams[key]
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str) and val.lower() in _FONT_SCALE_MAP:
        return float(mpl.rcParams['font.size']) * _FONT_SCALE_MAP[val.lower()]
    return float(mpl.rcParams['font.size'])


def _corner_scale_factor(n_params, n_ref=_CORNER_REF_PARAMS):
    """Multiplicative scale factor for corner-plot visual elements.

    Returns 1.0 when ``n_params <= n_ref`` and grows gently as
    ``(n_params / n_ref) ** 0.3`` beyond that.  The low exponent keeps
    the scaling lightweight: a 12-parameter plot gets ~28 % larger
    sizes, a 15-parameter plot ~39 %.

    Parameters
    ----------
    n_params : int
        Number of sampled parameters (corner-plot dimensions).
    n_ref : int, optional
        Reference count at which the factor equals 1.  Default is 5.

    Returns
    -------
    float
        Scale factor (>= 1.0).
    """
    return max(1.0, (n_params / n_ref) ** 0.3)


def _scaled_corner_sizes(n_params, n_ref=_CORNER_REF_PARAMS):
    """Return rcParam overrides with sizes scaled for *n_params* dimensions.

    Reads the **current** rcParams as base values (so the active style file
    or ``default_plotting()`` call determines the baseline).  Font and tick-
    label sizes scale by the full :func:`_corner_scale_factor`; stroke widths
    (lines, axes edges, tick strokes) scale by its square root so they
    thicken only very slightly.

    Parameters
    ----------
    n_params : int
        Number of parameters in the corner plot.
    n_ref : int, optional
        Reference parameter count.  Default is 5.

    Returns
    -------
    scaled_rc : dict
        Mapping of rcParam keys to scaled numeric values.
    scale : float
        The scale factor that was applied.
    """
    s = _corner_scale_factor(n_params, n_ref)
    sw = s ** 0.5  # stroke widths grow even more slowly

    font_keys = [
        'font.size', 'axes.labelsize', 'axes.titlesize',
        'xtick.labelsize', 'ytick.labelsize', 'legend.fontsize',
    ]
    tick_size_keys = [
        'xtick.major.size', 'xtick.minor.size',
        'ytick.major.size', 'ytick.minor.size',
    ]
    width_keys = [
        'xtick.major.width', 'ytick.major.width',
        'lines.linewidth', 'axes.linewidth',
    ]

    scaled_rc = {}
    for k in font_keys + tick_size_keys:
        scaled_rc[k] = _resolve_size(k) * s
    for k in width_keys:
        scaled_rc[k] = _resolve_size(k) * sw

    return scaled_rc, s


def _legend_bbox(n_params, n_ref=_CORNER_REF_PARAMS):
    """Return (x, y) in figure-relative coordinates for the corner-plot legend.

    Places the legend in the empty upper-right triangle.  For small *N* the
    legend sits inside the single free cell; for large *N* it drifts slightly
    right and drops into the triangle's centre.

    Parameters
    ----------
    n_params : int
        Number of corner-plot dimensions.
    n_ref : int, optional
        Not used directly; kept for signature consistency.

    Returns
    -------
    tuple of float
        ``(x, y)`` in figure-fraction coordinates.
    """
    x = 1.0 - (n_params - 2) * 0.02
    y = 0.85 - (n_params - 2) * 0.01
    return (x, y)


def _reposition_legend(fig, n_params, fontsize=None):
    """Find the first legend on *fig* and move it into the upper-right triangle.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    n_params : int
    fontsize : int or None
        If given, override the legend font size.
    """
    bbox = _legend_bbox(n_params)
    for ax in fig.get_axes():
        leg = ax.get_legend()
        if leg is not None:
            leg.set_bbox_to_anchor(bbox, transform=fig.transFigure)
            if fontsize is not None:
                for txt in leg.get_texts():
                    txt.set_fontsize(fontsize)
            break


def _resolve_columns(df, columns, chain_idx=0):
    """Return the column selection for a single DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame whose columns are being selected.
    columns : None, int, or list
        ``None``  → all columns.  ``int``  → first *n* columns.
        ``list``  → explicit column names (or list-of-lists for per-chain
        selections, in which case *chain_idx* picks the right sub-list).
    chain_idx : int, optional
        Index of the current chain when *columns* is a list of lists.

    Returns
    -------
    pandas.Index or list
        Column names to use.

    Raises
    ------
    ValueError
        If *columns* is not one of the accepted types.
    """
    if columns is None:
        return df.columns
    if isinstance(columns, int):
        return df.columns[:columns]
    if isinstance(columns, list):
        if columns and isinstance(columns[0], list):
            return columns[chain_idx]
        return columns
    raise ValueError("columns must be None, int, or list")


def _n_params_from_columns(dfs, columns):
    """Infer the number of plotted parameters from *columns* and *dfs*."""
    if columns is None:
        return len(dfs[0].columns)
    if isinstance(columns, int):
        return columns
    if isinstance(columns, list):
        if columns and isinstance(columns[0], list):
            return max(len(cols) for cols in columns)
        return len(columns)
    return len(dfs[0].columns)


#---- Plotting Stuff ----#

def which_corner():
    """Print the file path of the corner package currently in use."""
    print('You are using the corner package located at ' + str(c.__file__))

def check_latex():
    """
    Enable LaTeX rendering in matplotlib if a LaTeX installation is found on
    the system PATH, otherwise disable it.
    """
    mpl.rcParams['text.usetex'] = bool(find_executable('latex'))

def default_plotting(style='light', backcolor=None, frontcolor=None):
    """
    Set the default plotting parameters for matplotlib.

    Parameters:
    - style (str): The style of the plot. Can be 'light', 'dark'. Default is 'light'.
    - backcolor (str): The background color of the plot. Default is 'white'.
    - frontcolor (str): The foreground color of the plot. Default is 'black'.
    """

    default_rcParams = {
        'text.usetex': True,
        'font.family': 'serif',
        'font.weight':'medium',
        'mathtext.fontset': 'cm',
        'text.latex.preamble': r"\usepackage{amsmath}",
        'font.size': 20,
        'figure.figsize': (7, 7),
        'figure.titlesize': 22,
        'axes.formatter.use_mathtext': True,
        'axes.formatter.limits': [-2, 4],
        'axes.titlesize': 22,
        'axes.labelsize': 22,
        'xtick.top': True,
        'xtick.major.size': 5,
        'xtick.minor.size': 3,
        'xtick.major.width': 0.8,
        'xtick.minor.visible': True,
        'xtick.direction': 'in',
        'xtick.labelsize': 20,
        'ytick.right': True,
        'ytick.major.size': 5,
        'ytick.minor.size': 3,
        'ytick.major.width': 0.8,
        'ytick.minor.visible': True,
        'ytick.direction': 'in',
        'ytick.labelsize': 20,
        'legend.frameon': True,
        'legend.framealpha': 1,
        'legend.fontsize': 20,
        'legend.scatterpoints' : 3,
        #'lines.color': 'k',
        'lines.linewidth': 2,
        'patch.linewidth': 1,
        'hatch.linewidth': 1,
        'grid.linestyle': 'dashed',
        'savefig.dpi' : 200,
        'savefig.format' : 'pdf',
        'savefig.bbox' : 'tight',
        #'savefig.transparent' : True,
    }

    plt.rcParams.update(default_rcParams)

    if style == 'light':
        set_colors(backcolor='white', frontcolor='black')
    elif style == 'dark':
        set_colors(backcolor='none', frontcolor='white')
        mpl.rcParams['savefig.format'] = 'png'
        mpl.rcParams['savefig.transparent'] = True
    else:
        set_colors(backcolor=backcolor, frontcolor=frontcolor)

    custom_color_cycle()
    check_latex()

def set_colors(backcolor='white', frontcolor='black'):
    """
    Apply background and foreground colors to all relevant matplotlib rcParams.

    This updates axes, tick, legend, grid, and text colors in one call so that
    the entire figure adopts a consistent color scheme.

    Parameters
    ----------
    backcolor : str, optional
        Background color applied to the figure, axes, and legend.
        Accepts any matplotlib color string (name, hex, ``'none'``, …).
        Default is ``'white'``.
    frontcolor : str, optional
        Foreground color applied to text, labels, edges, ticks, grid lines,
        and legend entries.  Default is ``'black'``.
    """

    mpl.rcParams['text.color'] = frontcolor
    mpl.rcParams['axes.labelcolor'] = frontcolor
    mpl.rcParams['axes.edgecolor'] = frontcolor
    mpl.rcParams['xtick.color'] = frontcolor
    mpl.rcParams['ytick.color'] = frontcolor
    mpl.rcParams['axes.facecolor'] = backcolor
    mpl.rcParams['figure.facecolor'] = backcolor
    mpl.rcParams['legend.facecolor'] = backcolor
    #mpl.rcParams['legend.edgecolor'] = frontcolor
    mpl.rcParams['axes.titlecolor'] = frontcolor
    mpl.rcParams['legend.labelcolor'] = frontcolor
    mpl.rcParams['grid.color'] = frontcolor
    mpl.rcParams['lines.color'] = frontcolor

def reset_rc():
    """Reset all matplotlib rcParams to their default values."""
    mpl.rcParams.update(mpl.rcParamsDefault)

def set_ticker():
    """
    Create a ScalarFormatter that respects the current LaTeX rendering setting.

    Note: the formatter is returned but not applied to any axis; assign it
    explicitly with ``ax.xaxis.set_major_formatter(set_ticker())``.

    Returns
    -------
    ticker.ScalarFormatter
        Formatter instance configured to match ``text.usetex``.
    """
    return ticker.ScalarFormatter(useMathText=plt.rcParams['text.usetex'])

def custom_corner(function):
    """Decorator that wraps a corner-plot function with adaptive styling.

    Temporarily applies gently scaled rcParams (based on the number of
    parameters) inside a ``try/finally`` block so the caller's rcParams
    are always restored, even if the wrapped function raises.

    Pysco-specific keyword arguments consumed by the wrapper
    (not forwarded to the underlying corner function):

    * ``savefig``    – path-like; save the figure to this file.
    * ``n_ref``      – int; reference parameter count for scaling (default 5).
    * ``custom_rc``  – bool; apply corner-specific rcParam overrides (default True).
    * ``histtype``   – str; histogram type for 1-D marginals (default ``'step'``).
    * ``histalpha``  – float; face-colour alpha for 1-D histograms (default 0.1).

    Parameters
    ----------
    function : callable
        The function that produces the corner plot (receives ``*args, **kwargs``).

    Returns
    -------
    callable
        Wrapped function.
    """

    @wraps(function)
    def wrapper(*args, **kwargs):
        # ---- snapshot rcParams so we can restore later ----
        _saved_rc = mpl.rcParams.copy()

        try:
            # Detect parameter count for adaptive scaling
            data = args[0] if args else kwargs.get('data', np.empty((0, _CORNER_REF_PARAMS)))
            n_params = data.shape[1] if hasattr(data, 'shape') else len(data[0])

            # Pop pysco-specific keys (never forwarded to corner)
            n_ref = kwargs.pop('n_ref', _CORNER_REF_PARAMS)
            custom_rc = kwargs.pop('custom_rc', True)
            savefig = kwargs.pop('savefig', None)
            histtype = kwargs.pop('histtype', 'step')
            histalpha = kwargs.pop('histalpha', 0.1)

            # Compute sizes scaled to the current rcParams baseline
            scaled_rc, scale = _scaled_corner_sizes(n_params, n_ref=n_ref)

            frontcolor = plt.rcParams['text.color']

            # Corner-plot structural overrides (not size-dependent)
            if custom_rc:
                corner_rcParams = {
                    'xtick.top': False,
                    'xtick.minor.visible': False,
                    'xtick.direction': 'out',
                    'ytick.right': False,
                    'ytick.minor.visible': False,
                    'ytick.direction': 'out',
                }
                plt.rcParams.update(corner_rcParams)
                plt.rcParams.update(scaled_rc)

            # Build corner defaults (user kwargs override these)
            title_fontsize = scaled_rc.get('axes.titlesize', 16)
            lw = scaled_rc.get('lines.linewidth', 1.5)

            defaults_kwargs = dict(
                bins=50,
                smooth=0.5,
                title_kwargs=dict(fontsize=title_fontsize),
                color=PALETTES['cat'][0],
                truth_color=frontcolor,
                quantiles=[0.05, 0.5, 0.95],
                linestyle='-',
                plot_median=False,
                marginal_type='hist',
                quantiles_color=None,
                show_titles=True,
                title_fmt='.2e',
                levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
                plot_density=False,
                plot_datapoints=False,
                fill_contours=True,
                max_n_ticks=5,
                use_math_text=True,
                custom_whspace=0.05,
            )

            # User kwargs take priority over defaults
            merged = {**defaults_kwargs, **kwargs}
            color = merged['color']

            # Build hist_kwargs from resolved values
            hist_kwargs = dict(
                histtype=histtype,
                edgecolor=color,
                lw=lw,
                density=True,
                fc=to_rgba(color, alpha=histalpha),
            )
            merged['hist_kwargs'] = hist_kwargs

            fig = function(*args, **merged)

            # Move the legend (if any) into the upper-right triangle
            _reposition_legend(fig, n_params)

            if savefig is not None:
                fig.savefig(Path(savefig))

            return fig

        finally:
            mpl.rcParams.update(_saved_rc)

    return wrapper


@custom_corner
def corner(*args, **kwargs):
    """Produce a corner plot via :func:`corner.corner` with adaptive styling.

    Accepts all keyword arguments of :func:`corner.corner` plus the
    pysco-specific keys documented in :func:`custom_corner`.
    """
    return c.corner(*args, **kwargs)

def set_color_cycle_from_cmap(cmap=None):
    """
    Set the matplotlib color cycle from a colormap object.

    Parameters
    ----------
    cmap : matplotlib.colors.Colormap or None, optional
        A colormap whose ``.colors`` attribute is used as the new cycle.
        If ``None``, the default matplotlib color cycle is restored.
    """
    if cmap:
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", cmap.colors)
    else:
         plt.rcParams["axes.prop_cycle"] =  plt.rcParamsDefault["axes.prop_cycle"]
    
def get_colors_from_cmap(N, cmap='viridis', reverse=False):
    """
    Sample ``N`` evenly-spaced colors from a named matplotlib colormap.

    Colors are drawn from the upper 70 % of the colormap range
    (``[0.3, 1.0]``) to avoid overly light shades.

    Parameters
    ----------
    N : int
        Number of colors to return.
    cmap : str, optional
        Name of the matplotlib colormap to sample.  Default is ``'viridis'``.
    reverse : bool, optional
        If ``True``, return the colors in reversed order.  Default is ``False``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(N, 4)`` containing RGBA colors.
    """
    cmap = mpl.colormaps[cmap]
    colors = cmap(np.linspace(0.3, 1., N))
    if reverse:
        colors = colors[::-1]
    return colors

def custom_color_cycle(colors='colors6', linestyles=['-'], skip=0, lsfirst=False):
    """
    Configure the matplotlib color (and optionally linestyle) cycle.

    Parameters
    ----------
    colors : str or list, optional
        Either the name of a built-in color list (``'colors6'``, ``'colors8'``,
        ``'colors10'``, ``'fancy'`` — all sourced from arXiv:2107.02270 except
        ``'fancy'``) or an explicit list of matplotlib color strings.
        Default is ``'colors6'``.
    linestyles : list or int, optional
        Linestyles to include in the cycle.  Pass an ``int`` to take the first
        *n* styles from ``['-', '--', '-.', ':']``.  Default is ``['-']``.
    skip : int, optional
        Number of colors to skip from the beginning of the list.  Useful when
        the first color conflicts with other plot elements.  Default is ``0``.
    lsfirst : bool, optional
        If ``True``, the linestyle axis cycles faster than color (i.e.
        ``cycler(color) * cycler(linestyle)``).  If ``False`` (default),
        color cycles faster.
    """

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
    """Return a list of hex color strings for one of the built-in palettes.

    Parameters
    ----------
    colors : str, optional
        Key identifying the palette.  One of:

        * ``'colors6'``  — 6-color palette (arXiv:2107.02270)
        * ``'colors8'``  — 8-color palette (arXiv:2107.02270)
        * ``'colors10'`` — 10-color palette (arXiv:2107.02270)
        * ``'fancy'``    — a small personal palette
        * ``'cat'``      — CVD-safe categorical (8 colours)
        * ``'seq'``      — sequential anchor colours (lavender → navy)
        * ``'div'``      — diverging anchor colours (purple ↔ blue)

        Default is ``'colors6'``.

    Returns
    -------
    list of str
        Hex color strings for the requested palette.
    """
    if colors not in PALETTES:
        raise KeyError(
            f"Unknown palette {colors!r}. "
            f"Available: {list(PALETTES.keys())}"
        )
    return PALETTES[colors]


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

def chainplot(
    dfs,
    names=None,
    columns=None,
    truths=None,
    colors=None,
    savefig=None,
    ls=None,
    padding=(4.0, 2.5),
    n_ticks=4,
    chain_kwargs=None,
    chainconfig_kwargs=None,
    legend_kwargs=None,
    plotconfig_kwargs=None,
):
    """Plot MCMC chains using ChainConsumer with adaptive styling.

    Parameters
    ----------
    dfs : list, dict, or DataFrame
        Samples to plot.  Accepts a single DataFrame, a list of DataFrames,
        or a ``{name: DataFrame}`` dictionary.
    names : list of str, optional
        Display name for each chain.  When *dfs* is a dict the keys are used.
    columns : None, int, or list, optional
        Which columns to plot.  ``None`` → all, ``int`` → first *n*,
        ``list`` → explicit names, ``list[list]`` → per-chain selections.
    truths : dict, optional
        True parameter values passed to :class:`chainconsumer.Truth`.
    colors : str or list, optional
        Colour(s) for the chains.  A palette name (``'cat'``,
        ``'colors6'``, ``'colors10'``, …), a single colour string,
        or an explicit list.  Default is ``'cat'`` (8-colour CVD-safe).
    savefig : str or path-like, optional
        If given, save the figure to this path.
    ls : str, optional
        Linestyle applied to every chain.  ``None`` cycles
        ``['-', '--', '-.', ':']``.
    padding : float or tuple of float, optional
        Extra padding (in points) between tick labels and axis labels,
        passed directly to ``ax.xaxis.labelpad`` / ``ax.yaxis.labelpad``.
        A scalar applies to both axes; a 2-tuple sets ``(xpad, ypad)``.
        Default is ``(4.0, 2.5)``.
    n_ticks : int, optional
        Maximum number of major ticks per subplot axis.  A smaller value
        (e.g. 3) gives more breathing room between tick labels; a larger
        value (e.g. 5) adds more detail.  Default is ``4``.
    chain_kwargs : dict, optional
        Extra keyword arguments forwarded to :class:`chainconsumer.Chain`.
    chainconfig_kwargs : dict, optional
        Overrides for :class:`chainconsumer.ChainConfig`.
    legend_kwargs : dict, optional
        Overrides for the matplotlib legend.
    plotconfig_kwargs : dict, optional
        Overrides for :class:`chainconsumer.PlotConfig`.

    Returns
    -------
    C : chainconsumer.ChainConsumer
        The configured ChainConsumer object.
    fig : matplotlib.figure.Figure
        The resulting figure.
    """
    # Avoid mutable default arguments
    chain_kwargs = chain_kwargs or {}
    chainconfig_kwargs = chainconfig_kwargs or {}
    legend_kwargs = legend_kwargs or {}
    plotconfig_kwargs = plotconfig_kwargs or {}

    # ---- Normalise inputs ------------------------------------------------
    if not isinstance(dfs, list):
        if isinstance(dfs, dict):
            names = list(dfs.keys())
            dfs = list(dfs.values())
        else:
            names = [names] if isinstance(names, str) else ["Chain"]
            dfs = [dfs]
    else:
        if names is None:
            names = [f"Chain {i}" for i in range(len(dfs))]
        elif len(names) != len(dfs):
            raise ValueError("Number of names must match number of chains")

    n_chains = len(dfs)

    # ---- Linestyles ------------------------------------------------------
    lss = ["-", "--", "-.", ":"] if ls is None else [ls] * n_chains
    while n_chains > len(lss):
        lss = lss + lss

    # ---- Colours ---------------------------------------------------------
    # Escalation order: cat(8) → colors10(10) → continuous cmap fallback
    _colorlists = ['cat', 'colors10']
    colorstring = None

    def _cmap_to_hex(cmap_name, N):
        """Sample N evenly-spaced hex colours from a named colormap."""
        return [
            mpl.colors.to_hex(c)
            for c in mpl.colormaps[cmap_name](np.linspace(0.15, 0.85, N))
        ]

    if colors is None:
        colorstring = 'cat'
        colors = get_colorslist(colorstring)
    elif isinstance(colors, str):
        if colors in PALETTES:
            colorstring = colors
            colors = get_colorslist(colorstring)
        elif colors in mpl.colormaps:
            # Caller passed a colormap name → sample N hex colours immediately
            colors = _cmap_to_hex(colors, n_chains)
        else:
            # Assume it's a single colour string; replicate for every chain
            colors = [colors] * n_chains
    # else: assume it's already a list of valid colour specs

    if n_chains > len(colors):
        if colorstring is not None and colorstring in _colorlists:
            # Try successively larger palettes
            idx = _colorlists.index(colorstring)
            while n_chains > len(colors) and idx < len(_colorlists) - 1:
                idx += 1
                colors = get_colorslist(_colorlists[idx])
        if n_chains > len(colors):
            colors = _cmap_to_hex('pysco_seq', n_chains)

    # ---- Adaptive scaling ------------------------------------------------
    n_params = _n_params_from_columns(dfs, columns)
    scaled_rc, scale = _scaled_corner_sizes(n_params)

    # PlotConfig requires int font sizes; round here so all downstream uses
    # (legend_kwargs, PlotConfig, offset text) are already integers.
    fontsize = round(scaled_rc['font.size'])
    ticksize = round(scaled_rc['xtick.labelsize'])
    offset_fontsize = ticksize
    linewidth = scaled_rc['lines.linewidth']

    # Legend scales more aggressively than labels: exponent 0.6 instead
    # of the general 0.3, so that legend text stays readable in large grids.
    legend_scale = max(1.0, (n_params / _CORNER_REF_PARAMS) ** 0.6)
    base_fontsize = _resolve_size('font.size')       # unscaled baseline
    legend_fontsize = round(base_fontsize * legend_scale)

    # Snapshot rcParams and apply scaled overrides inside try/finally
    _saved_rc = mpl.rcParams.copy()
    try:
        plt.rcParams.update(scaled_rc)

        # ---- Build chains ------------------------------------------------
        C = chainconsumer.ChainConsumer()

        for i, (name, df) in enumerate(zip(names, dfs)):
            columns_here = _resolve_columns(df, columns, chain_idx=i)
            chain = chainconsumer.Chain(
                samples=df[columns_here],
                name=name,
                linewidth=linewidth,
                color=colors[i],
                linestyle=lss[i],
                **chain_kwargs,
            )
            C.add_chain(chain)

        if truths is not None:
            C.add_truth(chainconsumer.Truth(
                location=truths,
                color=mpl.rcParams['text.color'],
                line_style="-",
                name="True Values",
            ))

        # ---- ChainConfig -------------------------------------------------
        _default_chain_cfg = {
            "sigmas": [0, 1, 2],
            "shade_gradient": 1,
            "shade_alpha": 0.5,
            "statistics": "cumulative",
            "summary_area": 0.90,
        }
        chainconfig_kwargs = {**_default_chain_cfg, **chainconfig_kwargs}

        # ---- Legend (adaptive size; placement handled after plotting) ----
        _default_legend = {
            "fontsize": legend_fontsize,
            "frameon": False,
        }
        legend_kwargs = {**_default_legend, **legend_kwargs}

        # ---- PlotConfig --------------------------------------------------
        _default_plot = {
            "summarise": False,
            "tick_font_size": ticksize,
            "label_font_size": fontsize,
            "summary_font_size": fontsize,
            "contour_label_font_size": fontsize,
            "diagonal_tick_labels": True,
            "spacing": 3.0,
            "usetex": mpl.rcParams.get('text.usetex', True),
            "serif": True,
            "dpi": 300,
            "legend_kwargs": legend_kwargs,
            "legend_color_text": False,
            "show_legend": True,
            "max_ticks": n_ticks,
        }
        plotconfig_kwargs = {**_default_plot, **plotconfig_kwargs}

        C.set_override(chainconsumer.ChainConfig(**chainconfig_kwargs))
        C.set_plot_config(chainconsumer.PlotConfig(**plotconfig_kwargs))

        # ---- Plot --------------------------------------------------------
        fig = C.plotter.plot()
        axes = fig.get_axes()

        xpad, ypad = (padding, padding) if isinstance(padding, (int, float)) else padding

        # First pass: formatters and padding on all visible axes
        for ax in axes:
            if not ax.get_visible():
                continue
            ax.xaxis.set_major_formatter(
                ticker.ScalarFormatter(useMathText=True, useOffset=True))
            ax.yaxis.set_major_formatter(
                ticker.ScalarFormatter(useMathText=True, useOffset=True))

            x_off = ax.xaxis.get_offset_text()
            x_off.set_x(1.0)
            x_off.set_y(-0.3)
            x_off.set_fontsize(offset_fontsize)

            y_off = ax.yaxis.get_offset_text()
            y_off.set_fontsize(offset_fontsize)

            ax.xaxis.labelpad = xpad
            ax.yaxis.labelpad = ypad

        # Move legend into the upper-right triangle (scaled to n_params)
        _reposition_legend(fig, n_params, fontsize=legend_fontsize)

        # Align all axis labels to a common position regardless of tick width
        fig.align_labels()

        if savefig is not None:
            fig.savefig(Path(savefig))
            plt.close(fig)

    finally:
        mpl.rcParams.update(_saved_rc)

    return C, fig