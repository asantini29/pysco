# -*- coding: utf-8 -*-
# Credits: https://github.com/duetosymmetry/latex-mpl-fig-tips/blob/main

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files


def _packaged_style_path(name):
    return str(files("pysco.plots").joinpath("mplfiles", f"{name}.mplstyle"))


paper_style = _packaged_style_path("paper")
corner_style = _packaged_style_path("corner")

pt = 1./72.27 # Hundreds of years of history... 72.27 points to an inch.
golden_ratio = (1. + 5.**0.5)/2.

journal_sizes = {
                "prd": {"onecol": 246.*pt, "twocol": 510.*pt},
                "cqg": {"onecol": 374.*pt}, # CQG is only one column
    
                }

def get_style(style, journal="prd", cols="onecol", aspect=golden_ratio):
    """Return a style list for use with plt.style.context().

    Parameters
    ----------
    style : str
        Style name, e.g. 'paper'.
    journal : str
        Journal name, must be a key in journal_sizes.
    cols : str
        Column key, e.g. 'onecol' or 'twocol'.
    aspect : float
        Height = width / aspect. Defaults to the golden ratio.

    Returns
    -------
    list
        List of style files and parameters to use with plt.style.context() or plt.style.use().
    """
    if style.split(".")[-1] != "mplstyle":
        style = _packaged_style_path(style)

    width = journal_sizes[journal][cols]
    return [style, {"figure.figsize": (width, width / aspect)}]
