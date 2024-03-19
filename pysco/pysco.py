# -*- coding: utf-8 -*-

if __name__ != "__main__":
    __name__                = "pysco"
    __version__             = "0.0.2"
    __author__              = "Alessandro Santini"
    __author_email__        = "alessandro.santini@aei.mpg.de"
    __description__         = "PYthon ShortCuts & Others: plotting routines I like to use & overall collection of useful snippets"
    __license__             = "MIT"


from .plot import default_plotting, custom_color_cycle

default_plotting()
custom_color_cycle()