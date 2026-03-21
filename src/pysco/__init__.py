# -*- coding: utf-8 -*-
__name__                = "pysco"
__version__             = "0.1.1"
__author__              = "Alessandro Santini"
__author_email__        = "alessandro.santini@aei.mpg.de"
__description__         = "PYthon ShortCuts & Others: plotting routines I like to use & overall collection of useful snippets"
__license__             = "MIT"

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
import logging


log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

import pysco.plots
import pysco.utils

try:
    import pysco.eryn
    eryn_here = True

except (ImportError, ModuleNotFoundError):
    eryn_here = False
    log.warning("Eryn not found. Some features will be disabled.")
