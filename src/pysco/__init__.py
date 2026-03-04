# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
import logging


log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

from .pysco import __name__
from .pysco import __version__
from .pysco import __author__
from .pysco import __author_email__

from .pysco import *

import pysco.plots
import pysco.utils

try:
    import pysco.eryn
    eryn_here = True

except (ImportError, ModuleNotFoundError):
    eryn_here = False
    log.warning("Eryn not found. Some features will be disabled.")
