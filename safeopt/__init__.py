"""
The `safeopt` package provides...

Main classes
============
.. autosummary::
   SafeOpt
   SafeOptSwarm

Utilities
=========
.. autosummary::
   sample_gp_function
   linearly_spaced_combinations
   plot_2d_gp
   plot_3d_gp
   plot_contour_gp
"""

from __future__ import absolute_import

from .utilities import *
from .gp_opt import *


__all__ = [s for s in dir() if not s.startswith('_')]
