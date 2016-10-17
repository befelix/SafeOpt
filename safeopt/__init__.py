"""
The `safeopt` package provides...

Main classes
============

These classes provide the main functionality for Safe Bayesian optimization.

.. autosummary::
   SafeOpt
   SafeOptSwarm

Utilities
=========

The following are utilities to make testing and working with the library more pleasant.

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
