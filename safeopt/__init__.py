"""
The `safeopt` package implements tools for Safe Bayesian optimization.

Main classes
------------

These classes provide the main functionality for Safe Bayesian optimization.
:class:`SafeOpt` implements the exact algorithm, which is very inefficient
for large problems. :class:`SafeOptSwarm` scales to higher-dimensional
problems by relying on heuristics and adaptive swarm discretization.

.. autosummary::
   :template: template.rst
   :toctree:

   SafeOpt
   SafeOptSwarm

Utilities
---------

The following are utilities to make testing and working with the library more
pleasant.

.. autosummary::
   :template: template.rst
   :toctree:

   sample_gp_function
   linearly_spaced_combinations
   plot_2d_gp
   plot_3d_gp
   plot_contour_gp
"""

from __future__ import absolute_import

from .utilities import *
from .gp_opt import *
