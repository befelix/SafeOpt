"""Test the SafeOptSwarm method"""

from __future__ import division


import GPy
import numpy as np
import pytest

from safeopt import SafeOptSwarm


def test_empty_safe_set():
    """Make sure an error is raised with an empty safe set."""

    x = np.array([[0.]])
    y = np.array([[-1.]])
    gp = GPy.models.GPRegression(x, y, noise_var=0.01 ** 2)

    opt = SafeOptSwarm(gp, fmin=[0.], bounds=[[-1., 1.]])
    with pytest.raises(RuntimeError):
        opt.optimize()
