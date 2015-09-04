"""
Utilities to sample GP functions, compute hyperparameters and to create
linear combinations of differently discretized data.

Author: Felix Berkenkamp (befelix at inf dot ethz dot ch)
"""
from __future__ import print_function, absolute_import, division

from collections import Sequence            # isinstance(...,Sequence)
import numpy as np
import GPy
from scipy.interpolate import griddata      # For sampling GP functions


__all__ = ['linearly_spaced_combinations', 'get_hyperparameters',
           'sample_gp_function']


def linearly_spaced_combinations(bounds, num_samples):
    """
    Return 2-D array with all linearly spaced combinations with the bounds.

    Parameters
    ----------
    bounds: sequence of tuples
        The bounds for the variables, [(x1_min, x1_max), (x2_min, x2_max), ...]
    num_samples: integer or array_likem
        Number of samples to use for every dimension. Can be a constant if
        the same number should be used for all, or an array to fine-tune
        precision. Total number of data points is num_samples ** len(bounds).

    Returns
    -------
    combinations: 2-d array
        A 2-d arrray. If d = len(bounds) and l = prod(num_samples) then it
        is of size l x d, that is, every row contains one combination of
        inputs.
    """
    num_vars = len(bounds)
    if not isinstance(num_samples, Sequence):
        num_samples = [num_samples] * num_vars

    # Create linearly spaced test inputs
    inputs = [np.linspace(b[0], b[1], n) for b, n in zip(bounds,
                                                         num_samples)]

    # Convert to 2-D array
    return np.array([x.ravel() for x in np.meshgrid(*inputs)]).T


def get_hyperparameters(function, bounds, num_samples, kernel,
                        likelihood=GPy.likelihoods.gaussian.Gaussian()):
    """
    Optimize for hyperparameters by sampling inputs from a uniform grid.

    Parameters
    ----------
    function: method
        Returns the function values, needs to be vectorized to accept 2-D
        arrays as inputs for each variable
    bounds: array_like of tuples
        Each tuple consists of the upper and lower bounds of the variable
    N: integer
        Number of sample points per dimension, total = N ** len(bounds).
        Alternatively a list of sample points per dimension.
    kernel: instance of GPy.kern.*
    likelihood: instance of GPy.likelihoods.*
        Defaults to GPy.likelihoods.gaussian.Gaussian()

    Returns
    -------
    kernel: instance of GPy.kern.*
        Kernel with the optimized hyperparameters
    likelihood: instance of GPy.likelihoods.*
        Likelihood with the optimized hyperparameters

    Notes
    -----
    Constrained optimization of the hyperparameters can be handled by
    passing a kernel or likelihood with the corresponding constraints.
    For example:
    ``likelihood.constrain_fixed(warning=False)`` to fix the observation noise.
    """
    inputs = linearly_spaced_combinations(bounds, num_samples)
    output = function(inputs)

    inference_method = GPy.inference.latent_function_inference.\
        exact_gaussian_inference.ExactGaussianInference()

    gp = GPy.core.GP(X=inputs, Y=output[:, None],
                     kernel=kernel,
                     inference_method=inference_method,
                     likelihood=likelihood)
    gp.optimize()
    return gp.kern, gp.likelihood


def sample_gp_function(kernel, bounds, noise_var, num_samples):
    """
    Sample a function from a gp with corresponding kernel within its bounds.

    Parameters
    ----------
    kernel: instance of GPy.kern.*
    bounds: list of tuples
        [(x1_min, x1_max), (x2_min, x2_max), ...]
    noise_var: float
        Variance of the observation noise of the GP function
    num_samples: int or list
        If integer draws the corresponding number of samples in all
        dimensions and test all possible input combinations. If a list then
        the list entries correspond to the number of linearly spaced samples of
        the corresponding input

    Returns
    -------
    function: object
        function(x, noise=True)
        A function that takes as inputs new locations x to be evaluated and
        returns the corresponding noisy function values. If noise=False is
        set the true function values are returned (useful for plotting).
    """
    inputs = linearly_spaced_combinations(bounds, num_samples)
    cov = kernel.K(inputs) + np.eye(inputs.shape[0]) * 1e-6
    output = np.random.multivariate_normal(np.zeros(inputs.shape[0]),
                                           cov)

    def evaluate_gp_function(x, noise=True):
        x = np.atleast_2d(x)
        y = griddata(inputs, output, x, method='linear')
        y = np.atleast_2d(y)
        if noise:
            y += np.sqrt(noise_var) * np.random.randn(x.shape[0], 1)
        return y

    return evaluate_gp_function