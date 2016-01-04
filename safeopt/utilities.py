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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D     # Create 3D axes
from matplotlib import cm                   # 3D plot colors


__all__ = ['linearly_spaced_combinations', 'get_hyperparameters',
           'sample_gp_function', 'plot_2d_gp', 'plot_3d_gp', 'plot_contour_gp']


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
        """Evaluate the GP sample function."""
        x = np.atleast_2d(x)
        y = griddata(inputs, output, x, method='linear')
        y = np.atleast_2d(y)
        if noise:
            y += np.sqrt(noise_var) * np.random.randn(x.shape[0], 1)
        return y

    return evaluate_gp_function


def plot_2d_gp(gp, inputs, predictions=None, figure=None, axis=None,
               slice=None, beta=3, **kwargs):
        """
        Plot a 2D GP with uncertainty.

        Parameters
        ----------
        gp: Instance of GPy.models.GPRegression
        inputs: 2darray
            The input parameters at which the GP is to be evaluated
        predictions: ndarray
            Can be used to manually pass the GP predictions, set to None to
            use the gp directly. Is of the form (mean, variance)
        figure: matplotlib figure
            The figure on which to draw (ignored if axis is provided
        axis: matplotlib axis
            The axis on which to draw
        slice: int
            A list containing the input slices to be plotted, e.g. [0, 1]
        beta: float
            The confidence interval used
        """
        if slice is None :
            if gp.kern.input_dim > 1:
                raise NotImplementedError('This only works for 1D inputs')
            else:
                slice = 0

        if axis is None:
            if figure is None:
                figure = plt.figure()
                axis = figure.gca()
            else:
                axis = figure.gca()

        if predictions is None:
            mean, var = gp._raw_predict(inputs)
        else:
            mean, var = predictions

        output = mean.squeeze()
        std_dev = beta * np.sqrt(var.squeeze())

        axis.fill_between(inputs[:, slice],
                          output - std_dev,
                          output + std_dev,
                          facecolor='blue',
                          alpha=0.3)

        axis.plot(inputs[:, slice], output, **kwargs)
        axis.plot(gp.X[:, slice], gp.Y, 'kx', ms=10, mew=3)
        axis.set_xlim([np.min(inputs[:, slice]), np.max(inputs[:, slice])])


def plot_3d_gp(gp, inputs, predictions=None, figure=None, axis=None,
               slices=None, beta=3, **kwargs):
        """
        Plot a 3D gp with uncertainty

        Parameters
        ----------
        gp: Instance of GPy.models.GPRegression
        inputs: 2darray
            The input parameters at which the GP is to be evaluated
        predictions: ndarray
            Can be used to manually pass the GP predictions, set to None to
            use the gp directly. Is of the form [mean, variance]
        figure: matplotlib figure
            The figure on which to draw (ignored if axis is provided
        axis: matplotlib axis
            The axis on which to draw
        slices: list
            A list containing the input slices to be plotted, e.g. [0, 1]
        beta: float
            The confidence interval used
        """
        if slices is None:
            if gp.kern.input_dim > 2:
                raise NotImplementedError('This only works for 1D inputs')
            slices = [0, 1]
        elif len(slices) > 2:
            raise NotImplemented('Specify the correct number of slices')

        if axis is None:
            if figure is None:
                figure = plt.figure()
                axis = Axes3D(figure)
            else:
                axis = Axes3D(figure)

        if predictions is None:
            mean, var = gp._raw_predict(inputs)
        else:
            mean, var = predictions

        output = mean.squeeze()

        axis.plot_trisurf(inputs[:, slices[0]],
                          inputs[:, slices[1]],
                          output,
                          cmap=cm.jet, linewidth=0.2, alpha=0.5)

        axis.plot(gp.X[:, slices[0]],
                  gp.X[:, slices[1]],
                  gp.Y[:, 0],
                  'o')

        axis.set_xlim([np.min(inputs[:, slices[0]]),
                       np.max(inputs[:, slices[0]])])

        axis.set_ylim([np.min(inputs[:, slices[1]]),
                       np.max(inputs[:, slices[1]])])


def plot_contour_gp(gp, inputs, predictions=None, figure=None, axis=None):
        """
        Plot a 3D gp with uncertainty

        Parameters
        ----------
        gp: Instance of GPy.models.GPRegression
        inputs: list of arrays/floats
            The input parameters at which the GP is to be evaluated,
            here instead of the combinations of inputs the individual inputs
            that are spread in a grid are given. Only two of the arrays
            should have more than one value (not fixed).
        predictions: ndarray
            Can be used to manually pass the GP predictions, set to None to
            use the gp directly.
        figure: matplotlib figure
            The figure on which to draw (ignored if axis is provided
        axis: matplotlib axis
            The axis on which to draw
        """

        if axis is None:
            if figure is None:
                figure = plt.figure()
                axis = figure.gca()
            else:
                axis = figure.gca()

        slices = []
        lengths = []
        for i, inp in zip(range(len(inputs)), inputs):
            if isinstance(inp, np.ndarray):
                slices.append(i)
                lengths.append(inp.shape[0])

        # Convert to array with combinations of inputs
        gp_inputs = np.array([x.ravel() for x in np.meshgrid(*inputs)]).T

        if predictions is None:
            mean, var = gp._raw_predict(gp_inputs)
        else:
            mean, var = predictions

        output = mean.squeeze()

        c = axis.contour(inputs[slices[0]].squeeze(),
                         inputs[slices[1]].squeeze(),
                         output.reshape(*lengths),
                         20)

        plt.colorbar(c)
        axis.plot(gp.X[:, slices[0]], gp.X[:, slices[1]], 'ob')

        print(slices)
        print(np.min(inputs[slices[0]]))
        axis.set_xlim([np.min(inputs[slices[0]]),
                       np.max(inputs[slices[0]])])

        axis.set_ylim([np.min(inputs[slices[1]]),
                       np.max(inputs[slices[1]])])