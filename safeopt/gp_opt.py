"""
Classes that implement SafeOpt.

Author: Felix Berkenkamp (befelix at inf dot ethz dot ch)
"""

from __future__ import print_function, absolute_import, division

from .utilities import *

import sys
import numpy as np                          # ...
import scipy as sp
import GPy                                  # GPs
from GPy.util.linalg import dpotrs          # For rank-1 updates
from GPy.inference.latent_function_inference.posterior import Posterior
import matplotlib.pyplot as plt             # Plotting
from collections import Sequence            # isinstance(...,Sequence)
from matplotlib import cm                   # 3D plot colors
from scipy.spatial.distance import cdist    # Efficient distance computation
from mpl_toolkits.mplot3d import Axes3D     # Create 3D axes


__all__ = ['SafeOpt', 'GaussianProcessOptimization']


# For python 2 (python 3 is not yet supported by GPy)
if sys.version_info[0] < 3:
    range = xrange


class GaussianProcessOptimization(object):
    """
    Base class for GP optimization.

    Handles common functionality.

    Parameters:
    -----------
    function: object
        A function that returns the current value that we want to optimize.
    gp: GPy Gaussian process
    parameter_set: 2d-array
        List of parameters
    beta: float or callable
        A constant or a function of the time step that scales the confidence
        interval of the acquisition function.
    """
    def __init__(self, function, gp, parameter_set, beta, num_contexts):
        super(GaussianProcessOptimization, self).__init__()

        self.gp = gp
        self.function = function

        if hasattr(beta, '__call__'):
            # Beta is a function of t
            self.beta = beta
        else:
            # Assume that beta is a constant
            self.beta = lambda t: beta

        self._inputs = None
        self.bounds = None
        self.num_samples = 0
        self.num_contexts = num_contexts
        self.inputs = parameter_set.copy()

        if self.num_contexts > 0:
            context_shape = (self.inputs.shape[0], self.num_contexts)
            self.inputs = np.hstack((self.inputs,
                                     np.empty(context_shape,
                                              dtype=self.inputs.dtype)))

        # Time step
        self.t = 0

    @property
    def inputs(self):
        """Discrete parameter samples for Bayesian optimization."""
        return self._inputs

    @inputs.setter
    def inputs(self, parameter_set):
        self._inputs = parameter_set

        # Plotting bounds (min, max value
        self.bounds = list(zip(np.min(self._inputs, axis=0),
                               np.max(self._inputs, axis=0)))
        self.num_samples = [len(np.unique(self._inputs[:, i]))
                            for i in range(self._inputs.shape[1])]

    def plot(self, axis=None, figure=None, n_samples=None, plot_3d=False,
             **kwargs):
        """
        Plot the current state of the optimization.

        Parameters
        ----------
        axis: matplotlib axis
            The axis on which to draw (does not get cleared first)
        figure: matplotlib figure
            Ignored if axis is already defined
        n_samples: int
            How many samples to use for plotting (uses input parameters if
            None)
        plot_3d: boolean
            If set to true shows a 3D plot for 2 dimensional data
        """
        # 4d plots are tough...
        if self.gp.kern.input_dim > 2:
            return None

        if n_samples is None:
            inputs = self.inputs
            n_samples = self.num_samples
        else:
            if self.gp.kern.input_dim == 1 or plot_3d:
                inputs = linearly_spaced_combinations(self.bounds,
                                                      n_samples)
            if not isinstance(n_samples, Sequence):
                n_samples = [n_samples] * len(self.bounds)

        if self.gp.kern.input_dim - self.num_contexts > 1:   # 3D plot
            if plot_3d:
                plot_3d_gp(self.gp, inputs, figure=figure, axis=axis, **kwargs)
            else:
                plot_contour_gp(self.gp,
                                [np.linspace(self.bounds[0][0],
                                             self.bounds[0][1],
                                             n_samples[0]),
                                 np.linspace(self.bounds[1][0],
                                             self.bounds[1][1],
                                             n_samples[1])],
                                figure=figure,
                                axis=axis)

        else:   # 2D plots with uncertainty
            plot_2d_gp(self.gp, inputs, figure=figure, axis=axis,
                       slice=0, **kwargs)

    def add_new_data_point(self, x, y, gp=None):
        """
        Add a new function observation to the GP.

        Parameters
        ----------
        x: 2d-array
        y: 2d-array
        gp: instance of GPy.model.GPRegression
            The gp to which the data point should be added
        """
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        if gp is None:
            gp = self.gp

        # Add data to GP
        # self.gp.set_XY(np.vstack([self.gp.X, x]),
        #                np.vstack([self.gp.Y, y]))

        # Add data row/col to kernel (a, b)
        # [ K    a ]
        # [ a.T  b ]
        #
        # Now K = L.dot(L.T)
        # The new Cholesky decomposition is then
        # L_new = [ L    0 ]
        #         [ c.T  d ]
        a = gp.kern.K(gp.X, x)
        b = gp.kern.K(x, x)

        b += 1e-8 + gp.likelihood.gaussian_variance(
                gp.Y_metadata)

        L = gp.posterior.woodbury_chol
        c = sp.linalg.solve_triangular(gp.posterior.woodbury_chol, a,
                                       lower=True)

        d = np.sqrt(b - c.T.dot(c))

        L_new = np.asfortranarray(
                np.bmat([[L, np.zeros_like(c)],
                         [c.T, d]]))

        K_new = np.bmat([[gp.posterior._K, a],
                         [a.T, b]])

        gp.X = np.vstack((gp.X, x))
        gp.Y = np.vstack((gp.Y, y))

        alpha, _ = dpotrs(L_new, gp.Y, lower=1)
        gp.posterior = Posterior(woodbury_chol=L_new,
                                 woodbury_vector=alpha,
                                 K=K_new)

    def change_context(self, context):
        """Change the context of the input points.

        Parameters
        ----------
        context: ndarray
            New context that should be applied to the input parameters
        """
        self.inputs[:, -self.num_contexts:] = context

    def remove_last_data_point(self, gp=None):
        """Remove the data point that was last added to the GP.

        Parameters:
            gp: Instance of GPy.models.GPRegression
                The gp that the last data point should be removed from
        """
        if gp is None:
            gp = self.gp

        # self.gp.set_XY(self.gp.X[:-1, :], self.gp.Y[:-1, :])
        gp.X = gp.X[:-1, :]
        gp.Y = gp.Y[:-1, :]
        gp.posterior = Posterior(
                woodbury_chol=np.asfortranarray(
                        gp.posterior.woodbury_chol[:-1, :-1]),
                woodbury_vector=np.asfortranarray(
                        gp.posterior.woodbury_vector[:-1]),
                K=gp.posterior._K[:-1, :-1])


class SafeOpt(GaussianProcessOptimization):
    """
    A class to maximize a function using the adapted or original SafeOpt alg.

    Parameters
    ----------
    function: object
        A function that returns the current value that we want to optimize.
    gp: GPy Gaussian process
        A Gaussian process which is initialized with safe, initial data points.
        If a list of GPs then the first one is the value, while all the
        other ones are safety constraints.
    parameter_set: 2d-array
        List of parameters
    fmin: list of floats
        Safety threshold for the function value. If multiple safety constraints
        are used this can also be a list of floats (the first one is always
        the one for the values, can be set to None if not wanted)
    lipschitz: list of floats
        The Lipschitz constant of the system, if None the GP confidence
        intervals are used directly.
    beta: float or callable
        A constant or a function of the time step that scales the confidence
        interval of the acquisition function.
    threshold: float
        The algorithm will not try to expand any points that are below this
        threshold. This makes the algorithm stop expanding points eventually.

    """
    def __init__(self, function, gp, parameter_set, fmin,
                 lipschitz=None, beta=3.0, num_contexts=0, threshold=0):

        if isinstance(gp, list):
            self.gps = gp
            gp = self.gps[0]
        else:
            self.gps = [gp]

        super(SafeOpt, self).__init__(function, gp, parameter_set, beta,
                                      num_contexts)

        self.fmin = fmin
        self.liptschitz = lipschitz
        self.threshold = threshold

        if not isinstance(self.fmin, list):
            self.fmin = [self.fmin] * len(self.gps)
            if len(self.gps) > 1:
                self.fmin[0] = None
        self.fmin = np.atleast_1d(np.asarray(self.fmin).squeeze())

        if self.liptschitz is not None:
            if not isinstance(self.liptschitz, list):
               self.liptschitz = [self.liptschitz] * len(self.gps)
            self.liptschitz = np.atleast_1d(
                    np.asarray(self.liptschitz).squeeze())

        # Value intervals
        self.Q = np.empty((self.inputs.shape[0],2 * len(self.gps)),
                          dtype=np.float)

        # Safe set
        self.S = np.zeros(self.inputs.shape[0], dtype=np.bool)

        # Switch to use confidence intervals for safety
        if lipschitz is None:
            self._use_lipschitz = False
        else:
            self._use_lipschitz = True

        # Set of expanders and maximizers
        self.G = self.S.copy()
        self.M = self.S.copy()

        # Update the sets
        self.update_confidence_intervals()

    @property
    def use_lipschitz(self):
        """
        Boolean that determines whether to use the Lipschitz constant.

        By default this is set to False, which means the adapted SafeOpt
        algorithm is used, that uses the GP confidence intervals directly.
        If set to True, the `self.lipschitz` parameter is used to compute
        the safe and expanders sets.
        """
        return self._use_lipschitz

    @use_lipschitz.setter
    def use_lipschitz(self, value):
        if value and self.liptschitz is None:
            raise ValueError('Lipschitz constant not defined')
        self._use_lipschitz = value

    def update_confidence_intervals(self, context=None):
        """Recompute the confidence intervals form the GP.

        Parameters
        ----------
        context: ndarray
            Array that contains the context used to compute the sets
        """
        beta = self.beta(self.t)

        # Update context to current setting
        if context is not None:
            self.inputs[:, -self.num_contexts:] = context

        # Iterate over all functions
        for i in range(len(self.gps)):
            # Evaluate acquisition function
            mean, var = self.gps[i]._raw_predict(self.inputs)

            mean = mean.squeeze()
            std_dev = np.sqrt(var.squeeze())

            # Update confidence intervals
            self.Q[:, 2 * i] = mean - beta * std_dev
            self.Q[:, 2 * i + 1] = mean + beta * std_dev

    def compute_safe_set(self):
        """Compute only the safe set based on the current confidence bounds."""
        # Update safe set
        self.S[:] = np.all(self.Q[:, ::2] > self.fmin, axis=1)

    def compute_sets(self, full_sets=False):
        """
        Compute the safe set of points, based on current confidence bounds.

        Parameters
        ----------
        context: ndarray
            Array that contains the context used to compute the sets
        full_sets: boolean
            Whether to compute the full set of expanders or whether to omit
            computations that are not relevant for running SafeOpt
            (This option is only useful for plotting purposes)
        """
        beta = self.beta(self.t)

        # Update safe set
        self.compute_safe_set()

        # Reference to confidence intervals
        l, u = self.Q[:, :2].T

        if not np.any(self.S):
            raise EnvironmentError('There are no safe points to evaluate.')

        # Set of possible maximisers
        # Maximizers: safe upper bound above best, safe lower bound
        self.M[:] = False
        self.M[self.S] = u[self.S] >= np.max(l[self.S])
        max_var = np.max(u[self.M] - l[self.M])

        # Optimistic set of possible expanders
        l = self.Q[:, ::2]
        u = self.Q[:, 1::2]

        self.G[:] = False

        # For the run of the algorithm we do not need to calculate the
        # full set of potential expanders:
        # We can skip the ones already in M and ones that have lower
        # variance than the maximum variance in M, max_var or the threshold.
        # Amongst the remaining ones we only need to find the
        # potential expander with maximum variance
        if full_sets:
            s = self.S
        else:
            # skip points in M, they will already be evaluated
            s = np.logical_and(self.S, ~self.M)

            # Remove points with a variance that is too small
            s[s] = np.max(u[s, :] - l[s, :], axis=1) > max(max_var,
                                                           self.threshold)

        # no need to evaluate any points as expanders in G, exit
        if not np.any(s):
            return

        # def sort_generator(array):
        #     """Return the indeces of the biggest elements in order.
        #
        #     Avoids sorting everything, sorts the relevant parts at a time.
        #
        #     Parameters
        #     ----------
        #     array: 1d-array
        #         The array which we want to sort and iterate over
        #
        #     Returns
        #     -------
        #     iterable:
        #         Indeces of the largest elements in order
        #     """
        #     sort_id = np.argpartition(array, -1)
        #     yield sort_id[-1]
        #     for i in range(1, len(array)):
        #         sort_id[:-i] =\
        #             sort_id[:-i][np.argpartition(array[sort_id[:-i]], -1)]
        #         yield sort_id[-i - 1]

        # Rather than using a generator we could just straight out sort.
        # This is faster if we have to check more than log(n) points as
        # expanders before finding one
        def sort_generator(array):
            """Return the sorted array, largest element first."""
            return array.argsort()[::-1]

        # set of safe expanders
        G_safe = np.zeros(np.count_nonzero(s), dtype=np.bool)

        if not full_sets:
            # Sort, element with largest variance first
            sort_index = sort_generator(np.max(u[s, :] - l[s, :],
                                               axis=1))
        else:
            # Sort index is just an enumeration of all safe states
            sort_index = range(G_safe)

        for index in sort_index:
            if self.use_lipschitz:
                # Distance between current index point and all other unsafe
                # points
                d = cdist(self.inputs[s, :][[index], :],
                          self.inputs[~self.S, :])

                # Check if expander for all GPs
                for i in range(len(self.gps)):
                    # Skip evaluation if 'no' safety constraint
                    if self.fmin[i] == -np.inf:
                        continue
                    # Safety: u - L * d >= fmin
                    G_safe[index] =\
                        np.any(u[s, i][index] - self.liptschitz[i] * d >=
                               self.fmin[i])
                    # Stop evaluating if not expander according to one
                    # safety constraint
                    if not G_safe[index]:
                        break
            else:
                # Check if expander for all GPs
                for i in range(len(self.gps)):
                    # Skip evlauation if 'no' safety constraint
                    if self.fmin[i] == -np.inf:
                        continue

                    # Add safe point with its max possible value to the gp
                    self.add_new_data_point(self.inputs[s, :][index, :],
                                            u[s, i][index],
                                            gp=self.gps[i])

                    # Prediction of previously unsafe points based on that
                    mean2, var2 =\
                        self.gps[i]._raw_predict(self.inputs[~self.S])

                    # Remove the fake data point from the GP again
                    self.remove_last_data_point(gp=self.gps[i])

                    mean2 = mean2.squeeze()
                    var2 = var2.squeeze()
                    l2 = mean2 - beta * np.sqrt(var2)

                    # If any unsafe lower bound is suddenly above fmin then
                    # the point is an expander
                    G_safe[index] = np.any(l2 >= self.fmin[i])

                    # Break if one safety GP is not an expander
                    if not G_safe[index]:
                        break

            # Since we sorted by uncertainty and only the most
            # uncertain element gets picked by SafeOpt anyways, we can
            # stop after we found the first one
            if G_safe[index] and not full_sets:
                break

        # Update safe set (if full_sets is False this is at most one point
        self.G[s] = G_safe

    def compute_new_query_point(self):
        """
        Computes a new point at which to evaluate the function, based on the
        sets M and G.
        """
        # Get lower and upper bounds
        l, u = self.Q[:, :2].T

        MG = np.logical_or(self.M, self.G)
        value = u[MG] - l[MG]
        return self.inputs[MG][np.argmax(value)]

    def optimize(self, context=None):
        """Run one step of bayesian optimization.

        Parameters
        ----------
        context: ndarray
            A vector containing the current context
        """
        # Update confidence intervals based on current estimate
        if (context is not None and not
                np.all(self.inputs[0, -self.num_contexts:] == context)):
            self.update_confidence_intervals(context=context)
        # Update the sets
        self.compute_sets()
        # Get new input value
        x = self.compute_new_query_point()
        # Sample noisy output
        if context is None:
            value = self.function(x)
        else:
            value = self.function(x[:-self.num_contexts],
                                  x[-self.num_contexts:])

        value = np.asarray(value)
        value = np.atleast_1d(value.squeeze())

        # Add data point to the GP
        for i in range(len(self.gps)):
            if value[i] is not None:
                self.add_new_data_point(x, value[i], gp=self.gps[i])
        self.t += 1

        self.update_confidence_intervals(context=context)

    def get_maximum(self, context=None):
        """
        Return the current estimate for the maximum.

        Parameters
        ----------
        context: np.array
            A specific context that should be used. If None the current
            confidence intervals are used without modification.
        Returns
        -------
        x - ndarray
            Location of the maximum
        y - 0darray
            Maximum value

        """
        if (context is not None and not
                np.all(context == np.inputs[0, -self.num_contexts:])):
            self.update_confidence_intervals(context=context)
            self.compute_safe_set()

        l = self.Q[self.S, 0]

        max_id = np.argmax(l)
        return (self.inputs[self.S, :][max_id, :-self.num_contexts or None],
                l[max_id])
