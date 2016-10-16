"""
Classes that implement SafeOpt.

Authors: - Felix Berkenkamp (befelix at inf dot ethz dot ch)
         - Nicolas Carion (nicolas dot carion at gmail dot com)
"""

from __future__ import print_function, absolute_import, division

from .utilities import *

import sys
import numpy as np                          # ...
import scipy as sp
from GPy.util.linalg import dpotrs          # For rank-1 updates
from GPy.inference.latent_function_inference.posterior import Posterior
from collections import Sequence            # isinstance(...,Sequence)
from scipy.spatial.distance import cdist    # Efficient distance computation
from random import shuffle
from scipy.special import expit
from scipy.stats import norm

import logging


__all__ = ['SafeOpt', 'SafeOptSwarm', 'GaussianProcessOptimization']


# For python 2 compatibility
if sys.version_info[0] < 3:
    range = xrange


class GaussianProcessOptimization(object):
    """
    Base class for GP optimization.

    Handles common functionality.

    Parameters:
    -----------
    gp: GPy Gaussian process
    parameter_set: 2d-array
        List of parameters
    beta: float or callable
        A constant or a function of the time step that scales the confidence
        interval of the acquisition function.
    scaling: list of floats or "auto"
        A list used to scale the GP uncertainties to compensate for
        different input sizes. This should be set to the maximal variance of each kernel.
        You should probably leave this to "auto" unless your kernel is non stationnary
    """

    def __init__(self, gp, beta, num_contexts, scaling='auto'):
        super(GaussianProcessOptimization, self).__init__()

        if isinstance(gp, list):
            self.gps = gp
        else:
            self.gps = [gp]
        self.gp = self.gps[0]

        if hasattr(beta, '__call__'):
            # Beta is a function of t
            self.beta = beta
        else:
            # Assume that beta is a constant
            self.beta = lambda t: beta

        if scaling == 'auto':
            dummy_point = np.zeros((1, self.gps[0].input_dim))
            self.scaling = [self.gps[i].kern.K(
                dummy_point, dummy_point).squeeze() for i in range(len(self.gps))]
            self.scaling = np.asarray(self.scaling)
        else:
            self.scaling = np.asarray(scaling)
            if self.scaling.shape[0] != len(self.gps):
                raise ValueError(
                    "Error: the number of scaling values should be equal to the number of GPs")

        self._parameter_set = None
        self.bounds = None
        self.num_samples = 0
        self.num_contexts = num_contexts

        # Time step
        self.t = self.gp.X.shape[0]

    @property
    def data(self):
        """Return the data within the GP models."""
        x = self.gp.X.copy()
        y = np.empty((len(x), len(self.gps)), dtype=np.float)

        for i, gp in enumerate(self.gps):
            y[:, i] = gp.Y.squeeze()
        return x, y

    def plot(self, n_samples, axis=None, figure=None, plot_3d=False,
             **kwargs):
        """
        Plot the current state of the optimization.

        Parameters
        ----------
        n_samples: int
            How many samples to use for plotting
        axis: matplotlib axis
            The axis on which to draw (does not get cleared first)
        figure: matplotlib figure
            Ignored if axis is already defined
        plot_3d: boolean
            If set to true shows a 3D plot for 2 dimensional data
        """
        # Fix contexts to their current values
        if self.num_contexts > 0 and 'fixed_inputs' not in kwargs:
            kwargs.update(fixed_inputs=self.context_fixed_inputs)

        true_input_dim = self.gp.kern.input_dim - self.num_contexts
        if true_input_dim == 1 or plot_3d:
            inputs = np.zeros((n_samples ** true_input_dim, self.gp.input_dim))
            inputs[:, :true_input_dim] = linearly_spaced_combinations(self.bounds[:true_input_dim],
                                                                      n_samples)
        if not isinstance(n_samples, Sequence):
            n_samples = [n_samples] * len(self.bounds)

        if self.gp.input_dim - self.num_contexts == 1:
            # 2D plots with uncertainty
            plot_2d_gp(self.gp, inputs, figure=figure, axis=axis, **kwargs)
        else:
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

    def add_new_data_point(self, x, y, context=None, gp=None):
        """
        Add a new function observation to the GPs.

        Parameters
        ----------
        x: 2d-array
        y: 2d-array
        context: array_like
            The context(s) used for the data points
        gp: instance of GPy.model.GPRegression
            If specified, determines the GP to which we add the data point
        """
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)

        if self.num_contexts:
            context = np.asarray(context)
            x2 = np.empty((x.shape[0], x.shape[1] +
                           self.num_contexts), dtype=np.float)
            x2[:, :x.shape[1]] = x
            x2[:, x.shape[1]:] = context
            x = x2

        if gp is None:
            for i, gp in enumerate(self.gps):
                is_not_nan = ~np.isnan(y[:, i])
                if np.any(is_not_nan):
                    # Add data to GP
                    gp.set_XY(np.vstack([gp.X, x[is_not_nan, :]]),
                              np.vstack([gp.Y, y[is_not_nan, i]]))
        else:
            gp.set_XY(np.vstack([gp.X, x]),
                      np.vstack([gp.Y, y]))

        self.t += y.shape[1]

    def remove_last_data_point(self, gp=None):
        """Remove the data point that was last added to the GP.

        Parameters:
            gp: Instance of GPy.models.GPRegression
                The gp that the last data point should be removed from
        """
        if gp is None:
            for gp in self.gps:
                gp.set_XY(gp.X[:-1, :], gp.Y[:-1, :])
        else:
            gp.set_XY(gp.X[:-1, :], gp.Y[:-1, :])


class SafeOpt(GaussianProcessOptimization):
    """
    A class to maximize a function using the adapted or original SafeOpt alg.

    Parameters
    ----------
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
    scaling: list of floats or "auto"
        A list used to scale the GP uncertainties to compensate for
        different input sizes. This should be set to the maximal variance of each kernel.
        You should probably leave this to "auto" unless your kernel is non stationnary

    """

    def __init__(self, gp, parameter_set, fmin, lipschitz=None, beta=3.0,
                 num_contexts=0, threshold=0, scaling='auto'):

        super(SafeOpt, self).__init__(
            gp, beta, num_contexts, scaling)

        if self.num_contexts > 0:
            context_shape = (parameter_set.shape[0], self.num_contexts)
            self.inputs = np.hstack((parameter_set,
                                     np.zeros(context_shape,
                                              dtype=parameter_set.dtype)))
            self.parameter_set = self.inputs[:, :-self.num_contexts]
        else:
            self.inputs = self.parameter_set = parameter_set

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
        self.Q = np.empty((self.inputs.shape[0], 2 * len(self.gps)),
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

    @property
    def parameter_set(self):
        """Discrete parameter samples for Bayesian optimization."""
        return self._parameter_set

    @parameter_set.setter
    def parameter_set(self, parameter_set):
        self._parameter_set = parameter_set

        # Plotting bounds (min, max value
        self.bounds = list(zip(np.min(self._parameter_set, axis=0),
                               np.max(self._parameter_set, axis=0)))
        self.num_samples = [len(np.unique(self._parameter_set[:, i]))
                            for i in range(self._parameter_set.shape[1])]

    @property
    def context_fixed_inputs(self):
        """The fixed inputs for the current context"""
        n = self.gp.input_dim - 1
        nc = self.num_contexts
        if nc > 0:
            contexts = self.inputs[0, -self.num_contexts:]
            return list(zip(range(n, n - nc, -1), contexts))

    @property
    def context(self):
        """Return the current context variables."""
        if self.num_contexts:
            return self.inputs[0, -self.num_contexts:]

    @context.setter
    def context(self, context):
        """Set the current context and update confidence intervals.

        Parameters
        ----------
        context: ndarray
            New context that should be applied to the input parameters
        """
        if self.num_contexts:
            if context is None:
                raise ValueError('Need to provide value for context.')
            self.inputs[:, -self.num_contexts:] = context

    def update_confidence_intervals(self, context=None):
        """Recompute the confidence intervals form the GP.

        Parameters
        ----------
        context: ndarray
            Array that contains the context used to compute the sets
        """
        beta = self.beta(self.t)

        # Update context to current setting
        self.context = context

        # Iterate over all functions
        for i in range(len(self.gps)):
            # Evaluate acquisition function
            mean, var = self.gps[i].predict_noiseless(self.inputs)

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
            self.M[:] = False
            self.G[:] = False
            return

        # Set of possible maximisers
        # Maximizers: safe upper bound above best, safe lower bound
        self.M[:] = False
        self.M[self.S] = u[self.S] >= np.max(l[self.S])
        max_var = np.max(u[self.M] - l[self.M]) / self.scaling[0]

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
            s[s] = (np.max((u[s, :] - l[s, :]) / self.scaling, axis=1) >
                    max(max_var, self.threshold))

            if not np.any(s):
                # no need to evaluate any points as expanders in G, exit
                return

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
            sort_index = range(len(G_safe))

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
                    self.add_new_data_point(self.parameter_set[s, :][index, :],
                                            u[s, i][index],
                                            context=self.context,
                                            gp=self.gps[i])

                    # Prediction of previously unsafe points based on that
                    mean2, var2 =\
                        self.gps[i].predict_noiseless(self.inputs[~self.S])

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

    def get_new_query_point(self, ucb=False):
        """
        Computes a new point at which to evaluate the function, based on the
        sets M and G.

        Parameters
        ----------
        ucb: bool
            If True the safe-ucb criteria is used instead.

        Returns
        -------
        x: np.array
            The next parameters that should be evaluated.
        """
        if not np.any(self.S):
            raise EnvironmentError('There are no safe points to evaluate.')

        if ucb:
            max_id = np.argmax(self.Q[self.S, 1])
            x = self.inputs[self.S, :][max_id, :]
        else:
            # Get lower and upper bounds
            l = self.Q[:, ::2]
            u = self.Q[:, 1::2]

            MG = np.logical_or(self.M, self.G)
            value = np.max((u[MG] - l[MG]) / self.scaling, axis=1)
            x = self.inputs[MG, :][np.argmax(value), :]

        if self.num_contexts:
            return x[:-self.num_contexts]
        else:
            return x

    def optimize(self, context=None, ucb=False):
        """Run Safe Bayesian optimization and get the next parameters.

        Parameters
        ----------
        context: ndarray
            A vector containing the current context
        ucb: bool
            If True the safe-ucb criteria is used instead.

        Returns
        -------
        x: np.array
            The next parameters that should be evaluated.
        """
        # Update confidence intervals based on current estimate
        self.update_confidence_intervals(context=context)

        # Update the sets
        if ucb:
            self.compute_safe_set()
        else:
            self.compute_sets()

        return self.get_new_query_point(ucb=ucb)

    def get_maximum(self, context=None):
        """
        Return the current estimate for the maximum.

        Parameters
        ----------
        context: ndarray
            A vector containing the current context

        Returns
        -------
        x - ndarray
            Location of the maximum
        y - 0darray
            Maximum value

        Notes
        -----
        Uses the current context and confidence intervals!
        Run update_confidence_intervals first if you recently added a new data
        point.
        """
        self.update_confidence_intervals(context=context)

        # Compute the safe set (that's cheap anyways)
        self.compute_safe_set()

        # Return nothing if there are no safe points
        if not np.any(self.S):
            return None

        l = self.Q[self.S, 0]

        max_id = np.argmax(l)
        return (self.inputs[self.S, :][max_id, :-self.num_contexts or None],
                l[max_id])


class SafeOptSwarm(GaussianProcessOptimization):
    """
    A class to maximize a function using SafeOpt with a Particle Swarm Optimization
    heuristic.
    Note that it doesn't support the use of a Lipschitz constant

    You can set your logging level to INFO to get more insigt on the optimization process

    Parameters
    ----------
    gp: GPy Gaussian process
        A Gaussian process which is initialized with safe, initial data points.
        If a list of GPs then the first one is the value, while all the
        other ones are safety constraints.
    fmin: list of floats
        Safety threshold for the function value. If multiple safety constraints
        are used this can also be a list of floats (the first one is always
        the one for the values, can be set to None if not wanted)
    beta: float or callable
        A constant or a function of the time step that scales the confidence
        interval of the acquisition function.
    threshold: float
        The algorithm will not try to expand any points that are below this
        threshold. This makes the algorithm stop expanding points eventually.
    scaling: list of floats or "auto"
        A list used to scale the GP uncertainties to compensate for
        different input sizes. This should be set to the maximal variance of each kernel.
        You should probably leave this to "auto" unless your kernel is non stationnary
    bounds: pair of floats or list of pairs of floats
        If a list is given, then each pair represents the lower/upper bound in
        each dimension.
        Otherwise, we assume the same bounds for all dimensions
    swarm_size: int
        The number of particles in each of the optimization swarms

    """

    def __init__(self, gp, fmin, beta=3.0, num_contexts=0, threshold=0,
                 scaling='auto', bounds=(-5, 5), swarm_size=20):
        super(SafeOptSwarm, self).__init__(
            gp, beta, num_contexts, scaling)

        self.fmin = fmin
        if not isinstance(self.fmin, list):
            self.fmin = [self.fmin] * len(self.gps)
        self.fmin = np.atleast_1d(np.asarray(self.fmin).squeeze())

        # Safe set
        self.S = np.asarray(self.gps[0].X)

        self.swarm_size = swarm_size
        self.max_iters = 100  # number of swarm iterations

        if not isinstance(bounds, list):
            self.bounds = [bounds] * self.S.shape[1]
        else:
            self.bounds = bounds

        self.var_max = gp.kern.K(np.atleast_2d(
            gp.X[-1, :]), np.atleast_2d(gp.X[-1, :]))

        # These are estimates of the best lower bound, and its location
        self.best_lower_bound = -np.inf
        self.greedy_point = self.S[0, :]

    def _compute_penalty(self, slack, scaling):
        """
        Return the penalty associated to a constraint violation

        Parameters
        ----------
        slack: ndarray
            A vector corresponding to how much the constraint was violated
        scaling: float
            A float corresponding to the maximal variance of the corresponding GP
        Returns
        -------
        penalties - ndarray
            The value of the penalties
        """
        unsafe = slack < 0
        penalties = np.atleast_1d(np.clip(slack, -100000, 0))
        penalties[slack < -1 * scaling] = - penalties[slack < -1 * scaling]**2
        penalties[np.logical_and(unsafe, slack > -0.001 * scaling)] *= 2
        penalties[np.logical_and(unsafe, np.logical_and(
            slack <= -0.001 * scaling, slack > -0.1 * scaling))] *= 5
        penalties[np.logical_and(unsafe, np.logical_and(
            slack <= -0.1 * scaling, slack > -1 * scaling))] *= 10
        penalties[np.logical_and(
            unsafe, slack < -1 * scaling)] *= 300
        return penalties

    def _compute_particle_fitness(self, particles, swarm_type):
        """
        Return the value of the particles and the safety information.

        Parameters
        ----------
        particles: ndarray
            A vector containing the coordinates of the particles
        swarm_type: string
            A string corresponding to the swarm type. It can be any of the following:
                - "greedy" : estimate of the best lower bound
                - "expander" : find expender points
                - "maximizer" : find maximizer points
                - "safe_set" : special parameter to check only safety
        Returns
        -------
        values - ndarray
            The values of the particles
        global_safe - ndarray
            A boolean mask indicating safety status of all particles
            (note that in the case of a greedy swarm, this is not computed and we return a True mask)
        """
        beta = self.beta(self.t)

        # classify the particles points
        mean, var = self.gps[0].predict_noiseless(particles)
        mean = mean.squeeze()
        std_dev = np.sqrt(var.squeeze())

        # compute the confidence interval
        lower_bound = np.atleast_1d(mean - beta * std_dev)
        upper_bound = np.atleast_1d(mean + beta * std_dev)

        # the greedy swarm optimizes for the lower bound
        if swarm_type == 'greedy':
            return lower_bound, np.full(np.shape(lower_bound)[0], True, dtype=bool)

        # value we are optimizing for. Expanders and maximizers seek high
        # variance points
        values = std_dev / self.scaling[0]

        # define the interest function based on the particle type
        if swarm_type == 'maximizers':
            interest_function = expit(upper_bound - self.best_lower_bound)
        elif swarm_type == 'expanders' or swarm_type == 'safe_set':
            interest_function = np.ones(np.shape(values))
        else:
            # unknown particle type (shouldn't happen)
            raise AssertionError("Invalid swarm type")

        # boolean mask that tell if the particles are safe according to all gps
        global_safe = np.full(np.shape(particles)[0], True, dtype=bool)
        total_penalty = np.zeros(particles.shape[0])
        for i in range(len(self.gps)):
            if i == 0:
                cur_lower_bound = lower_bound  # reuse computation
            else:
                # classify using the current GP
                cur_mean, cur_var = self.gps[i].predict_noiseless(particles)
                cur_mean = cur_mean.squeeze()
                cur_std_dev = np.sqrt(cur_var.squeeze())
                cur_lower_bound = cur_mean - beta * cur_std_dev
                value = np.maximum(value, cur_std_dev / self.scaling[i])

            # if the current GP has no safety constrain, we skip it
            if self.fmin[i] == -np.inf:
                continue

            slack = np.atleast_1d(cur_lower_bound - self.fmin[i])

            # computing penalties
            penalties = np.zeros(np.shape(values))
            safe = slack >= 0
            unsafe = slack < 0
            global_safe = np.logical_and(safe, global_safe)

            total_penalty += self._compute_penalty(slack, self.scaling[i])

            if swarm_type == 'expanders':
                # check if the particles are expanders for the current gp
                interest_function = interest_function * (
                    norm.pdf(cur_lower_bound, loc=self.fmin[i]))

        # add penalty
        values += total_penalty

        # apply the mask for current interest function
        values *= interest_function

        # this swarm type is only interested in knowing whether the particles
        # are safe.
        if swarm_type == 'safe_set':
            values = lower_bound

        return values, global_safe

    def get_new_query_point(self, swarm_type):
        """
        Computes a new point at which to evaluate the function, depending
        on the swarm type.

        This function relies on a Particle Swarm Optimization (PSO) to find the
        optimum of the objective function (which depends on the swarm type).
        Parameters
        ----------
        swarm_type: string
            This parameter controls the type of point that should be found.
              -If set to "expanders", it will try to find a point that increases
               the safe set
              -If set to "maximizers", it will try to find a point that maximizes
               the objective function within the safe set
              -"Greedy" is a convenience parameter to retrieve an estimate of the best
               currently known point, given the observations already made.

        Returns
        -------
        global_best: np.array
            The next parameters that should be evaluated.
        max_std_dev: float
            The current standard deviation in the point to be evaluated
        """

        beta = self.beta(self.t)
        safe_size = np.shape(self.S)[0]
        input_dim = np.shape(self.S)[1]

        # Parameters of PSO
        c1 = 2  # coefficient of the regret term
        c2 = 2  # coefficient of the social term
        inertia_beginning = 1.2  # Inertia term at the beginning of optimization
        inertia_end = 0.1  # Inertia term at the end of optimization

        # Make sure the safe set is still safe
        lower_bound, safe = self._compute_particle_fitness(self.S, 'safe_set')
        unsafe = np.logical_not(safe)
        if not np.all(safe):
            logging.warning("Warning: %d unsafe points removed. Model might be violated" % (
                np.count(unsafe)))
            try:
                self.S = self.S[safe]
                safe_size = np.shape(self.S)[0]
            except:
                pass

        # init particles
        if swarm_type == 'greedy':
            # we pick particles u.a.r in the safe set
            particles = self.S[np.random.randint(
                safe_size, size=self.swarm_size - 3), :]
            # we make sure that we include in the initial particles the
            # following points (to speed up convergence):
            particles = np.append(particles, np.atleast_2d(
                self.greedy_point), axis=0)  # the previous greedy estimate
            particles = np.append(particles, np.atleast_2d(
                self.gp.X[-1, :]), axis=0)  # the last sampled point
            best_sampled_point = np.argmax(self.gp.Y)
            particles = np.append(particles, np.atleast_2d(
                self.gp.X[best_sampled_point, :]), axis=0)  # the best sampled point
        else:
            # we pick particles u.a.r in the safe set
            particles = self.S[np.random.randint(
                safe_size, size=self.swarm_size), :]

        # we now find a velocity that gets us away from the current points, but that doesn't get too far (ie we seek points that still highly correlated)
        # the following is a binary search
        velocity_found = False
        current_coef_up = 1000.
        current_coef_down = 0.
        while not velocity_found:
            mid = (current_coef_up + current_coef_down) / 2
            velocities = mid * np.random.rand(self.swarm_size, input_dim)

            # simulate one step of movement
            tmp_particles = inertia_beginning * velocities + particles
            for cur_dim in range(input_dim):
                tmp_particles[:, cur_dim] = np.clip(tmp_particles[:, cur_dim], self.bounds[
                                                    cur_dim][0], self.bounds[cur_dim][1])

            # compute correlation with current safe set.
            # TODO maybe make this dependent on all the kernels
            kernel_matrix = self.gps[0].kern.K(
                tmp_particles, self.S) / self.scaling[0]
            closest = np.max(kernel_matrix, axis=1)
            # make sure that the velocity is not too big (takes us out of safe
            # set)
            velocity_reasonable = np.min(closest) >= 0.9
            # make sure that the velocity is big enough (for exploration
            # purposes)
            velocity_enough = np.max(closest) <= 0.98
            velocity_found = (velocity_reasonable and velocity_enough) or abs(
                current_coef_down - current_coef_up) < 0.001
            if not velocity_enough:
                current_coef_down = mid
            else:
                current_coef_up = mid

        # compute initial fitness
        best_value, safe = self._compute_particle_fitness(
            particles, swarm_type)

        # initialization of the best estimates
        best_position = particles
        global_best = best_position[np.argmax(best_value), :]
        old_best = np.max(best_value)

        inertia_coef = (inertia_end - inertia_beginning) / self.max_iters
        for i in range(self.max_iters):
            # update velocities
            delta_global_best = (global_best - particles)
            delta_self_best = (best_position - particles)
            inertia = i * inertia_coef + inertia_beginning
            r1 = np.random.rand(self.swarm_size, input_dim)
            r2 = np.random.rand(self.swarm_size, input_dim)
            velocities = inertia * velocities + c1 * r1 * \
                delta_self_best + c2 * r2 * delta_global_best

            # clip
            velocities = np.clip(velocities, -4, 4)

            # update position
            particles = velocities + particles
            for cur_dim in range(input_dim):
                particles[:, cur_dim] = np.clip(particles[:, cur_dim], self.bounds[
                                                cur_dim][0], self.bounds[cur_dim][1])

            # compute fitness
            values, safe = self._compute_particle_fitness(
                particles, swarm_type)

            # find out which particles are improving
            improving = values > best_value

            # update whenever safety and improvement are guarenteed
            update_set = np.logical_and(improving, safe)
            best_value[update_set] = values[update_set]
            best_position[update_set] = particles[update_set]
            global_best = best_position[np.argmax(best_value), :]

        # expand safe set
        if swarm_type != 'greedy':
            selected_point_id = np.argmax(best_value)
            append = 0
            # compute correlation between new candidates and current safe set
            mat = self.gp.kern.K(best_position, np.append(
                self.S, best_position, axis=0)) / self.scaling[0]
            initial_safe = np.shape(self.S)[0]
            n, m = np.shape(mat)
            # this mask keeps track of the points that we have added in the
            # safe set to account for them when adding a new point
            mask = np.full(m, False, dtype=bool)
            mask[0:initial_safe] = True
            for j in range(n):
                # make sure correlation with old points is relatively low
                good_correlation = np.all(mat[j, mask] <= 0.95)
                # Note that we force addition of the highest variance point
                if j == selected_point_id or good_correlation:
                    pt = np.atleast_2d(best_position[j, :])
                    self.S = np.append(self.S, pt, axis=0)
                    append += 1
                    mask[initial_safe + j] = True
            logging.info("At the end of swarm %s, %d points were appended to safeset" % (
                swarm_type, append))
        else:
            # check whether we found a better estimate of the lower bound
            mean, var = self.gps[0].predict_noiseless(
                np.atleast_2d(self.greedy_point))
            mean = mean.squeeze()
            std_dev = np.sqrt(var.squeeze())
            lower_bound1 = mean - beta * std_dev
            if lower_bound1 < np.max(best_value):
                self.greedy_point = global_best

        if swarm_type == 'greedy':
            return global_best, np.max(best_value)

        # compute the variance of the point picked
        _, var = self.gps[0].predict_noiseless(np.atleast_2d(global_best))
        max_std_dev = np.sqrt(var.squeeze()) / self.scaling[0]
        for i in range(1, len(self.gps)):
            _, var = self.gps[i].predict_noiseless(np.atleast_2d(global_best))
            max_std_dev = np.max(
                np.sqrt(var.squeeze()) / self.scaling[i], max_std_dev)
        return global_best, max_std_dev

    def optimize(self):
        """Run Safe Bayesian optimization and get the next parameters.

        Parameters
        ----------

        Returns
        -------
        x: np.array
            The next parameters that should be evaluated.
        """

        # compute estimate of the lower bound
        self.greedy, self.best_lower_bound = self.get_new_query_point(
            'greedy')

        # Run both swarm:
        # Maximizers
        x_maxi, val_maxi = self.get_new_query_point('maximizers')
        # Expanders
        x_exp, val_exp = self.get_new_query_point('expanders')

        logging.info("The best maximizer has variance %f" % val_maxi)
        logging.info("The best expander has variance %f" % val_exp)
        logging.info("The greedy estimate of lower bound has value %f" %
                     self.best_lower_bound)

        if val_maxi > val_exp:
            x = x_maxi
        else:
            x = x_exp
        self.t += 1
        return x

    def get_maximum(self):
        """
        Return the current estimate for the maximum.

        Parameters
        ----------

        Returns
        -------
        x - ndarray
            Location of the maximum
        y - 0darray
            Maximum value

        """
        maxi = np.argmax(self.gp.Y)
        return self.gp.X[maxi, :], self.gp.Y[maxi]
