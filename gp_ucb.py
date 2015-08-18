"""
Classes that implement SafeOpt.

Author: Felix Berkenkamp (befelix at inf dot ethz dot ch)
"""

from __future__ import print_function, absolute_import, division

import numpy as np                          # ...
import GPy                                  # GPs
import matplotlib.pyplot as plt             # Plotting
from collections import Sequence            # isinstance(...,Sequence)
from matplotlib import cm                   # 3D plot colors
from scipy.spatial.distance import cdist    # Efficient distance computation
from scipy.interpolate import griddata      # For sampling GP functions
from mpl_toolkits.mplot3d import Axes3D     # Create 3D axes


def create_linearly_spaced_combinations(bounds, num_samples):
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


class GaussianProcessOptimization(object):
    """
    Base class for GP optimization.

    Handles common functionality.
    """
    def __init__(self, function, bounds, kernel, likelihood, num_samples,
                 beta):
        super(GaussianProcessOptimization, self).__init__()

        self.kernel = kernel
        self.gp = None

        self.bounds = bounds
        self.function = function

        self._likelihood = likelihood

        self.x_max = None
        self.y_max = -np.inf

        if hasattr(beta, '__call__'):
            # Beta is a function of t
            self.beta = beta
        else:
            # Assume that beta is a constant
            self.beta = lambda t: beta

        # Create test inputs for optimization
        if not isinstance(num_samples, Sequence):
            self.num_samples = [num_samples] * len(self.bounds)
        else:
            self.num_samples = num_samples
        self.inputs = create_linearly_spaced_combinations(self.bounds,
                                                          self.num_samples)

        self.optimization_finished = False

        # Time step
        self.t = 0

    @property
    def likelihood(self):
        if self.gp is None:
            return self._likelihood
        else:
            return self.gp.likelihood

    def plot(self, axis=None, n_samples=None, plot_3d=False):
        """
        Plot the current state of the optimization.

        Parameters
        ----------
        axis: matplotlib axis
            The axis on which to draw (does not get cleared first)
        """
        # 4d plots are tough...
        if self.kernel.input_dim > 2:
            return None

        if n_samples is None:
            inputs = self.inputs
            n_samples = self.num_samples
        else:
            inputs = create_linearly_spaced_combinations(self.bounds,
                                                         n_samples)
            if not isinstance(n_samples, Sequence):
                n_samples = [n_samples] * len(self.bounds)

        if self.kernel.input_dim > 1:   # 3D plot
            if self.gp is None:
                return None

            if plot_3d:
                fig = plt.figure()
                ax = Axes3D(fig)

                output, var = self.gp.predict(inputs)
                # output += 2 * np.sqrt(var)

                ax.plot_trisurf(inputs[:, 0], inputs[:, 1], output[:, 0],
                                cmap=cm.jet, linewidth=0.2, alpha=0.5)

                ax.plot(self.gp.X[:, 0], self.gp.X[:, 1], self.gp.Y[:, 0], 'o')

            else:
                # Use 2D level set plot, 3D is too slow
                fig = plt.figure()
                ax = fig.gca()
                output, var = self.gp.predict(inputs)
                if np.all(output == output[0, 0]):
                    plt.xlim(self.bounds[0])
                    plt.ylim(self.bounds[1])
                    return None
                c = ax.contour(np.linspace(self.bounds[0][0],
                                           self.bounds[0][1],
                                           n_samples[0]),
                               np.linspace(self.bounds[1][0],
                                           self.bounds[1][1],
                                           n_samples[1]),
                               output.reshape(*n_samples),
                               20)
                plt.colorbar(c)
                ax.plot(self.gp.X[:, 0], self.gp.X[:, 1], 'ob')

        else:   # 2D plots with uncertainty
            if self.gp is None:
                gram_diag = self.kernel.Kdiag(inputs)
                std_dev = self.beta(self.t) * np.sqrt(gram_diag)
                plt.fill_between(inputs[:, 0], -std_dev, std_dev,
                                 facecolor='blue',
                                 alpha=0.1)
            else:
                output, var = self.gp.predict(inputs[1:, :])
                output = output.squeeze()
                std_dev = self.beta(self.t) * np.sqrt(var.squeeze())
                plt.fill_between(inputs[1:, 0], output - std_dev,
                                 output + std_dev,
                                 facecolor='blue',
                                 alpha=0.3)
                plt.plot(inputs[1:, 0], output)
                plt.plot(self.gp.X, self.gp.Y, 'kx', ms=10, mew=3)
                # self.gp.plot(plot_limits=np.array(self.bounds).squeeze(),
                #              ax=axis)

    def add_new_data_point(self, x, y):
        """Add a new function observation to the GP."""
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        if self.gp is None:
            # Initialize GP
            inference_method = GPy.inference.latent_function_inference.\
                exact_gaussian_inference.ExactGaussianInference()
            self.gp = GPy.core.GP(X=x, Y=y, kernel=self.kernel,
                                  inference_method=inference_method,
                                  likelihood=self.likelihood)
        else:
            # Add data to GP
            self.gp.set_XY(np.vstack([self.gp.X, x]),
                           np.vstack([self.gp.Y, y]))

        # Increment time step
        self.t += 1

    def remove_last_data_point(self):
        """Remove the data point that was last added to the GP."""
        self.gp.set_XY(self.gp.X[:-1, :], self.gp.Y[:-1, :])
        self.t -= 1


class GaussianProcessUCB(GaussianProcessOptimization):
    """
    A class to maximize a function using GP-UCB.

    Parameters
    ---------
    function: object
        A function that returns the current value that we want to optimize.
    bounds: array_like of tuples
        An array of tuples where each tuple consists of the lower and upper
        bound on the optimization variable. E.g. for two variables, x1 and
        x2, with 0 <= x1 <= 3 and 2 <= x2 <= 4 bounds = [(0, 3), (2, 4)]
    kernel: instance of GPy.kern.*
    likelihood: instance of GPy.likelihoods.*
    num_samples: integer or list of integers
        Number of data points to use for the optimization and plotting
    beta: float or callable
        A constant or a function of the time step that scales the confidence
        interval of the acquisition function.

    """
    def __init__(self, function, bounds, kernel, likelihood, num_samples,
                 beta=3.0):
        super(GaussianProcessUCB, self).__init__(function, bounds, kernel,
                                                 likelihood, num_samples, beta)

    def acquisition_function(self, x, jac=True):
        """Computes -value and -gradient of the acquisition function at x."""
        beta = self.beta(self.t)
        x = np.atleast_2d(x)

        mu, var = self.gp.predict(x)
        value = mu + beta * np.sqrt(var)
        if not jac:
            return -value.squeeze()

        dmu, dvar = self.gp.predictive_gradients(x)
        gradient = dmu + 0.5 * beta * (var ** -0.5) * dvar.T

        if x.shape[1] > 1:
            gradient = gradient.squeeze()

        return -value.squeeze(), -gradient

    def compute_new_query_point(self):
        """
        Computes a new point at which to evaluate the function.

        The algorithm relies on discretizing all possible values and
        evaluating all of them. Fast, but memory inefficient.
        """
        # GPy is stupid in that it can only be initialized with data,
        # so just pick a random starting value in the middle
        if self.gp is None:
            return np.mean(self.bounds, axis=1)

        # Evaluate acquisition function
        values = self.acquisition_function(self.inputs, jac=False)

        return self.inputs[np.argmin(values), :]

    def optimize(self):
        """Run one step of bayesian optimization."""
        # Get new input value
        x = self.compute_new_query_point()
        # Sample noisy output
        value = self.function(x)
        # Add data point to the GP
        self.add_new_data_point(x, value)


def _nearest_neighbour(data, x):
    """Find the id of the nearest neighbour of x in data."""
    x = np.atleast_2d(x)
    return np.argmin(np.sum((data - x) ** 2, 1))


class GaussianProcessSafeUCB(GaussianProcessOptimization):
    """
    A class to maximize a function using GP-UCB.

    Parameters
    ---------
    function: object
        A function that returns the current value that we want to optimize.
    bounds: array_like of tuples
        An array of tuples where each tuple consists of the lower and upper
        bound on the optimization variable. E.g. for two variables, x1 and
        x2, with 0 <= x1 <= 3 and 2 <= x2 <= 4 bounds = [(0, 3), (2, 4)]
    kernel: instance of GPy.kern.*
    likelihood: instance of GPy.likelihoods.*
    num_samples: integer or list of integers
        Number of data points to use for the optimization and plotting
    fmin: float
        Safety threshold for the function value
    x0: float
        Initial point for the optimization
    beta: float or callable
        A constant or a function of the time step that scales the confidence
        interval of the acquisition function.

    """
    def __init__(self, function, bounds, kernel, likelihood, num_samples,
                 fmin, x0, lipschitz, beta=3.0):
        super(GaussianProcessSafeUCB, self).__init__(function, bounds, kernel,
                                                     likelihood, num_samples,
                                                     beta)
        self.fmin = fmin
        self.liptschitz = lipschitz

        # make sure initial point is in optimization points
        self.inputs = np.vstack([np.atleast_2d(x0), self.inputs])

        # Value intervals
        self.C = np.empty((self.inputs.shape[0], 2), dtype=np.float)
        self.C[:] = [-np.inf, np.inf]
        self.Q = self.C.copy()

        # Safe set
        self.S = np.zeros(self.inputs.shape[0], dtype=np.bool)

        # Get initial value and add it to the GP and the safe set
        value = self.function(self.inputs[0, :])
        if value < fmin:
            raise ValueError('Initial point is unsafe')
        else:
            self.add_new_data_point(self.inputs[0, :], value)
            self.S[0] = True

        self.C[self.S, 0] = self.fmin

        # Set of expanders and maximizers
        self.G = np.zeros_like(self.S, dtype=np.bool)
        self.M = self.G.copy()

        # Switch to use confidence intervals for safety
        self.use_confidence_safety = False
        self.use_confidence_sets = False

    def compute_sets(self, full_sets=False):
        """
        Compute the safe set of points

        Parameters:
        ----------
        full_sets: boolean
            Whether to compute the full set of expanders or whether to omit
            computations that are not relevant for running SafeOpt
            (This option is only useful for plotting purposes)
        """
        if self.gp is None:
            raise RuntimeError('self.gp should be initialized at this point.')

        beta = self.beta(self.t)

        # Evaluate acquisition function
        mean, var = self.gp.predict(self.inputs)
        mean = mean.squeeze()
        std_dev = np.sqrt(var.squeeze())

        # Update confidence intervals
        self.Q[:, 0] = mean - beta * std_dev
        self.Q[:, 1] = mean + beta * std_dev

        Q_l, Q_u = self.Q.T

        # Update confidence intervals if they're being used
        if not (self.use_confidence_sets and self.use_confidence_safety):
            # Convenient views on C (changing them will change C)
            C_l, C_u = self.C.T

            # Update value interval, make sure C(t+1) is contained in C(t)
            self.C[:, 0] = np.where(C_l < Q_l, np.min([Q_l, C_u], 0), C_l)
            self.C[:, 1] = np.where(C_u > Q_u, np.max([Q_u, C_l], 0), C_u)

        # Expand safe set
        if self.use_confidence_safety:
            self.S[:] = Q_l >= self.fmin
        else:
            # Euclidean distance between all safe and unsafe points
            d = cdist(self.inputs[self.S], self.inputs[~self.S])

            # Apply Lipschitz constant to determine new safe points
            self.S[~self.S] = \
                np.any(C_l[self.S, None] - self.liptschitz * d >= self.fmin, 0)

        # Set of possible maximisers

        # Get lower and upper bounds
        if self.use_confidence_sets:
            l, u = self.Q.T
        else:
            l, u = self.C.T

        # Maximizers: safe upper bound above best, safe lower bound
        self.M[:] = False
        self.M[self.S] = u[self.S] >= np.max(l[self.S])
        max_var = np.max(u[self.M] - l[self.M])

        # Optimistic set of possible expanders
        self.G[:] = False

        # For the run of the algorithm we do not need to calculate the
        # full set of potential expanders:
        # We can skip the ones already in M and ones that have lower
        # variance than the maximum variance in M, max_var.
        # Amongst the remaining ones we only need to find the
        # potential expander with maximum variance
        if not full_sets:
            # skip points in M
            s = np.logical_and(self.S, ~self.M)

            # Remove points with a variance that is too small
            s[s] = Q_u[s] - Q_l[s] > max_var
        else:
            s = self.S

        if self.use_confidence_sets:
            if not full_sets:
                # Sort, element with largest variance first
                sort_index = np.flipud((Q_u[s] - Q_l[s]).argsort())

                # Id to restore original order
                unsort_index = np.empty_like(sort_index)
                unsort_index[sort_index] = np.arange(len(sort_index))
            else:
                sort_index = unsort_index = self.S

            # set of safe expanders
            G_safe = np.zeros(np.count_nonzero(s), dtype=np.bool)

            # Iterate over all expander candidates
            for i, (x, y) in enumerate(zip(self.inputs[s, :][sort_index, :],
                                           Q_u[s][sort_index])):
                # Add safe point with it's max possible value to the gp
                self.add_new_data_point(x, y)

                # Prediction of unsafe points based on that
                mean2, var2 = self.gp.predict(self.inputs[~self.S])

                # Remove the fake data point from the GP again
                self.remove_last_data_point()

                mean2 = mean2.squeeze()
                var2 = var2.squeeze()
                l2 = mean2 - beta * np.sqrt(var2)

                # If the unsafe lower bound is suddenly above fmin: expander
                if np.any(l2 >= self.fmin):
                    G_safe[i] = True
                    # Since we sorted by uncertainty and only the most
                    # uncertain element gets picked by SafeOpt anyways, we can
                    # stop after we found the first one
                    if not full_sets:
                        break

            self.G[s] = G_safe[unsort_index]
        else:
            # Doing the same partial-prediction stuff as above is possible,
            # but not implemented since numpy is super fast anyways
            d = cdist(self.inputs[s], self.inputs[~self.S])
            self.G[s] = np.any(
                C_u[s, None] - self.liptschitz * d >= self.fmin, 1)

    def compute_new_query_point(self):
        """
        Computes a new point at which to evaluate the function, based on the
        sets M and G.
        """
        # Get lower and upper bounds
        if self.use_confidence_sets:
            l, u = self.Q.T
        else:
            l, u = self.C.T

        MG = np.logical_or(self.M, self.G)
        value = u[MG] - l[MG]
        return self.inputs[MG][np.argmax(value)]

    def optimize(self):
        """Run one step of bayesian optimization."""
        # Update the sets
        self.compute_sets()
        # Get new input value
        x = self.compute_new_query_point()
        # Sample noisy output
        value = self.function(x)
        # Add data point to the GP
        self.add_new_data_point(x, value)

        # Keep track of best observed value (not necessarily the maximum)
        if value > self.y_max:
            self.y_max = value
            self.x_max = np.atleast_1d(x)

    def get_maximum(self):
        """
        Return the current estimate for the maximum.

        Returns:
        --------
        x - ndarray
            Location of the maximum
        y - 0darray
            Maximum value

        """
        if self.use_confidence_sets:
            max_id = np.argmax(self.Q[:, 0])
            return self.inputs[max_id, :], self.Q[max_id, 0]
        else:
            max_id = np.argmax(self.C[:, 0])
            return self.inputs[max_id, :], self.C[max_id, 0]


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
    inputs = create_linearly_spaced_combinations(bounds, num_samples)
    output = function(inputs)

    inference_method = GPy.inference.latent_function_inference.\
        exact_gaussian_inference.ExactGaussianInference()

    gp = GPy.core.GP(X=inputs, Y=output[:, None],
                     kernel=kernel,
                     inference_method=inference_method,
                     likelihood=likelihood)
    gp.optimize()
    return gp.kern, gp.likelihood


def sample_gp_function(kernel, bounds, noise, num_samples):
    """
    Sample a function from a gp with corresponding kernel within its bounds.

    Parameters
    ----------
    kernel: instance of GPy.kern.*
    bounds: list of tuples
        [(x1_min, x1_max), (x2_min, x2_max), ...]
    noise: float
        Variance of the observation noise of the GP function
    num_samples: int or list
        If integer draws the corresponding number of samples in all
        dimensions and test all possible input combinations. If a list then
        the list entries correspond to the number of linearly spaced samples of
        the corresponding input

    Returns
    -------
    function: object
        A function that takes as inputs new locations x to be evaluated and
        returns the corresponding noisy function values
    """
    inputs = create_linearly_spaced_combinations(bounds, num_samples)
    output = np.random.multivariate_normal(np.zeros(inputs.shape[0]),
                                           kernel.K(inputs))

    def evaluate_gp_function(x, return_data=False):
        if return_data:
            return inputs, output
        x = np.atleast_2d(x)
        return griddata(inputs, output, x, method='linear') + \
            np.sqrt(noise) * np.random.randn(x.shape[0], 1)

    return evaluate_gp_function
