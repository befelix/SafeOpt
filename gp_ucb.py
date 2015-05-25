from __future__ import print_function, absolute_import, division

__author__ = 'felix'

import numpy as np                    # ...
from scipy.optimize import minimize   # Optimization for continuous case
import GPy                            # GPs
import matplotlib.pyplot as plt       # Plotting
from collections import Sequence      # isinstance(...,Sequence)
from matplotlib import cm             # 3D plot colors
from scipy.spatial.distance import cdist, squareform


def create_linear_spaced_combinations(bounds, num_samples):
    """
    Return 2-D array with all linearly spaced combinations with the bounds.

    Parameters
    ----------
    bounds: sequence of tuples
        The bounds for the variables, [(x1_min, x1_max), (x2_min, x2_max), ...]
    num_samples: integer or array_like
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
    def __init__(self, function, bounds, kernel, likelihood):
        super(GaussianProcessOptimization, self).__init__()
        self.kernel = kernel
        self.gp = None

        self.bounds = bounds
        self.function = function

        self._likelihood = likelihood

        self.x_max = None
        self.y_max = -np.inf

        self.optimization_finished = False

    @property
    def likelihood(self):
        if self.gp is None:
            return self._likelihood
        else:
            return self.gp.likelihood

    def plot(self, axis=None):
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

        if self.kernel.input_dim > 1:   # 3D plot
            if self.gp is None:
                return None
            else:
                fig = plt.figure()
                ax = fig.gca(projection='3d')

                inputs = create_linear_spaced_combinations(self.bounds, 100)
                output, var = self.gp.predict(inputs)
                output += 2 * np.sqrt(var)

                ax.plot_trisurf(inputs[:, 0], inputs[:, 1], output[:, 0],
                                cmap=cm.jet, linewidth=0.2, alpha=0.5)

                ax.plot(self.gp.X[:, 0], self.gp.X[:, 1], self.gp.Y[:, 0], 'o')
        else:   # 2D plots with uncertainty
            if self.gp is None:
                x = np.linspace(self.bounds[0][0], self.bounds[0][1], 50)
                K = self.kernel.Kdiag(x[:, None])
                std_dev = np.sqrt(K)
                plt.fill_between(x, -std_dev, std_dev, facecolor='blue',
                                 alpha=0.5, axis=axis)
            else:
                self.gp.plot(plot_limits=np.array(self.bounds).squeeze(),
                             ax=axis)

    def add_new_data_point(self, x, y):
        """Add a new function observation to the gp"""
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        if self.gp is None:
            # Initialize GP
            inference_method = GPy.inference.latent_function_inference.\
                exact_gaussian_inference.ExactGaussianInference()
            self.gp = GPy.core.GP(X=x,Y=y, kernel=self.kernel,
                                  inference_method=inference_method,
                                  likelihood=self.likelihood)
        else:
            # Add data to GP
            self.gp.set_XY(np.vstack([self.gp.X, x]),
                           np.vstack([self.gp.Y, y]))


class GaussianProcessUCB(GaussianProcessOptimization):
    """
    A class to maximize a function using GP-UCB.

    Parameters
    ---------
    kernel: instance of Gpy.kern.*
    function: object
        A function that returns the current value that we want to optimize.
    bounds: array_like of tuples
        An array of tuples where each tuple consists of the lower and upper
        bound on the optimization variable. E.g. for two variables, x1 and
        x2, with 0 <= x1 <= 3 and 2 <= x2 <= 4 bounds = [(0, 3), (2, 4)]

    """
    def __init__(self, function, bounds, kernel, likelihood):
        super(GaussianProcessUCB, self).__init__(function, bounds, kernel,
                                                 likelihood)

    def acquisition_function(self, x, jac=True):
        """Computes -value and -gradient of the acquisition function at x."""
        beta = 3.
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
        """Computes a new point at which to evaluate the function."""
        # GPy is stupid in that it can only be initialized with data,
        # so just pick a random starting value in the middle
        if self.gp is None:
            return np.mean(self.bounds, axis=1)

        x_max = 0.5
        v_max = -np.inf

        # TODO: This needs to be less cured
        for i in range(50):
            x0 = np.array([np.random.uniform(b[0], b[1]) for b in self.bounds])

            res = minimize(self.acquisition_function, x0,
                           jac=True, bounds=self.bounds, method='L-BFGS-B')

            # Keep track of maximum
            if -res.fun >= v_max:
                v_max = -res.fun
                x_max = res.x
        return x_max

    def compute_new_query_point_discrete(self):
        """
        Computes a new point at which to evaluate the function.

        The algorithm relies on discretizing all possible values and
        evaluating all of them.
        """
        # GPy is stupid in that it can only be initialized with data,
        # so just pick a random starting value in the middle
        if self.gp is None:
            return np.mean(self.bounds, axis=1)

        inputs = create_linear_spaced_combinations(self.bounds, 200)

        # Evaluate acquisition function
        values = self.acquisition_function(inputs, jac=False)

        return inputs[np.argmin(values), :]

    def optimize(self):
        """Run one step of bayesian optimization."""
        x = self.compute_new_query_point_discrete()
        value = self.function(x)
        self.add_new_data_point(x, value)

        if value > self.y_max:
            self.y_max = value
            self.x_max = np.atleast_1d(x)

def _nearest_neighbour(data, x):
    """Find the id of the nearest neighbour of x in data."""
    x = np.atleast_2d(x)
    return np.argmin(np.sum((data - x) ** 2, 1))


class GaussianProcessSafeUCB(GaussianProcessOptimization):
    """
    A class to maximize a function using GP-UCB.

    Parameters
    ---------
    kernel: instance of Gpy.kern.*
    function: object
        A function that returns the current value that we want to optimize.
    bounds: array_like of tuples
        An array of tuples where each tuple consists of the lower and upper
        bound on the optimization variable. E.g. for two variables, x1 and
        x2, with 0 <= x1 <= 3 and 2 <= x2 <= 4 bounds = [(0, 3), (2, 4)]

    """
    def __init__(self, function, bounds, kernel, likelihood, fmin, x0, L):
        super(GaussianProcessSafeUCB, self).__init__(function, bounds, kernel,
                                                     likelihood)
        self.inputs = create_linear_spaced_combinations(self.bounds, 1000)
        self.f_min = fmin
        self.liptschitz = L

        self.C = np.empty((self.inputs.shape[0], 2), dtype=np.float)
        self.C[:, 0] = self.f_min
        self.C[:, 1] = np.inf

        self.Q = np.empty_like(self.C)
        self.Q[:, 0] = -np.inf
        self.Q[:, 1] = np.inf

        x0_id = _nearest_neighbour(self.inputs, x0)
        value = self.function(self.inputs[x0_id])
        if value < fmin:
            raise ValueError('Initial point is unsafe')
        self.add_new_data_point(self.inputs[x0_id], value)

        self.S = np.zeros(self.inputs.shape[0], dtype=np.bool)
        self.S[x0_id] = True

    def compute_new_query_point_discrete(self):
        """
        Computes a new point at which to evaluate the function.

        The algorithm relies on discretizing all possible values and
        evaluating all of them.
        """
        # GPy is stupid in that it can only be initialized with data,
        # so just pick a random starting value in the middle
        if self.gp is None:
            raise RuntimeError('self.gp should be initialized at this point.')

        beta = 3.

        # Evaluate acquisition function
        mean, var = self.gp.predict(self.inputs)

        mean = mean.squeeze()
        var = var.squeeze()

        # Update confidence intervals
        self.Q[:, 0] = mean - beta * np.sqrt(var)
        self.Q[:, 1] = mean + beta * np.sqrt(var)

        # Convenient views on C (changing them will change C)
        l = self.C[:, 0]
        u = self.C[:, 1]

        # Update value interval
        self.C[:, 0] = np.where(l < self.Q[:, 0], self.Q[:, 0], l)
        self.C[:, 1] = np.where(u > self.Q[:, 1], self.Q[:, 1], u)

        # Expand safe set
        # Euclidean distance between all safe and unsafe points
        d = cdist(self.inputs[self.S], self.inputs[~self.S])
        # Apply Lipschitz constant to determine new safe points
        self.S[~self.S] = np.any(
            l[self.S, None] - self.liptschitz * d >= self.f_min, 0)

        # Optimistic set of possible expanders
        G = np.zeros_like(self.S)
        # TODO: Use the relevant parts of the previously computed d here
        d = cdist(self.inputs[self.S], self.inputs[~self.S])
        G[self.S] = np.any(
            u[self.S, None] - self.liptschitz * d >= self.f_min, 1)

        # Set of possible maximisers
        M = np.zeros_like(self.S)
        M[self.S] = u[self.S] >= np.max(l[self.S])

        MG = np.logical_or(M, G)
        value = u[MG] - l[MG]
        return self.inputs[MG][np.argmax(value)]

    def optimize(self):
        """Run one step of bayesian optimization."""
        x = self.compute_new_query_point_discrete()
        value = self.function(x)
        self.add_new_data_point(x, value)

        if value > self.y_max:
            self.y_max = value
            self.x_max = np.atleast_1d(x)


def get_hyperparameters(function, bounds, num_samples, kernel,
                        likelihood=GPy.likelihoods.gaussian.Gaussian()):
    """
    Optimize for hyperparameters by sampling a function from the uniform grid.

    Parameters
    ----------
    function: method
        Returns the function values, needs to be vectorized to accept 2-D
        arrays as inputs for each variable
    bounds: array_like of tuples
        Each tuple consists of the upper and lower bounds of the variable
    N: integer
        Number of sample points per dimension, total = N ** len(bounds)
    kernel: instance of GPy.kern.*
    likelihood: instance of GPy.likelihoods.*
        Defaults to GPy.likelihoods.gaussian.Gaussian()

    Returns
    -------
    kernel: instance of GPy.kern.*
    likelihood: instance of GPy.likelihoods.*
    """
    inputs = create_linear_spaced_combinations(bounds, num_samples)
    output = function(inputs)

    inference_method = GPy.inference.latent_function_inference.\
        exact_gaussian_inference.ExactGaussianInference()

    gp = GPy.core.GP(X=inputs, Y=output[:, None],
                     kernel=kernel,
                     inference_method=inference_method,
                     likelihood=likelihood)
    gp.optimize()
    return gp.kern, gp.likelihood


def sample_gp_function(kernel, bounds, noise_std_dev, num_samples):
    """
    Sample a function from a gp with corresponding kernel within its bounds.

    Parameters
    ----------
    kernel: instance of GPy.kern.*
    bounds: list of tuples
        [(x1_min, x1_max), (x2_min, x2_max), ...]

    Returns
    -------
    function: object
        A function that takes as inputs new locations x to be evaluated and
        returns the corresponding noise function values
    """
    from scipy.interpolate import griddata

    inputs = create_linear_spaced_combinations(bounds, num_samples)
    output = np.random.multivariate_normal(np.zeros(inputs.shape[0]),
                                           kernel.K(inputs))

    def evaluate_gp_function(x, return_data=False):
        if return_data:
            return inputs, output
        x = np.atleast_2d(x)
        return griddata(inputs, output, x, method='linear') + \
               noise_std_dev * np.random.randn(x.shape[0], 1)

    return evaluate_gp_function