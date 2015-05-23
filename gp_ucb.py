from __future__ import print_function, absolute_import, division

__author__ = 'felix'

import numpy as np
from scipy.optimize import minimize
import GPy
import matplotlib.pyplot as plt


class GaussianProcessUCB:
    """A class to maximize a function using GP-UCB.

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
    def __init__(self, kernel, function, bounds):
        self.kernel = kernel
        self.gp = None

        self.bounds = bounds
        self.function = function

        self.x_max = None
        self.y_max = -np.inf

    def plot(self):
        """Plot the current state of the optimization."""
        if self.kernel.input_dim > 1:
            return None
        if self.gp is None:
            x = np.linspace(self.bounds[0][0], self.bounds[0][1], 50)
            K = self.kernel.Kdiag(x[:, None])
            std_dev = np.sqrt(K)
            plt.fill_between(x, -std_dev, std_dev, facecolor='blue', alpha=0.5)
            plt.show()
        else:
            self.gp.plot()

    def acquisition_function(self, x):
        """Computes -value and -gradient of the acquisition function at x."""
        beta = 2.
        x = np.atleast_2d(x)
        mu, var = self.gp.predict(x)
        dmu, dvar = self.gp.predictive_gradients(x)
        value = mu + beta * np.sqrt(var)

        gradient = dmu + 0.5 * beta * (var ** -0.5) * dvar.T

        if x.shape[1] > 1:
            gradient = gradient.squeeze()

        return -value.squeeze(), -gradient

    def compute_new_query_point(self):
        """Computes a new point at which to evaluate the function"""
        if self.gp is None:
            return np.mean(self.bounds, axis=1)

        x_max = 0.5
        v_max = -np.inf

        for i in range(50):

            x0 = np.array([np.random.uniform(b[0], b[1]) for b in self.bounds])

            res = minimize(self.acquisition_function, x0,
                           jac=True, bounds=self.bounds, method='L-BFGS-B')

            # Keep track of maximum
            if -res.fun >= v_max:
                v_max = -res.fun
                x_max = res.x
        return x_max

    def add_new_data_point(self, x, y):
        """Add a new function observation to the gp"""
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        if self.gp is None:
            self.gp = GPy.models.GPRegression(x, y, self.kernel)
            self.gp.likelihood.variance = 0.05**2
            self.gp.likelihood.variance.constrain_fixed(0.05**2)
        else:
            self.gp.set_XY(np.vstack([self.gp.X, x]),
                           np.vstack([self.gp.Y, y]))

        # if len(self.gp.Y) >= 5:
        #     self.gp.optimize()

    def optimize(self):
        """Run one step of bayesian optimization."""
        x = self.compute_new_query_point()
        value = self.function(x)
        self.add_new_data_point(x, value)

        if value > self.y_max:
            self.y_max = value
            self.x_max = x

def get_hyperparameters(function, kernel, bounds, N):
    """
    Optimize for hyperparameters by sampling a function from the uniform grid.

    Parameters
    ----------
    function: method
        Returns the function values, needs to be vectorized to accept 2-D
        arrays as inputs for each variable
    kernel: instance of GPy.kern.*
    bounds: array_like of tuples
        Each tuple consists of the upper and lower bounds of the variable
    N: integer
        Number of sample points per dimension, total = N ** len(bounds)
    """
    num_vars = len(bounds)

    test_vars = np.empty((num_vars, N), dtype=np.float)
    for row in range(num_vars):
        test_vars[row, :] = np.linspace(bounds[row][0], bounds[row][1], N)

    grid = np.array([x.ravel() for x in np.meshgrid(*test_vars)])
    values = function(*grid)

    gp = GPy.models.GPRegression(grid.T, values[:, None], kernel)
    gp.optimize()
    return gp

def f(x):
    x = np.asarray(x)
    return 2 * np.abs(x) + 0.05 * np.random.randn(*x.shape)

kernel = GPy.kern.RBF(input_dim=1, variance=2., lengthscale=1.0, ARD=True)
kernel = get_hyperparameters(f,  kernel, [(-1, 1)], 50).kern

a = GaussianProcessUCB(kernel, f, [(-2, 0)])

for i in range(10):
    a.optimize()

a.gp.plot()

print(a.gp)
# a.plot()
# a.optimize()
#
# # a.gp.plot()
# print(a.gp)
#
# print(a.x_max, a.y_max)

