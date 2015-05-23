from __future__ import print_function, absolute_import, division

__author__ = 'felix'

import numpy as np
from scipy.optimize import minimize
import GPy
import matplotlib.pyplot as plt


class GaussianProcessUCB:
    """A class to optimize a function using GP-UCB.

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

    def acquisition_function(self, x):
        """Computes -value and -gradient of the acquisition function at x."""
        beta = 2.
        x = np.atleast_2d(x)
        mu, var = self.gp.predict(x)
        dmu, dvar = self.gp.predictive_gradients(x)
        value = mu + beta * np.sqrt(var)

        gradient = dmu + 0.5 * beta * (var ** -0.5) * dvar.T

        return -value.squeeze(), -gradient.squeeze()

    def compute_new_query_point(self):
        """Computes a new point at which to evaluate the function"""
        if self.gp is None:
            return np.mean(self.bounds, axis=1)

        x_max = 0.5
        v_max = -np.inf

        for i in range(50):

            x0 = [np.random.uniform(b[0], b[1]) for b in self.bounds]

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
        value = self.function(x) + 0.05 * np.random.randn()
        self.add_new_data_point(x, value)

        if value > self.y_max:
            self.y_max = value
            self.x_max = x

def f(x):
    return 2 * abs(x)

kernel = GPy.kern.RBF(input_dim=2, variance=1000., lengthscale=1.0)
a = GaussianProcessUCB(kernel, f, [(-2, 0), (-2, 0)])

for i in range(20):
    a.optimize()

a.gp.plot()
print(a.gp)

print(a.x_max, a.y_max)

