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
    def __init__(self, function, bounds, kernel, likelihood):
        self.kernel = kernel
        self.gp = None

        self.bounds = bounds
        self.function = function

        self._likelihood = likelihood

        self.x_max = None
        self.y_max = -np.inf

    @property
    def likelihood(self):
        if self.gp is None:
            return self._likelihood
        else:
            return self.gp.likelihood

    def plot(self):
        """Plot the current state of the optimization."""
        if self.kernel.input_dim > 1:
            return None
        if self.gp is None:
            x = np.linspace(self.bounds[0][0], self.bounds[0][1], 50)
            K = self.kernel.Kdiag(x[:, None])
            std_dev = np.sqrt(K)
            plt.fill_between(x, -std_dev, std_dev, facecolor='blue',
                             alpha=0.5)
            plt.show()
        else:
            plt.close()
            self.gp.plot(plot_limits=np.array(self.bounds).squeeze())
            plt.show()

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

        num_vars = len(self.bounds)
        num_samples = [1000] * num_vars

        # Create linearly spaced test inputs
        inputs = [np.linspace(b[0], b[1], n) for b, n in zip(self.bounds,
                                                             num_samples)]

        # Convert to 2-D array
        inputs = np.array([x.ravel() for x in np.meshgrid(*inputs)]).T

        # Evaluate acquisition function
        values = self.acquisition_function(inputs, jac=False)

        return inputs[np.argmin(values), :]

    def add_new_data_point(self, x, y):
        """Add a new function observation to the gp"""
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        if self.gp is None:
            inference_method = GPy.inference.latent_function_inference.\
                exact_gaussian_inference.ExactGaussianInference()
            self.gp = GPy.core.GP(X=x,Y=y, kernel=self.kernel,
                                  inference_method=inference_method,
                                  likelihood=self.likelihood)
        else:
            self.gp.set_XY(np.vstack([self.gp.X, x]),
                           np.vstack([self.gp.Y, y]))

        # if len(self.gp.Y) >= 5:
        #     self.gp.optimize()

    def optimize(self):
        """Run one step of bayesian optimization."""
        x = self.compute_new_query_point_discrete()
        value = self.function(x)
        self.add_new_data_point(x, value)

        if value > self.y_max:
            self.y_max = value
            self.x_max = x

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
    num_vars = len(bounds)

    # linearly space test inputs
    test_vars = np.empty((num_vars, num_samples), dtype=np.float)
    for row in range(num_vars):
        test_vars[row, :] = np.linspace(bounds[row][0], bounds[row][1],
                                        num_samples)

    # Store test inputs in the appropriate form
    inputs = np.array([x.ravel() for x in np.meshgrid(*test_vars)])
    output = function(*inputs)

    inference_method = GPy.inference.latent_function_inference.\
        exact_gaussian_inference.ExactGaussianInference()

    gp = GPy.core.GP(X=inputs.T, Y=output[:, None],
                     kernel=kernel,
                     inference_method=inference_method,
                     likelihood=likelihood)
    gp.optimize()
    return gp.kern, gp.likelihood


if __name__ == '__main__':

    noise_std_dev = 0.05

    # Optimization function
    def fun(x):
        x = np.asarray(x)
        return x ** 2 + noise_std_dev * np.random.randn(*x.shape)

    # Set fixed Gaussian measurement noise
    likelihood = GPy.likelihoods.gaussian.Gaussian(variance=noise_std_dev**2)
    likelihood.constrain_fixed()

    # Define Kernel
    kernel = GPy.kern.RBF(input_dim=1, variance=2., lengthscale=1.0, ARD=True)

    # Optimize hyperparameters
    bounds = [(-0.9, 1)]
    kernel, likelihood = get_hyperparameters(fun, bounds, 100,
                                             kernel, likelihood)

    # Init UCB algorithm
    gp_ucb = GaussianProcessUCB(fun, bounds, kernel, likelihood)

    # Optimize
    for i in range(7):
        gp_ucb.optimize()
        gp_ucb.plot()
        a = raw_input('wait')

    # Show results
    print(gp_ucb.gp)
    print('maximum at x={0} with value of y={1}'.format(gp_ucb.x_max,
                                                        gp_ucb.y_max))