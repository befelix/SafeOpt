# SafeOpt-robotics

This code implements an adapted version of the safe, Bayesian optimization algorithm, SafeOpt [1], that can be found in [2]. It can be used to automatically tune parameters subject to a safety constraint. The prefered way of citing this code is by referring to [2].

[1] Y. Sui, A. Gotovos, J. W. Burdick, and A. Krause, “Safe exploration for optimization with Gaussian processes” in Proc. of the International Conference on Machine Learning (ICML), 2015, pp. 997–1005.

[2] F. Berkenkamp, A. P. Schoellig, A. Krause, "..." in Proc. of the IEEE International Conference on Robotics and Automation (ICRA), 2016, (submitted).

## Installation
The easiest way to install the necessary python libraries is by installing pip (e.g. ```sudo apt-get install python-pip``` on Ubuntu) and running

```sudo pip install -r requirements.txt```

## Usage

For examples see the two interactive example ipython notebooks.

The algorithm is implemented in the ```gp_ucb.py``` file. Next to some helper functions, the class ```GaussianProcessSafeOpt``` implements the core algorithm. It can be initialized as

```GaussianProcessSafeOpt(function, gp, bounds, num_samples, fmin, lipschitz=None, beta=3.0)```

where ```function``` is the function that we are trying to optimize, ```gp``` is a Gaussian process from the ```GPy``` toolbox in <url>https://github.com/SheffieldML/GPy</url>. This Gaussian process should already include the points of the initial, safe set. The area over which the function is optimized is defined by ```bounds```, which is a sequence of lower and upper bounds for each variable (```[[x1_min, x1_max], [x2_min, x2_max]...]```). The number of points that are uniformly sampled within each of these intervals is given with ```num_samples```, which is either a constant or a list of discretization intervals for each variable. Lastly, fmin defines the safe lower bounds on the function values.

The class has two optional arguments: when ```lipschitz``` is not None, the original SafeOpt algorithm from [1] without self-contained intervals is used, instead of the modified algorithm from [2]. The confidence interval that is used can be specified by ```beta```, which can be a constant or a function of the iteration number.

There are two internal variables that influence the behavior of the algorithm:
```GaussianProcessSafeOpt.use_lipschitz``` determines whether to use the lipschitz constant or the Gaussian process confidence intervals to determine the sets of maximizers and expanders and ```GaussianProcessSafeOpt.use_constained_sets``` determines whether to enforce the sets of possible values to be contained in one another (as required by the proof in [1]).

Once the class is initialized, it's ```optimize``` method can be used to sample a new point. The ```plot``` method illustrates the Gaussian process intervals in 1 or 2 dimensions.

For a more detailed documentation see the class/method docstrings within the source code.

## License

The code is licenced under the MIT license and free to use by anyone without any restrictions.
