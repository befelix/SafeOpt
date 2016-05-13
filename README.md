# SafeOpt - Safe Bayesian Optimization

This code implements an adapted version of the safe, Bayesian optimization algorithm, SafeOpt [1,2]. It also provides an implementation for the original algorithm in [3]. The code can be used to automatically optimize a performance measures subject to a safety constraint by adapting parameters. The prefered way of citing this code is by referring to [1, 2].

###### A video of the experiments in [1]:
<a href="http://www.youtube.com/watch?feature=player_embedded&v=GiqNQdzc5TI" target="_blank"><img src="http://img.youtube.com/vi/GiqNQdzc5TI/0.jpg" alt="SafeOpt video" width="240" height="180" border="0" /></a>

[1] F. Berkenkamp, A. P. Schoellig, A. Krause, "Safe Controller Optimization for Quadrotors with Gaussian Processes" in Proc. of the IEEE International Conference on Robotics and Automation (ICRA), 2016 <a href="http://arxiv.org/abs/1509.01066" target="_blank">arXiv:1509.01066 [cs.RO]</a>

[2] F. Berkenkamp, A. Krause, A. P. Schoellig, "Bayesian Optimization with Safety Constraints: Safe and Automatic Parameter Tuning in Robotics", ArXiv, 2016, <a href="http://arxiv.org/abs/1602.04450" target=_blank>arXiv:1509.01066 [cs.RO]</a>

[3] Y. Sui, A. Gotovos, J. W. Burdick, and A. Krause, “Safe exploration for optimization with Gaussian processes” in Proc. of the International Conference on Machine Learning (ICML), 2015, pp. 997–1005. <a href="https://las.inf.ethz.ch/files/sui15icml-long.pdf" target=_blank>[PDF]</a>


## Installation
The easiest way to install the necessary python libraries is by installing pip (e.g. ```sudo apt-get install python-pip``` on Ubuntu) and running

```sudo pip install -r requirements.txt```

## Usage

<b>The easiest way to get familiar with the library is to take a look at the example ipython notebooks!</b>

Make sure that the ```ipywidgets``` module is installed.<br>
All functions and classes are documented on <a href="http://safeopt.readthedocs.org/en/latest/" target="_blank">Read The Docs</a>.
<br><br>

##### Details
The algorithm is implemented in the ```gp_opt.py``` file. Next to some helper
functions, the class ```SafeOpt``` implements the core algorithm. It can be
initialized as

```SafeOpt(function, gp, parameter_set, fmin, lipschitz=None, beta=3.0)```

where ```function``` is the function that we are trying to optimize, ```gp``` is a Gaussian process from the ```GPy``` toolbox in <url>https://github.com/SheffieldML/GPy</url>. This Gaussian process should already include the points of the initial, safe set. The ```parameter_set``` is a 2d-array of sampling locations for the GP, which is used to compute new evaluation points. It can, for example, be create with the ```linearly_spaced_combinations``` function in the safeopt library. Lastly, fmin defines the safe lower bounds on the function values.

The class several optional arguments: The ```lipschitz``` argument can be used to specify the Lipschitz constant to determine the set of expanders. If it is not None, the algorithm in [1] is used to determine expanders directly with the confidence itnervals. The confidence interval that is used can be specified by ```beta```, which can be a constant or a function of the iteration number.

Once the class is initialized, its ```optimize``` method can be used to sample a new point. The ```plot``` method illustrates the Gaussian process intervals in 1 or 2 dimensions.

For a more detailed documentation see the class/method docstrings within the source code.

## License

The code is licenced under the MIT license and free to use by anyone without any restrictions.
