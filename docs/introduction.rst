This code implements an adapted version of the safe, Bayesian optimization algorithm, SafeOpt [1,2]. It also provides an implementation for the original algorithm in [3]. The code can be used to automatically optimize a performance measures subject to a safety constraint by adapting parameters. The prefered way of citing this code is by referring to [1, 2].

| A video of the experiments can be found at https://youtu.be/GiqNQdzc5TI
| The source code is available at https://github.com/befelix/SafeOpt
| 

[1] F. Berkenkamp, A. P. Schoellig, A. Krause, "Safe Controller Optimization for Quadrotors with Gaussian Processes" in Proc. of the IEEE International Conference on Robotics and Automation (ICRA), 2016, `arXiv:1509.01066 [cs.RO] <http://arxiv.org/abs/1509.01066>`_

[2] F. Berkenkamp, A. Krause, A. P. Schoellig, "Bayesian Optimization with Safety Constraints: Safe and Automatic Parameter Tuning in Robotics", ArXiv, 2016, `arXiv:1509.01066 [cs.RO] <http://arxiv.org/abs/1602.04450>`_

[3] Y. Sui, A. Gotovos, J. W. Burdick, and A. Krause, “Safe exploration for optimization with Gaussian processes” in Proc. of the International Conference on Machine Learning (ICML), 2015, pp. 997–1005. `[PDF] <https://las.inf.ethz.ch/files/sui15icml-long.pdf>`_