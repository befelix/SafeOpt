====================================
SafeOpt - Safe Bayesian Optimization
====================================

.. image:: https://travis-ci.org/befelix/SafeOpt.svg?branch=master
  :target: https://travis-ci.org/befelix/SafeOpt
  :alt: Build Status
.. image:: https://readthedocs.org/projects/safeopt/badge/?version=latest
  :target: http://safeopt.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status

This code implements an adapted version of the safe, Bayesian optimization algorithm, SafeOpt [1]_, [2]_. It also provides a more scalable implementation based on [3]_ as well as an implementation for the original algorithm in [4]_.
The code can be used to automatically optimize a performance measures subject to a safety constraint by adapting parameters.
The prefered way of citing this code is by referring to [1] or [2].

.. image:: http://img.youtube.com/vi/GiqNQdzc5TI/0.jpg
  :target: http://www.youtube.com/watch?feature=player_embedded&v=GiqNQdzc5TI
  :alt: Youtube video

.. [1] F. Berkenkamp, A. P. Schoellig, A. Krause,
  `Safe Controller Optimization for Quadrotors with Gaussian Processes <http://arxiv.org/abs/1509.01066>`_
  in Proc. of the IEEE International Conference on Robotics and Automation (ICRA), 2016, pp. 491-496.

.. [2] F. Berkenkamp, A. Krause, A. P. Schoellig,
  `Bayesian Optimization with Safety Constraints: Safe and Automatic Parameter Tuning in Robotics  <http://arxiv.org/abs/1602.04450>`_,
  ArXiv, 2016, arXiv:1602.04450 [cs.RO].

.. [3] Rikky R.P.R. Duivenvoorden, Felix Berkenkamp, Nicolas Carion, Andreas Krause, Angela P. Schoellig,
  `Constrained Bayesian optimization with Particle Swarms for Safe Adaptive Controller Tuning <http://www.dynsyslab.org/wp-content/papercite-data/pdf/duivenvoorden-ifac17.pdf>`_,
  in Proc. of the IFAC (International Federation of Automatic Control) World Congress, 2017.

.. [4] Y. Sui, A. Gotovos, J. W. Burdick, and A. Krause,
  `Safe exploration for optimization with Gaussian processes <https://las.inf.ethz.ch/files/sui15icml-long.pdf>`_
  in Proc. of the International Conference on Machine Learning (ICML), 2015, pp. 997–1005.

Warning: Maintenance mode
-------------------------
This package is no longer actively maintained. That bein said, pull requests to add functionality or fix bugs are always welcome.

Installation
------------
The easiest way to install the necessary python libraries is by installing pip (e.g. ``apt-get install python-pip`` on Ubuntu) and running

``pip install safeopt``

Alternatively you can clone the repository and install it using

``python setup.py install``

Usage
-----

*The easiest way to get familiar with the library is to run the interactive example ipython notebooks!*

Make sure that the ``ipywidgets`` module is installed. All functions and classes are documented on `Read The Docs <http://safeopt.readthedocs.org/en/latest/>`_.


License
-------

The code is licenced under the MIT license and free to use by anyone without any restrictions.
