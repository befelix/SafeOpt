from __future__ import division

import mock
import pytest

import numpy as np
import GPy
from numpy.testing import assert_allclose

from safeopt.gp_opt import GaussianProcessOptimization


class TestGPOptimization(object):

    @pytest.fixture
    def gps(self):

        kernel1 = GPy.kern.RBF(1, variance=2)
        kernel2 = GPy.kern.Matern32(1, variance=4)

        gp1 = GPy.models.GPRegression(np.array([[0]]), np.array([[0]]),
                                      kernel=kernel1)
        gp2 = GPy.models.GPRegression(np.array([[0]]), np.array([[0]]),
                                      kernel=kernel2)
        return gp1, gp2

    def test_init(self, gps):
        """Test the initialization and beta functions."""
        gp1, gp2 = gps

        opt = GaussianProcessOptimization(gp1,
                                          fmin=0,
                                          beta=2,
                                          num_contexts=1,
                                          threshold=0,
                                          scaling='auto')
        assert opt.beta(0) == 2

        opt = GaussianProcessOptimization(gp1,
                                          fmin=[0],
                                          beta=lambda x: 5,
                                          num_contexts=1,
                                          threshold=0,
                                          scaling='auto')

        assert opt.beta(10) == 5

    def test_multi_init(self, gps):
        """Test initialization with multiple GPs"""
        gp1, gp2 = gps

        opt = GaussianProcessOptimization([gp1, gp2],
                                          fmin=0,
                                          beta=2,
                                          num_contexts=1,
                                          threshold=0,
                                          scaling='auto')

        # Check scaling
        assert_allclose(opt.scaling, np.array([np.sqrt(2), np.sqrt(4)]))

    def test_scaling(self, gps):
        """Test the scaling argument."""
        gp1, gp2 = gps

        pytest.raises(ValueError, GaussianProcessOptimization, [gp1, gp2], 2,
                      scaling=[5])

        opt = GaussianProcessOptimization([gp1, gp2],
                                          fmin=[1, 0],
                                          beta=2,
                                          num_contexts=1,
                                          threshold=0,
                                          scaling=[1, 2])
        assert_allclose(opt.scaling, np.array([1, 2]))

    def test_data_adding(self, gps):
        """Test adding data points."""
        gp1, gp2 = gps

        # Test simple 1D case
        gp1.set_XY(np.array([[0.]]), np.array([[1.]]))
        opt = GaussianProcessOptimization(gp1, 0)
        opt.add_new_data_point(2, 3)

        x, y = opt.data
        assert_allclose(x, np.array([[0], [2]]))
        assert_allclose(y, np.array([[1], [3]]))

        # Test 2D case
        gp1.set_XY(np.array([[0.]]), np.array([[1.]]))
        gp2.set_XY(np.array([[0.]]), np.array([[11.]]))

        opt = GaussianProcessOptimization([gp1, gp2], [0, 1])
        opt.add_new_data_point(2, [2, 3])
        x, y = opt.data
        assert_allclose(x, np.array([[0], [2]]))
        assert_allclose(y, np.array([[1, 11], [2, 3]]))

        # Test adding NAN data
        opt.add_new_data_point(3, [2, np.nan])

        assert_allclose(opt.x, np.array([[0], [2], [3]]))
        assert_allclose(opt.y, np.array([[1, 11], [2, 3], [2, np.nan]]))

        for i, gp in enumerate(opt.gps):
            not_nan = ~np.isnan(opt.y[:, i])
            assert_allclose(gp.X, opt.x[not_nan, :])
            assert_allclose(gp.Y[:, 0], opt.y[not_nan, i])

        # Test removing data
        opt.remove_last_data_point()

        assert_allclose(opt.x, np.array([[0], [2]]))
        assert_allclose(opt.y, np.array([[1, 11], [2, 3]]))

        for i, gp in enumerate(opt.gps):
            not_nan = ~np.isnan(opt.y[:, i])
            assert_allclose(gp.X, opt.x[not_nan, :])
            assert_allclose(gp.Y[:, 0], opt.y[not_nan, i])

    def test_contexts(self):
        """Test contexts and adding data."""
        kernel1 = GPy.kern.RBF(2, variance=2)
        kernel2 = GPy.kern.Matern32(2, variance=4)

        gp1 = GPy.models.GPRegression(np.array([[0, 0]]), np.array([[5]]),
                                      kernel=kernel1)
        gp2 = GPy.models.GPRegression(np.array([[0, 0]]), np.array([[6]]),
                                      kernel=kernel2)

        opt = GaussianProcessOptimization([gp1, gp2],
                                          fmin=[0, 0],
                                          num_contexts=1)
        opt.add_new_data_point(1, [3, 4], context=2)

        assert_allclose(opt.x, np.array([[0, 0], [1, 2]]))
        assert_allclose(opt.y, np.array([[5, 6], [3, 4]]))

        for i, gp in enumerate(opt.gps):
            assert_allclose(gp.X, opt.x)
            assert_allclose(gp.Y[:, 0], opt.y[:, i])
