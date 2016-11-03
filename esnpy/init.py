# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse


def _uniform(shape, bmin, bmax):
    return np.random.rand(*shape).dot(bmax-bmin) + bmin


def _normal(shape, mu, sigma):
    return np.random.randn(*shape).dot(sigma) + mu


class UniformDenseInit():
    def __init__(self, bmin, bmax):
        self._min = bmin
        self._max = bmax

    def init(self, srow, scol):
        return _uniform((srow, scol), self._min, self._max)


class NormalDenseInit():
    def __init__(self, mu, sigma):
        self._sigma = sigma
        self._mu = mu

    def init(self, srow, scol):
        return _normal((srow, scol), self._mu, self._sigma)


class SparseInit():
    def __init__(self, density):
        self._d = density

    def init(self, srow, scol):
        m = scipy.sparse.rand(srow, scol, density=self._d)
        m.data = self._rand_data(len(m.data))
        return m.tocsr()

    def _rand_data(self, size):
        raise NotImplementedError()


class UniformSparseInit(SparseInit):
    def __init__(self, bmin, bmax, density):
        super(UniformSparseInit, self).__init__(density)
        self._min = bmin
        self._max = bmax

    def _rand_data(self, size):
        return _uniform((size,), self._max, self._min)


class NormalSparseInit(SparseInit):
    def __init__(self, mu, sigma, density):
        super(NormalSparseInit, self).__init__(density)
        self._mu = mu
        self._sigma = sigma

    def _rand_data(self, size):
        return _normal((size,), self._mu, self._sigma)
