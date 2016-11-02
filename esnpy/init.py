# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse


class UniformDenseInit():
    def __init__(self, bmin, bmax):
        self._min = bmin
        self._max = bmax

    def init(self, srow, scol):
        return np.random.rand(srow, scol).dot(self._max-self._min) + self._min


class NormalDenseInit():
    def __init__(self, mu, sigma):
        self._sigma = sigma
        self._mu = mu

    def init(self, srow, scol):
        return np.random.randn(srow, scol).dot(self._sigma) + self._mu


class UniformSparseInit():
    def __init__(self, bmin, bmax, density):
        self._d = density
        self._min = bmin
        self._max = bmax

    def init(self, srow, scol):
        m = scipy.sparse.rand(srow, scol, density=self._d).dot(self._max-self._min)
        m.data = np.random.rand(len(m.data),).dot(self._max-self._min) + self._min 
        return m.tocsr()
