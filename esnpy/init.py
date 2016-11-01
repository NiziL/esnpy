# -*- coding: utf-8 -*-
import numpy as np


class UniformDenseInit():
    def __init__(self, bmin, bmax):
        self._min = bmin
        self._max = bmax

    def init(self, srow, scol):
        return np.random.rand(srow, scol)*(self._max-self._min) + self._min


