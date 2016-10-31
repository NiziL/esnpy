# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg


class SpectralRadiusSetter():
    def __init__(self, rho_objectif):
        self._rho = rho_objectif

    def tune(self, reservoir):
        rho_max = np.max(np.abs(linalg.eig(reservoir._W)[0]))
        reservoir._W *= self._rho / rho_max


class RemoveInputBias():
    def tune(self, reservoir):
        reservoir._Win[0, :] = np.zeros(reservoir._W[0, :].shape)


class UniformDenseInit():
    def __init__(self, bmin, bmax):
        self._min = bmin
        self._max = bmax

    def init(self, srow, scol):
        return np.random.rand(srow, scol)*(self._max-self._min) + self._min


#TODO: implements "improving reservoir with intrisic plasticity", Schrauwen
class IntrisicPlasitictyLearner():
    pass
