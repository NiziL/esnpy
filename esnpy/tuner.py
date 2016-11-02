# -*- coding: utf-8 -*-
import numpy as np
from scipy.sparse import linalg


class ReservoirTuner():
    def tune(self, reservoir):
        raise NotImplementedError()


class SpectralRadiusSetter(ReservoirTuner):
    def __init__(self, rho_objectif, k=6):
        self._rho = rho_objectif
        self._k = k

    def tune(self, reservoir):
        #rho_max = np.max(np.abs(linalg.eig(reservoir._W)[0]))
        rho_max = np.max(np.abs(linalg.eigs(reservoir._W, k=self._k)[0]))
        reservoir._W *= self._rho / rho_max


class RemoveInputBias(ReservoirTuner):
    def tune(self, reservoir):
        reservoir._Win[0, :] = np.zeros(reservoir._W[0, :].shape)


#TODO: implements "improving reservoir with intrisic plasticity", Schrauwen
class IntrisicPlasiticty(ReservoirTuner):
    pass
