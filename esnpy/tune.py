# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from .type import MatrixType
import numpy as np
from scipy.sparse import issparse
from scipy.linalg import eigvals
from scipy.sparse.linalg import eigs


class Tuner(ABC):
    @abstractmethod
    def tune(self, weights: MatrixType) -> MatrixType:
        pass


class SpectralRadiusSetter(Tuner):
    def __init__(self, rho: float):
        super().__init__()
        self._rho = rho

    def tune(self, weights: MatrixType) -> MatrixType:
        if issparse(weights):
            rho_max = np.max(np.abs(eigs(weights, k=1)[0]))
        else:
            rho_max = np.max(np.abs(eigvals(weights)))
        return weights * self._rho / rho_max
