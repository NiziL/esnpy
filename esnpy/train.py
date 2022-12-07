# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from .type import MatrixType
import numpy as np
from scipy import linalg


class Trainer(ABC):
    @abstractmethod
    def train(self, data: MatrixType, target: MatrixType) -> MatrixType:
        pass

    @property
    @abstractmethod
    def has_bias(self):
        pass


class RidgeTrainer(Trainer):
    def __init__(self, regul_coef: float, use_bias: bool = True):
        super().__init__()
        self._alpha = regul_coef
        self._bias = use_bias

    @property
    def has_bias(self):
        return self._bias

    def train(self, data: MatrixType, target: MatrixType) -> MatrixType:
        if self._bias:
            data = np.hstack((np.ones((data.shape[0], 1)), data))
        return linalg.solve(
            np.dot(data.T, data) + self._alpha * np.eye(data.shape[1]),
            np.dot(data.T, target),
        )
