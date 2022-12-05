# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from .type import MatrixType
import numpy as np
from scipy import linalg


class Trainer(ABC):
    @abstractmethod
    def train(self, data: MatrixType, target: MatrixType) -> MatrixType:
        pass


class RidgeTrainer(Trainer):
    def __init__(self, regul_coef):
        super().__init__()
        self._b = regul_coef

    def train(self, data: MatrixType, target: MatrixType) -> MatrixType:
        return target.T.dot(data).dot(
            linalg.inv(data.T.dot(data) + self._b * np.eye(data.shape[1]))
        )
