# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from .type import MatrixType
import numpy as np
from scipy import linalg


class Trainer(ABC):
    @abstractmethod
    def train(
        self, data: MatrixType, states: MatrixType, target: MatrixType
    ) -> MatrixType:
        pass

    @property
    @abstractmethod
    def use_bias(self):
        pass

    @property
    @abstractmethod
    def use_input(self):
        pass


class RidgeTrainer(Trainer):
    def __init__(self, alpha: float, use_bias: bool = True, use_input=True):
        super().__init__()
        self._alpha = alpha
        self._bias = use_bias
        self._input = use_input

    @property
    def use_bias(self):
        return self._bias

    @property
    def use_input(self):
        return self._input

    def train(
        self, inputs: MatrixType, states: MatrixType, target: MatrixType
    ) -> MatrixType:
        data = []
        if self._bias:
            data.append(np.ones((states.shape[0], 1)))
        if self._input:
            data.append(inputs)
        data.append(states)
        data = np.hstack(data)
        return linalg.solve(
            np.dot(data.T, data) + self._alpha * np.eye(data.shape[1]),
            np.dot(data.T, target),
        )
