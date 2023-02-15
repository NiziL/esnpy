# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import numpy as np
from scipy import linalg
from .type import MatrixType
from .reader import MatrixReader, SklearnReader

__all__ = ["Trainer", "RidgeTrainer", "SklearnTrainer"]


class Trainer(ABC):
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
        return self.compute_readout(data, target)

    @abstractmethod
    def compute_readout(
        self, data: MatrixType, target: MatrixType
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

    def compute_readout(
        self, data: MatrixType, target: MatrixType
    ) -> MatrixType:
        return MatrixReader(
            linalg.solve(
                data.T @ data + self._alpha * np.eye(data.shape[1]),
                data.T @ target,
            )
        )


class SklearnTrainer(Trainer):
    def __init__(self, sklearn_model, use_bias: bool = True, use_input=True):
        super().__init__()
        self._model = sklearn_model
        self._bias = use_bias
        self._input = use_input

    @property
    def use_bias(self):
        return self._bias

    @property
    def use_input(self):
        return self._input

    def compute_readout(
        self, data: MatrixType, target: MatrixType
    ) -> MatrixType:
        self._model.fit(data, target)
        return SklearnReader(self._model)
