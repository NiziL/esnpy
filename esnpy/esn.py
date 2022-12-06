# -*- coding: utf-8 -*-
import numpy as np
from dataclasses import dataclass
from typing import Callable
from .init import Initializer
from .tune import Tuner
from .train import Trainer
from .type import MatrixType, VectorType


@dataclass
class ReservoirConfig:
    """ """

    size: int
    leaky: float
    fn: Callable
    input_size: int
    input_bias: bool
    input_init: Initializer
    input_tuners: list[Tuner]
    intern_init: Initializer
    intern_tuners: list[Tuner]


class _Reservoir:
    """Built from a ReservoirConfig"""

    def __init__(self, config: ReservoirConfig):
        super().__init__()
        # save parameters
        self._input_bias = config.input_bias
        self._a = config.leaky
        self._fn = config.fn
        # init internal state
        self._state = np.zeros(config.size)
        # init input weights
        self._Win = config.input_init.init(
            (config.input_size + config.input_bias, config.size)
        )
        for tuner in config.input_tuners:
            self._Win = tuner.tune(self._Win)
        # init internal weights
        self._W = config.intern_init.init((config.size, config.size))
        for tuner in config.intern_tuners:
            self._W = tuner.tune(self._W)

    def __update(self, vector: VectorType) -> MatrixType:
        update = vector.dot(self._Win) + self._W.dot(self._state)
        self._state = self._a * self._fn(update) + (1 - self._a) * self._state
        return self._state

    def __call__(self, data: MatrixType) -> MatrixType:
        outputs = np.zeros((data.shape[0], self._W.shape[0]))
        for i, vec in enumerate(data):
            if self._input_bias:
                vec = np.hstack((1, vec))
            outputs[i] = self.__update(vec)
        return outputs


class _Identity:
    """Built from a ReservoirConfig equals to None"""

    def __call__(self, data: MatrixType) -> MatrixType:
        return data


class ESN:
    """ """

    def __init__(self, config: ReservoirConfig, trainer: Trainer):
        super().__init__()
        self._reservoir = _Reservoir(config)
        self._trainer = trainer
        self._readout = None

    def fit(self, warmup: MatrixType, data: MatrixType, target: MatrixType):
        self._reservoir(warmup)
        states = self._reservoir(data)
        self._readout = self._trainer.train(states, target)

    def transform(self, data: MatrixType) -> MatrixType:
        if self._readout is None:
            raise RuntimeError("Don't call transform before fit !")
        reservoir_output = self._reservoir(data).T
        if self._trainer.has_bias:
            ones = np.ones((reservoir_output.shape[1], 1))
            reservoir_output = np.vstack((ones, reservoir_output))
        return self._readout.dot(reservoir_output)


class DeepESN:
    """ """

    def __init__(self, configs: list[ReservoirConfig], trainer: Trainer):
        super().__init__()

    def fit(self, warmup: MatrixType, data: MatrixType, target: MatrixType):
        ...

    def transform(self, data: MatrixType) -> MatrixType:
        ...
