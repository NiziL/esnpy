# -*- coding: utf-8 -*-
import numpy as np
from dataclasses import dataclass, field
from .type import MatrixType, VectorType
from typing import Callable
from .init import Initializer
from .tune import Tuner

__all__ = ["ReservoirBuilder"]


@dataclass
class ReservoirBuilder:
    """Dataclass helper to configure and build a reservoir."""

    size: int
    leaky: float
    input_size: int
    input_init: Initializer
    intern_init: Initializer
    input_bias: bool = field(default=True)
    input_tuners: list[Tuner] = field(default_factory=lambda: [])
    intern_tuners: list[Tuner] = field(default_factory=lambda: [])
    fn: Callable = field(default=np.tanh)

    def build(self, seed=None):
        """Build a reservoir according to the configuration."""
        if seed is not None:
            np.random.seed(seed)
        return Reservoir(self)


class Reservoir:
    def __init__(self, config: ReservoirBuilder):
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
        update = vector @ self._Win + self._W @ self._state
        self._state = self._a * self._fn(update) + (1 - self._a) * self._state
        return self._state

    def __call__(self, data: MatrixType) -> MatrixType:
        outputs = np.zeros((data.shape[0], self._W.shape[0]))
        for i, vec in enumerate(data):
            if self._input_bias:
                vec = np.hstack((1, vec))
            outputs[i] = self.__update(vec)
        return outputs
