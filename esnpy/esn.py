# -*- coding: utf-8 -*-
import numpy as np
from abc import ABC, abstractmethod
from .train import Trainer
from .type import MatrixType
from .reservoir import ReservoirBuilder, Reservoir

__all__ = ["ESN", "DeepESN"]


class BaseESN(ABC):
    def __init__(self, trainer: Trainer):
        super().__init__()
        self._trainer = trainer
        self._reader = None

    @abstractmethod
    def _warmup(self, data: MatrixType):
        pass

    @abstractmethod
    def _forward(self, data: MatrixType) -> MatrixType:
        pass

    def fit(self, warmup: MatrixType, data: MatrixType, target: MatrixType):
        self._warmup(warmup)
        states = self._forward(data)
        self._reader = self._trainer.train(data, states, target)

    def transform(self, data: MatrixType) -> MatrixType:
        if self._reader is None:
            raise RuntimeError("Don't call transform before fit !")
        states = self._forward(data)
        inputs = []
        if self._trainer.use_bias:
            inputs.append(np.ones((states.shape[0], 1)))
        if self._trainer.use_input:
            inputs.append(data)
        inputs.append(states)
        return self._reader(np.hstack(inputs))


class ESN(BaseESN):
    """Echo State Network implementation."""

    def __init__(self, reservoir: Reservoir, trainer: Trainer):
        super().__init__(trainer)
        self._reservoir = reservoir

    def _warmup(self, data: MatrixType):
        self._reservoir(data)

    def _forward(self, data: MatrixType) -> MatrixType:
        return self._reservoir(data)


class DeepESN(BaseESN):
    """DeepESN implementation."""

    def __init__(
        self,
        reservoirs: list[ReservoirBuilder],
        trainer: Trainer,
        mask: list[bool] = None,
    ):
        super().__init__(trainer)
        self._reservoirs = reservoirs
        if mask is None:
            self._mask = [True] * len(self._reservoirs)
        else:
            self._mask = mask

    def _warmup(self, data: MatrixType):
        sizes = []
        for reservoir, masked in zip(self._reservoirs, self._mask):
            data = reservoir(data)
            if masked:
                sizes.append(data.shape[1])
            else:
                sizes.append(0)
        self._sizes = sizes

    def _forward(self, data: MatrixType) -> MatrixType:
        states = np.zeros((data.shape[0], sum(self._sizes)))
        for i, (reservoir, masked) in enumerate(
            zip(self._reservoirs, self._mask)
        ):
            data = reservoir(data)
            if masked:
                store_from = sum(self._sizes[:i])
                store_to = sum(self._sizes[: i + 1])
                states[:, store_from:store_to] = data
        return states
