# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from .type import MatrixType, VectorType
import numpy as np
import scipy


def _uniform(shape, bmin, bmax):
    return np.random.rand(*shape).dot(bmax - bmin) + bmin


def _normal(shape, mu, sigma):
    return np.random.randn(*shape).dot(sigma) + mu


class Initializer(ABC):
    @abstractmethod
    def init(self, shape: tuple[int, int]) -> MatrixType:
        pass


class UniformDenseInit(Initializer):
    def __init__(self, min_value: float, max_value: float):
        super().__init__()
        self._min = min_value
        self._max = max_value

    def init(self, shape: tuple[int, int]) -> MatrixType:
        return _uniform(shape, self._min, self._max)


class NormalDenseInit(Initializer):
    def __init__(self, mean: float, std: float):
        super().__init__()
        self._mu = mean
        self._sigma = std

    def init(self, shape: tuple[int, int]) -> MatrixType:
        return _normal(shape, self._mu, self._sigma)


class SparseInitializer(Initializer):
    def __init__(self, density: float):
        self._d = density

    def init(self, shape: tuple[int, int]) -> MatrixType:
        m = scipy.sparse.rand(*shape, density=self._d)
        m.data = self._sparse_init(m.data.shape[0])
        return m

    @abstractmethod
    def _sparse_init(self, size: int) -> VectorType:
        pass


class UniformSparseInit(SparseInitializer):
    def __init__(self, min_value, max_value, density):
        super().__init__(density)
        self._min = min_value
        self._max = max_value

    def _sparse_init(self, size: int) -> VectorType:
        return _uniform((size,), self._min, self._max)


class NormalSparseInit(SparseInitializer):
    def __init__(self, mean, std, density):
        super().__init__(density)
        self._mu = mean
        self._sigma = std

    def _sparse_init(self, size: int) -> VectorType:
        return _normal((size,), self._mu, self._sigma)
