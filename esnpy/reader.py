# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from .type import MatrixType


class Reader(ABC):
    @abstractmethod
    def __call__(self, data: MatrixType) -> MatrixType:
        pass


class MatrixReader(Reader):
    def __init__(self, Wout: MatrixType):
        super().__init__()
        self._Wout = Wout

    def __call__(self, data: MatrixType) -> MatrixType:
        return data @ self._Wout


class SklearnReader(Reader):
    def __init__(self, model):
        super().__init__()
        self._model = model

    def __call__(self, data: MatrixType) -> MatrixType:
        return self._model.predict(data)
