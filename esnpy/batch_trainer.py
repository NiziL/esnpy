# -*- coding: utf-8 -*-
import numpy as np
from .core import BasicReadout, SklearnReadout
from scipy import linalg


class BatchTrainer():
    
    def __init__(self, _from):
        self._from = _from

    @property
    def input_list(self):
        return self._from

    def compute_readout(self, mem, target):
        raise NotImplementedError()


class BasicBatchTrainer(BatchTrainer):

    def __init__(self, _from = ['bias', 'input', 'reservoir']):
        super(BasicBatchTrainer, self).__init__(_from)

    def compute_readout(self, mem, target):
        return BasicReadout(self._compute_wout(mem, target), self._from)

    def _compute_wout(self, mem, target):
        raise NotImplementedError()


class Ridge(BasicBatchTrainer):

    def __init__(self, regul_coef, _from = ['bias', 'input', 'reservoir']):
        super(Ridge, self).__init__(_from)
        self._b = regul_coef

    def _compute_wout(self, mem, target):
        return np.dot(np.dot(target.T, mem),
                      linalg.inv(np.dot(mem.T, mem) +
                                 self._b*np.eye(mem.shape[1])))


class PseudoInv(BasicBatchTrainer):

    def __init__(self, _from = ['bias', 'input', 'reservoir']):
        super(PseudoInv, self).__init__(_from)

    def _compute_wout(self, mem, target):
        return np.dot(target.T, linalg.pinv(mem))


class SklearnReg(BatchTrainer):

    def __init__(self, regression_model, _from = ['bias', 'input', 'reservoir']):
        super(SklearnReg, self).__init__(_from)
        self._reg = regression_model

    def compute_readout(self, mem, target):
        self._reg.fit(mem, target)
        return SklearnReadout(self._reg, self._from)
