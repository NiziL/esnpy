# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg


class BatchTrainer():
    
    def __init__(self, _from):
        self._from = _from

    @property
    def input_list(self):
        return self._from

    def compute_output_w(self, mem, target):
        raise NotImplementedError()


class Ridge(BatchTrainer):

    def __init__(self, regul_coef, _from = ['bias', 'input', 'reservoir']):
        super(Ridge, self).__init__(_from)
        self._b = regul_coef

    def compute_output_w(self, mem, target):
        return np.dot(np.dot(target.T, mem),
                      linalg.inv(np.dot(mem.T, mem) +
                                 self._b*np.eye(mem.shape[1])))


class PseudoInv(BatchTrainer):

    def __init__(self, _from):
        super(PseudoInv, self).__init__(_from)

    def compute_output_w(self, mem, target):
        return np.dot(target.T, linalg.pinv(mem))


class SklearnReg(BatchTrainer):

    def __init__(self, _from, regression_model):
        super(SklearnReg, self).__init__(_from)
        self._reg = regression_model

    def compute_output_w(self, mem, target):
        return self._reg.fit(mem, target).coef_
