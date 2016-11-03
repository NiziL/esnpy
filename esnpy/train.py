# -*- coding: utf-8 -*-
import numpy as np


class Training():

    def train(self, esn, trainer, transient_data, learning_data, target):
        raise NotImplementedError()


class BatchTraining(Training):

    def __init__(self):
        self._session = None

    def train(self, esn, trainer, transient_data, learning_data, target):
        r = self.batch(esn._reservoir)\
                .transient(transient_data)\
                .memorize(learning_data)\
                .train_readout(trainer, target)\
                .get_readout()
        esn.set_readout(r)
        return self
    
    def batch(self, reservoir):
        return _Batch(reservoir)


class _Batch():

    def __init__(self, reservoir):
        self._reservoir = reservoir
        self._mem = None
        self._readouts = []
        
    def _get_mem_shrunk(self, what, size_only=False):
        part = None
        size = 0
        if what == 'bias':
            size = 1
            part = self._mem[:, :1]
        elif what == 'input':
            size = self._reservoir.in_size
            part = self._mem[:, 1:1+size]
        elif what == 'reservoir':
            size = self._reservoir.nn_size
            part = self._mem[:, -size:]
        if size_only:
            return size
        else:
            return part, size

    def transient(self, data):
        self._reservoir.warm_up(data)
        return self

    def memorize(self, data):
        size = self._reservoir.in_size + self._reservoir.nn_size +1
        self._mem = np.zeros((data.shape[0], size))
        for i in range(data.shape[0]):
            u = data[i, :]
            self._reservoir.update(u)
            self._mem[i, :] = np.hstack((1, u, self._reservoir.state))
        return self

    def train_readout(self, trainer, target):
        size = 0
        for i in trainer.input_list:
            size += self._get_mem_shrunk(i, size_only=True)
        mem = np.ones((target.shape[0], size))
        at = 0
        for i in trainer.input_list:
            part, size = self._get_mem_shrunk(i)
            mem[:, at:at+size] = part
            at += size
        self._readouts.append(trainer.compute_readout(mem, target))
        return self

    def get_readout(self):
        if len(self._readouts) == 1:
            return self._readouts[0]
        return self._readouts
