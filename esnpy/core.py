# -*- coding: utf-8 -*-
import numpy as np


class ESN():

    def __init__(self, reservoir, readout=None):
        self._reservoir = reservoir
        self._readout = readout

    def set_readout(self, readout):
        self._readout = readout

    def predict(self, input):
        self._reservoir.update(input)
        return self._readout.compute(self._reservoir)


class _ReadoutLayer():

    def __init__(self, Wout, _from = ['bias', 'input', 'reservoir']):
        self._Wout = Wout
        self._from = _from

    def compute(self, reservoir):
        return np.dot(self._Wout, np.hstack(map(reservoir.get, self._from)))


class Reservoir():

    def __init__(self, Win, W, leaky, activation):
        self._Win = Win
        self._W = W
        self._x = np.zeros(W.shape[0])
        self._a = leaky
        self._f = activation
        self._u = None

    def get(self, what):
        if what == 'bias':
            return 1
        elif what == 'input':
            return self._u
        elif what == 'reservoir':
            return self._x
        else:
            raise Error('Da fuck do you want ?')

    @property
    def in_size(self):
        return self._Win.shape[0]-1

    @property
    def nn_size(self):
        return self._W.shape[0]

    @property
    def state(self):
        return self._x

    #TODO handle feedback
    def update(self, input_vector):
        self._u = input_vector
        u = np.hstack((1, input_vector))
        self._x = (1-self._a) * self._x + self._a * \
                  self._f(u.dot(self._Win) + self._W.dot(self._x))

    def warm_up(self, data):
        for u in data:
            self.update(u)

    #TODO
    def save(self, path):
        pass
    def load(self, path):
        pass


class ReservoirBuilder():

    def __init__(self, in_size=0, nn_size=0, leaky=1, activation=np.tanh, Win_init=None, W_init=None, tuners=[]):
        self._K = in_size
        self._N = nn_size
        self._tuners = tuners
        self._Win_init = Win_init
        self._W_init = W_init
        self._a = leaky
        self._f = activation

    def set_input_size(self, size):
        self._K = size
        return self

    def set_intern_size(self, size):
        self._N = size
        return self

    def add_tuner(self, tuner):
        self._tuners.append(tuner)
        return self

    def set_input_init(self, init):
        self._Win_init = init
        return self

    def set_intern_init(self, init):
        self._W_init = init
        return self

    def set_leaky(self, leaky):
        self._a = leaky
        return self

    def set_activation_function(self, activation):
        self._f = activation
        return self

    def build(self):
        Win = self._Win_init.init(self._K+1, self._N) # +1 is for bias
        W = self._W_init.init(self._N, self._N)
        r = Reservoir(Win, W, self._a, self._f)
        for tuner in self._tuners:
            tuner.tune(r)
        self.__init__(self, self._K, self._N, self._a, self._f, self._tuners)
        return r
