# -*- coding: utf-8 -*-
import numpy as np
from sklearn import linear_model


#TODO handle feedback
class Reservoir():

    def __init__(self, Win, W, leaky, activation):
        self._Win = Win
        self._W = W
        self._x = np.zeros(W.shape[0])
        self._a = leaky
        self._f = activation

    @property
    def state(self):
        return self._x

    @property
    def in_size(self):
        return self._Win.shape[0]-1

    @property
    def nn_size(self):
        return self._W.shape[0]

    #TODO save Win/W/a/f to file
    def save(self):
        pass

    #TODO load from file
    def load(self, path):
        pass

    def update(self, input):
        self._x = (1-self._a)*self._x + self._a*self._f(np.dot(np.hstack((1, input)), self._Win) + np.dot(self._x, self._W))


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


class ESN():

    def __init__(self, reservoir, readout):
        self._reservoir = reservoir
        self._readout = readout

    def compute(self, input):
        self._reservoir.update(input)
        return self._readout(self._reservoir, input)


class ESNTrainer():

    def train(self, reservoir, init_data, learn_data, target):
        for i in range(init_data.shape[1]):
            reservoir.update(init_data[:, i])
        
        mem = np.ones((1+reservoir.in_size+reservoir.nn_size, learn_data.shape[1]))
        for i in range(learn_data.shape[1]):
            reservoir.update(learn_data[:, i])
            mem[1:, i] = np.hstack((learn_data[:, i], reservoir.state))

        reg = linear_model.Ridge(alpha=1e-8, solver='lsqr')
        reg.fit(mem.T, target.T)
        Wout = reg.coef_

        def compute(reservoir, input):
            reservoir.update(input)
            return np.dot(Wout, np.hstack((1, input, reservoir.state)))

        return compute
