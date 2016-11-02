# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg


#TODO handle feedback, save and load from file
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

    def save(self):
        pass

    def load(self, path):
        pass

    def update(self, input_vector):
        self._u = input_vector
        u = np.hstack((1, input_vector))
        self._x = (1-self._a) * self._x + self._a * \
                  self._f(u.dot(self._Win) + self._W.dot(self._x))

    def warm_up(self, data):
        for u in data:
            self.update(u)


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

    def __init__(self, reservoir, readout=None):
        self._reservoir = reservoir
        self._readout = readout

    def predict(self, input):
        self._reservoir.update(input)
        return self._readout.compute(self._reservoir)

    def warm_up(self, data):
        for i in range(data.shape[0]):
            self._reservoir.update(data[i, :])

    def batch_learning(self, trainer, transient_data, learning_data, target, _from=['bias', 'input', 'reservoir']):
        self._reservoir.warm_up(transient_data)
        training = BatchTraining(self._reservoir)
        training.memorize(learning_data)
        self._readout = training.create_readout(trainer, target)


class ReadoutLayer():

    def __init__(self, Wout, _from = ['bias', 'input', 'reservoir']):
        self._Wout = Wout
        self._from = _from

    def compute(self, reservoir):
        return np.dot(self._Wout, np.hstack(map(reservoir.get, self._from)))


class BatchTraining():

    def __init__(self, reservoir):
        self._reservoir = reservoir
        self._mem = None

    def memorize(self, data):
        size = self._reservoir.in_size + self._reservoir.nn_size +1
        self._mem = np.zeros((data.shape[0], size))
        for i in range(data.shape[0]):
            u = data[i, :]
            self._reservoir.update(u)
            self._mem[i, :] = np.hstack((1, u, self._reservoir.state))

    def create_readout(self, trainer, target):
        size = int('bias' in trainer.input_list)
        if 'input' in trainer.input_list:
            size += self._reservoir.in_size
        if 'reservoir' in trainer.input_list:
            size += self._reservoir.nn_size

        mem = np.ones((target.shape[0], size))
        at = 0
        for i in trainer.input_list:
            if i == 'bias':
                size = 1
                part = mem[:, :1]
            elif i == 'input':
                size = self._reservoir.in_size
                part = mem[:, 1:1+size]
            elif i == 'reservoir':
                size = self._reservoir.nn_size
                part = mem[:, 1+self._reservoir.in_size:]
            mem[:, at:at+size] = part
            at += size

        return ReadoutLayer(trainer.compute_output_w(mem, target), trainer.input_list)
