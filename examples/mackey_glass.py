#! /usr/bin/python
# -*- coding: utf-8 -*-
import esnpy
from esnpy import init, tuner, batch_trainer
import numpy as np
from sklearn import linear_model

def main():
    print("Loading data...")
    data = np.loadtxt('data/MackeyGlass_t17.txt')

    print("Creating a reservoir...")
    builder = esnpy.ReservoirBuilder()
    builder.set_leaky(0.3)\
           .set_activation_function(np.tanh)\
           .set_input_size(1)\
           .set_input_init(init.UniformDenseInit(-0.5, 0.5))\
           .set_intern_size(1000)\
           .set_intern_init(init.UniformSparseInit(-0.5, 0.5, 0.01))\
           .add_tuner(tuner.SpectralRadiusSetter(1.25))\

    reservoir = builder.build()
    esn = esnpy.ESN(reservoir)

    print("Training...")
    training = esnpy.BatchTraining()
    training.train(esn, batch_trainer.Ridge(1e-8),
                   transient_data = data[:100, None],
                   learning_data = data[100:2000, None],
                   target = data[101:2001, None])

    print("Testing...")
    err = np.zeros((1, 500))
    u = data[2000, None]
    for i in range(500):
        y = esn.predict(u)
        err = data[2000+i, None]-y
        u = y

    print('MSE: '+str(np.sum(np.square(err))/500))
        

if __name__ == '__main__':
    main()
