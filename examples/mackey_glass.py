#! /usr/bin/python
# -*- coding: utf-8 -*-
import esnpy
from esnpy import batch_trainer
import numpy as np
import pylab


def main():
    ## Parameters
    # reservoir
    size = 1000
    leaky = 0.1
    func = np.tanh
    input_init = esnpy.init.UniformDenseInit(-0.5, 0.5)
    intern_init = esnpy.init.UniformSparseInit(-0.5, 0.5, 0.01)
    spectral_radius = esnpy.tuner.SpectralRadiusSetter(1.25)

    # learning
    transient_len = 100
    learn_len = 2000
    ridge_regul_coef = 1e-8

    # testing
    test_len = 500

    print("Loading data...")
    data = np.loadtxt('data/MackeyGlass_t17.txt')

    print("Creating a reservoir...")
    builder = esnpy.ReservoirBuilder()
    builder.set_leaky(leaky)\
           .set_activation_function(func)\
           .set_input_size(1)\
           .set_input_init(input_init)\
           .set_intern_size(size)\
           .set_intern_init(intern_init)\
           .add_tuner(spectral_radius)

    reservoir = builder.build()
    esn = esnpy.ESN(reservoir)

    print("Training...")
    transient_data = data[:transient_len, None]
    learn_data = data[transient_len:learn_len, None]
    target_data = data[transient_len+1:learn_len+1, None]

    training = esnpy.train.BatchTraining()
    training.train(esn,
                   batch_trainer.Ridge(1e-8),
                   transient_data,
                   learn_data,
                   target_data)

    print("Testing...")
    test_data = data[learn_len:learn_len+test_len, None]
    test_target = data[learn_len+1:learn_len+1+test_len, None]
    prediction = np.zeros((test_len, 1))

    u = test_data[0, :]
    for i in range(test_len):
        prediction[i, :] = esn.predict(u)
        u = prediction[i, :]

    mse = np.mean(np.square(test_target-prediction))
    print('MSE: '+str(mse))

    pylab.plot(test_target, label='target')
    pylab.plot(prediction, label='prediction')
    pylab.legend()
    pylab.show()


if __name__ == '__main__':
    main()
