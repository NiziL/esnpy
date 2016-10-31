# -*- coding: utf-8 -*-
#! /usr/bin/env python
import esnpy
import esnpy.reservoir_utils as utils
import numpy as np

def main():
    print("Loading data...")
    data = np.vstack(np.loadtxt('data/MackeyGlass_t17.txt')).T

    print("Creating a reservoir...")
    builder = esnpy.ReservoirBuilder()
    builder.set_leaky(0.3)\
           .set_activation_function(np.tanh)\
           .set_input_size(1)\
           .set_input_init(utils.UniformDenseInit(-0.5, 0.5))\
           .set_intern_size(1000)\
           .set_intern_init(utils.UniformDenseInit(-0.5, 0.5))\
           .add_tuner(utils.SpectralRadiusSetter(1.25))\

    reservoir = builder.build()

    print("Training the readout...")
    trainer = esnpy.ESNTrainer()
    esn = trainer.train(reservoir, data[:, :100], data[:, 100:2000], data[:, 101:2001])

    print("Testing...")
    err = np.zeros((1, 500))
    u = data[:, 2000]
    for i in range(500):
        y = esn(reservoir, u)
        err = data[:, 2000+i]-y

    print('MSE= '+str(np.sum(np.square(err))/500))
        

if __name__ == '__main__':
    main()
