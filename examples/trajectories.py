#! /usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import esnpy
from esnpy import batch_trainer
from sklearn import linear_model


def class1(t, beta):
    return [np.sin(t+beta)*np.abs(np.sin(t)), np.cos(t+beta)*np.abs(np.sin(t))]


def class2(t, beta):
    return [np.sin(t+beta)*np.sin(3*t), np.cos(t+beta)*np.sin(3*t)]


def class3(t, beta):
    return [np.sin(t+beta)*np.sin(2*t), np.cos(t+beta)*np.sin(2*t)]


def prepare_data(nb_by_class, traj_size):
    clazz = [(class1, [1,0,0]),
             (class2, [0,1,0]), 
             (class3, [0,0,1])]

    data = np.zeros((nb_by_class*traj_size, 2))
    target = np.zeros((nb_by_class*traj_size, 3))
    for i in range(nb_by_class):
        np.random.shuffle(clazz)
        d = clazz[0][0](np.linspace(0, 2*np.pi, traj_size), np.random.rand()*2*np.pi)
        data[i*traj_size:i*traj_size+traj_size, :] = np.vstack(d).T
        target[i*traj_size:i*traj_size+traj_size, :] = np.repeat(np.vstack(clazz[0][1]), traj_size, axis=1).T

    return data, target


def main():
    traj_size = 30
    nb_by_class = 50

    learn_data, learn_target = prepare_data(nb_by_class, traj_size)

    builder = esnpy.ReservoirBuilder()
    builder.set_leaky(0.8)\
           .set_input_size(2)\
           .set_input_init(esnpy.init.UniformDenseInit(-0.5, 0.5))\
           .set_intern_size(1000)\
           .set_intern_init(esnpy.init.UniformSparseInit(-0.5, 0.5, 0.01))\
           .add_tuner(esnpy.tuner.SpectralRadiusSetter(1.25))

    reservoir = builder.build()

    training = esnpy.train.BatchTraining()
    trainers = [ batch_trainer.Ridge(1e-5),
                 batch_trainer.SklearnReg(linear_model.LogisticRegression())]
    trainer_names = ["ridge", "sklog"]

    batch = training.batch(reservoir)\
                   .transient(learn_data[:traj_size :])\
                   .memorize(learn_data[traj_size:, :])
    for trainer in trainers[:-1]:
        batch.train_readout(trainer, learn_target[traj_size:, :])
    batch.train_readout(trainers[-1], np.argmax(learn_target[traj_size:, :], axis=1))

    readout = batch.get_readout()

    data, target = prepare_data(nb_by_class, traj_size)
    predictions = np.zeros((nb_by_class*traj_size, len(readout)))
    for i in range(traj_size*nb_by_class):
        reservoir.update(data[i, :])
        for j in range(len(readout)-1):
            predictions[i, j] = np.argmax(readout[j].compute(reservoir))
        predictions[i, -1] = readout[-1].compute(reservoir)

    errors = np.zeros((nb_by_class*traj_size, len(readout)))
    for i in range(traj_size*nb_by_class):
        errors[i, :] = predictions[i, :] == np.argmax(target[i, :])

    step_acc = np.zeros((traj_size, len(readout)))
    for k in range(traj_size):
        step_acc[k, :] = np.mean(errors[k::traj_size, :], axis=0)

    for i, acc in zip(range(len(readout)), step_acc.T):
        print("{} readout's accuracy: {}\n{}".format(trainer_names[i], np.mean(acc), acc))


if __name__ == '__main__':
    main()
