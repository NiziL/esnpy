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
    clazz = [(class1, (0,)),
             (class2, (1,)), 
             (class3, (2,))]

    data = np.zeros((nb_by_class*traj_size, 2))
    target = np.zeros((nb_by_class*traj_size, 1))
    for i in range(nb_by_class):
        np.random.shuffle(clazz)
        d = clazz[0][0](np.linspace(0, 2*np.pi, traj_size), np.random.rand()*2*np.pi)
        data[i*traj_size:i*traj_size+traj_size, :] = np.vstack(d).T
        target[i*traj_size:i*traj_size+traj_size, :] = np.repeat(np.vstack(clazz[0][1]), traj_size, axis=1).T

    return data, target


def main():
    traj_size = 30
    nb_by_class = 50

    print("Loading data...")
    learn_data, learn_target = prepare_data(nb_by_class, traj_size)

    print("Creating a reservoir...")
    builder = esnpy.ReservoirBuilder()
    builder.set_leaky(0.8)\
           .set_input_size(2)\
           .set_input_init(esnpy.init.UniformDenseInit(-2, 2))\
           .set_intern_size(1000)\
           .set_intern_init(esnpy.init.NormalSparseInit(0, 1, 0.01))\
           .add_tuner(esnpy.tuner.SpectralRadiusSetter(1.3))

    reservoir = builder.build()

    models = [linear_model.LogisticRegression(),
              linear_model.SGDClassifier(loss='log'),
              linear_model.SGDClassifier(loss='hinge'),
              linear_model.SGDClassifier(loss='modified_huber'),
              linear_model.SGDClassifier(loss='perceptron')]
    trainers = list(map(lambda x: batch_trainer.SklearnReg(x), models))
    trainer_names = ["Logistic regression",
                     "SGD log",
                     "SGD hinge",
                     "SGD huber",
                     "SGD perceptron"]

    print("Starting the batch training...")
    training = esnpy.train.BatchTraining()
    batch = training.batch(reservoir)\
                    .transient(learn_data[:traj_size :])\
                    .memorize(learn_data[traj_size:, :])
    for name, trainer in zip(trainer_names, trainers):
        print("Training \"{}\"...".format(name))
        batch.train_readout(trainer, learn_target[traj_size:, 0])

    readout_layers = batch.get_readout()

    print("Testing...")
    nb_by_class = 150
    data, target = prepare_data(nb_by_class, traj_size)
    predictions = np.zeros((nb_by_class*traj_size, len(readout_layers)))
    for i in range(traj_size*nb_by_class):
        reservoir.update(data[i, :])
        for j, readout in enumerate(readout_layers):
            predictions[i, j] = readout.compute(reservoir)

    errors = np.zeros((nb_by_class*traj_size, len(readout_layers)))
    for j, prediction in enumerate(predictions.T):
        errors[:, j] = prediction == target[:, 0]

    step_acc = np.zeros((traj_size, len(readout_layers)))
    for k in range(traj_size):
        step_acc[k, :] = np.mean(errors[k::traj_size, :], axis=0)

    for i, acc in enumerate(step_acc.T):
        print("{} readout's accuracy: {}\n{}".format(trainer_names[i], np.mean(acc), acc))


if __name__ == '__main__':
    main()
