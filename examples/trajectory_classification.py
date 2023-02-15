#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import esnpy
import time
from sklearn import linear_model, tree


def class1(t, beta):
    return [
        np.sin(t + beta) * np.abs(np.sin(t)),
        np.cos(t + beta) * np.abs(np.sin(t)),
    ]


def class2(t, beta):
    return [
        np.sin(t + beta) * np.sin(3 * t),
        np.cos(t + beta) * np.sin(3 * t),
    ]


def class3(t, beta):
    return [
        np.sin(t + beta) * np.sin(2 * t),
        np.cos(t + beta) * np.sin(2 * t),
    ]


POINTS_PER_TRAJECTORY = 30
CLASSES = (class1, class2, class3)


def prepare_data(nb_by_class, traj_size=POINTS_PER_TRAJECTORY):
    todo = list(range(len(CLASSES))) * nb_by_class
    np.random.shuffle(todo)

    data = np.zeros((len(CLASSES) * nb_by_class * traj_size, 2))
    target = np.zeros((len(CLASSES) * nb_by_class * traj_size, 1))
    for i, class_id in enumerate(todo):
        traj = CLASSES[class_id](
            np.linspace(0, 2 * np.pi, traj_size),
            np.random.rand() * 2 * np.pi,
        )
        data[i * traj_size : (i + 1) * traj_size] = np.vstack(traj).T
        target[i * traj_size : (i + 1) * traj_size] = np.repeat(
            np.vstack((class_id,)), traj_size, axis=1
        ).T
    return data, target


def main():
    print("Creating data...", end=" ")
    warmup_data, _ = prepare_data(1)
    learn_data, learn_target = prepare_data(50)
    test_data, test_target = prepare_data(50)
    print("ok")

    print("Creating the reservoir...", end=" ")
    reservoir = esnpy.ReservoirBuilder(
        size=512,
        leaky=0.8,
        fn=np.tanh,
        input_size=2,
        input_bias=True,
        input_init=esnpy.init.UniformDenseInit(-2, 2),
        input_tuners=[],
        intern_init=esnpy.init.NormalSparseInit(0, 1, density=0.01),
        intern_tuners=[esnpy.tune.SpectralRadiusTuner(1.3)],
    ).build()
    print("ok")

    print("Training with RidgeTrainer...", end=" ")
    esn = esnpy.ESN(
        reservoir,
        esnpy.train.RidgeTrainer(1e-8),
    )
    start = time.perf_counter_ns()
    esn.fit(warmup_data, learn_data, learn_target)
    end = time.perf_counter_ns()
    print(f"done in {(end - start) / 1e6}ms")

    print("Testing...", end=" ")
    pred = esn.transform(test_data)
    pred = np.rint(pred).clip(0, 2)

    acc = (pred == test_target).mean()
    print(f"Average accuracy: {acc:0.2f}")

    print("Training with sklearn...")
    esn = esnpy.ESN(
        reservoir,
        esnpy.train.SklearnTrainer(linear_model.RidgeClassifier(alpha=1e-8)),
    )
    start = time.perf_counter_ns()
    esn.fit(warmup_data, learn_data, learn_target)
    end = time.perf_counter_ns()
    print(f"done in {(end - start) / 1e6}ms")

    print("Testing...", end=" ")
    pred = esn.transform(test_data)
    pred = pred.reshape(-1, 1)
    acc = (pred == test_target).mean()
    print(f"Average accuracy: {acc:0.2f}")


if __name__ == "__main__":
    main()
