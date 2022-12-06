#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import esnpy
import numpy as np
import time

WARMUP_LEN = 100
LEARN_LEN = 2000
TEST_LEN = 2000
ERROR_LEN = 500


def load_data():
    data = np.loadtxt("MackeyGlass_t17.txt")[:, None]
    warmup = data[:WARMUP_LEN]
    train = data[WARMUP_LEN:LEARN_LEN]
    target = data[WARMUP_LEN + 1 : LEARN_LEN + 1]
    test = data[LEARN_LEN : LEARN_LEN + TEST_LEN]
    return warmup, train, target, test


def run(cfg: esnpy.ReservoirConfig, trainer: esnpy.train.Trainer):

    warmup_data, input_data, target_data, test_data = load_data()

    esn = esnpy.ESN(cfg, trainer=trainer)
    esn.fit(warmup_data, input_data, target_data)

    predictions = np.zeros((TEST_LEN, 1))
    input_data = test_data[0, :]
    for i in range(TEST_LEN):
        pred = esn.transform(input_data)
        predictions[i, :] = pred
        input_data = pred

    print(
        f"MSE: {np.mean(np.square(test_data[1:ERROR_LEN+1] - predictions[:ERROR_LEN]))}"
    )


if __name__ == "__main__":
    print("Learn with a dense internal matrix")
    start_time = time.perf_counter_ns()
    run(
        esnpy.ReservoirConfig(
            input_size=1,
            size=1000,
            leaky=0.3,
            fn=np.tanh,
            input_bias=True,
            input_init=esnpy.init.UniformDenseInit(-0.5, 0.5),
            input_tuners=[],
            intern_init=esnpy.init.UniformDenseInit(-0.5, 0.5),
            intern_tuners=[esnpy.tune.SpectralRadiusSetter(1.25)],
        ),
        esnpy.train.RidgeTrainer(1e-8),
    )
    stop_time = time.perf_counter_ns()
    print(f"Computed in {(stop_time-start_time)/1e6:.3f}ms")

    print("Learn with a sparse internal matrix")
    start_time = time.perf_counter_ns()
    run(
        esnpy.ReservoirConfig(
            input_size=1,
            size=1000,
            leaky=0.3,
            fn=np.tanh,
            input_bias=True,
            input_init=esnpy.init.UniformDenseInit(-0.5, 0.5),
            input_tuners=[],
            intern_init=esnpy.init.UniformSparseInit(-0.5, 0.5, density=0.01),
            intern_tuners=[esnpy.tune.SpectralRadiusSetter(1.25)],
        ),
        esnpy.train.RidgeTrainer(1e-8),
    )
    stop_time = time.perf_counter_ns()
    print(f"Computed in {(stop_time-start_time)/1e6:.3f}ms")
