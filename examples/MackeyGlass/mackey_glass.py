#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import esnpy
import numpy as np

WARMUP_LEN = 100
LEARN_LEN = 2000
TEST_LEN = 500


def load_data():
    data = np.loadtxt("MackeyGlass_t17.txt")[:, None]
    warmup = data[:WARMUP_LEN]
    start = WARMUP_LEN
    stop = WARMUP_LEN + LEARN_LEN
    train = data[start:stop]
    start = WARMUP_LEN + 1
    stop = WARMUP_LEN + LEARN_LEN + 1
    target = data[start:stop]
    start = WARMUP_LEN + LEARN_LEN + 1
    stop = WARMUP_LEN + LEARN_LEN + 1 + TEST_LEN
    test = data[start:stop]
    return warmup, train, target, test


def run(cfg: esnpy.ReservoirConfig):

    warmup_data, input_data, target_data, test_data = load_data()

    esn = esnpy.ESN(cfg, trainer=esnpy.RidgeTrainer(1e-8))
    esn.fit(warmup_data, input_data, target_data)

    predictions = np.zeros((TEST_LEN, 1))
    input_data = test_data[0, :]
    for i in range(TEST_LEN):
        predictions[i, :] = esn.transform(input_data)
        input_data = predictions[i, :]

    print(f"MSE: {np.mean(np.square(test_data[1:, :] - predictions[:-1, :]))}")


if __name__ == "__main__":
    print("Learn with a dense internal matrix")
    run(
        esnpy.ReservoirConfig(
            input_size=1,
            size=1000,
            leaky=0.1,
            fn=np.tanh,
            input_bias=True,
            input_init=esnpy.init.UniformDenseInit(-0.5, 0.5),
            input_tuners=[],
            intern_init=esnpy.init.UniformDenseInit(-0.5, 0.5),
            intern_tuners=[esnpy.tune.SpectralRadiusSetter(1.25)],
        )
    )

    print("Learn with a sparse internal matrix")
    run(
        esnpy.ReservoirConfig(
            input_size=1,
            size=1000,
            leaky=0.1,
            fn=np.tanh,
            input_bias=True,
            input_init=esnpy.init.UniformDenseInit(-0.5, 0.5),
            input_tuners=[],
            intern_init=esnpy.init.UniformSparseInit(-0.5, 0.5, density=0.01),
            intern_tuners=[esnpy.tune.SpectralRadiusSetter(1.25)],
        )
    )
