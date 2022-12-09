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
    test = data[LEARN_LEN : LEARN_LEN + TEST_LEN + 1]
    return warmup, train, target, test


def run(
    cfg: list[esnpy.ReservoirConfig],
    trainer: esnpy.train.Trainer,
    mask: list[bool] = None,
):
    warmup_data, input_data, target_data, test_data = load_data()

    if len(cfg) == 1:
        esn = esnpy.ESN(cfg[0], trainer)
    else:
        esn = esnpy.DeepESN(cfg, trainer)

    start_time = time.perf_counter_ns()
    esn.fit(warmup_data, input_data, target_data)
    end_time = time.perf_counter_ns()

    predictions = np.zeros((TEST_LEN, 1))
    input_data = test_data[0][None]  # ensure shape is (1, 1) and not (1,)
    for i in range(TEST_LEN):
        pred = esn.transform(input_data)
        predictions[i, :] = pred
        # generative mode
        input_data = pred
        # predictive mode
        # input_data = test_data[i + 1]

    err = test_data[1 : ERROR_LEN + 1] - predictions[:ERROR_LEN]
    print(f"MSE: {np.mean(np.square(err))}")
    print(f"Trained in {(end_time-start_time)/1e6:.3f}ms")


if __name__ == "__main__":
    print("Learn with a dense internal matrix")
    run(
        [
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
            )
        ],
        esnpy.train.RidgeTrainer(1e-8),
    )

    print("Learn with a sparse internal matrix")
    run(
        [
            esnpy.ReservoirConfig(
                input_size=1,
                size=1000,
                leaky=0.3,
                fn=np.tanh,
                input_bias=True,
                input_init=esnpy.init.UniformDenseInit(-0.5, 0.5),
                input_tuners=[],
                intern_init=esnpy.init.UniformSparseInit(
                    -0.5, 0.5, density=0.01
                ),
                intern_tuners=[esnpy.tune.SpectralRadiusSetter(1.25)],
            )
        ],
        esnpy.train.RidgeTrainer(1e-8),
    )

    print("Use the input in the ridge regression")
    run(
        [
            None,
            esnpy.ReservoirConfig(
                input_size=1,
                size=1000,
                leaky=0.3,
                fn=np.tanh,
                input_bias=True,
                input_init=esnpy.init.UniformDenseInit(-0.5, 0.5),
                input_tuners=[],
                intern_init=esnpy.init.UniformSparseInit(
                    -0.5, 0.5, density=0.01
                ),
                intern_tuners=[esnpy.tune.SpectralRadiusSetter(1.25)],
            ),
        ],
        esnpy.train.RidgeTrainer(1e-8),
    )

    print("DeepESN with masking")
    run(
        [
            None,
            esnpy.ReservoirConfig(
                input_size=1,
                size=1000,
                leaky=0.3,
                fn=np.tanh,
                input_bias=True,
                input_init=esnpy.init.UniformDenseInit(-0.5, 0.5),
                input_tuners=[],
                intern_init=esnpy.init.UniformSparseInit(
                    -0.5, 0.5, density=0.01
                ),
                intern_tuners=[esnpy.tune.SpectralRadiusSetter(1.25)],
            ),
            esnpy.ReservoirConfig(
                input_size=1000,
                size=100,
                leaky=0.3,
                fn=np.tanh,
                input_bias=True,
                input_init=esnpy.init.UniformDenseInit(-0.5, 0.5),
                input_tuners=[],
                intern_init=esnpy.init.UniformSparseInit(
                    -0.5, 0.5, density=0.01
                ),
                intern_tuners=[esnpy.tune.SpectralRadiusSetter(1.25)],
            ),
        ],
        esnpy.train.RidgeTrainer(1e-8),
        mask=[True, False, True],
    )
