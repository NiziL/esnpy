#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import esnpy
import numpy as np
import time
from typing import Union
from pathlib import Path

WARMUP_LEN = 100
LEARN_LEN = 2000
TEST_LEN = 2000
ERROR_LEN = 500


def load_data():
    data_path = Path(__file__).parent.absolute()
    data = np.loadtxt(data_path / "data" / "MackeyGlass_t17.txt")[:, None]
    warmup = data[:WARMUP_LEN]
    train = data[WARMUP_LEN:LEARN_LEN]
    target = data[WARMUP_LEN + 1 : LEARN_LEN + 1]
    test = data[LEARN_LEN : LEARN_LEN + TEST_LEN + 1]
    return warmup, train, target, test


def run(esn: Union[esnpy.ESN, esnpy.DeepESN]):
    warmup_data, input_data, target_data, test_data = load_data()

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
    print("ESN with a dense internal matrix")
    run(
        esnpy.ESN(
            esnpy.ReservoirBuilder(
                size=1000,
                leaky=0.3,
                input_size=1,
                input_init=esnpy.init.UniformDenseInit(-0.5, 0.5),
                intern_init=esnpy.init.UniformDenseInit(-0.5, 0.5),
                intern_tuners=[esnpy.tune.SpectralRadiusTuner(1.25)],
            ).build(),
            esnpy.train.RidgeTrainer(1e-8),
        )
    )

    print("ESN with a sparse internal matrix")
    run(
        esnpy.ESN(
            esnpy.ReservoirBuilder(
                size=1000,
                leaky=0.3,
                input_size=1,
                input_init=esnpy.init.UniformDenseInit(-0.5, 0.5),
                intern_init=esnpy.init.UniformSparseInit(-0.5, 0.5, 0.01),
                intern_tuners=[esnpy.tune.SpectralRadiusTuner(1.25)],
            ).build(),
            esnpy.train.RidgeTrainer(1e-8),
        )
    )

    print("DeepESN with masking")
    run(
        esnpy.DeepESN(
            [
                esnpy.ReservoirBuilder(
                    size=1024,
                    leaky=0.3,
                    input_size=1,
                    input_init=esnpy.init.UniformDenseInit(-0.5, 0.5),
                    intern_init=esnpy.init.UniformSparseInit(
                        -0.5, 0.5, density=0.01
                    ),
                    intern_tuners=[esnpy.tune.SpectralRadiusTuner(1.25)],
                ).build(),
                esnpy.ReservoirBuilder(
                    size=512,
                    leaky=0.3,
                    input_size=1024,
                    input_init=esnpy.init.UniformDenseInit(-0.5, 0.5),
                    intern_init=esnpy.init.UniformSparseInit(
                        -0.5, 0.5, density=0.01
                    ),
                    intern_tuners=[esnpy.tune.SpectralRadiusTuner(1.25)],
                ).build(),
                esnpy.ReservoirBuilder(
                    size=128,
                    leaky=0.3,
                    input_size=512,
                    input_init=esnpy.init.UniformDenseInit(-0.5, 0.5),
                    intern_init=esnpy.init.UniformSparseInit(
                        -0.5, 0.5, density=0.01
                    ),
                    intern_tuners=[esnpy.tune.SpectralRadiusTuner(1.25)],
                ).build(),
            ],
            trainer=esnpy.train.RidgeTrainer(1e-8),
            # only use the two last reservoirs for training
            mask=[False, True, True],
        )
    )
