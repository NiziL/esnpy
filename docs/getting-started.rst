Getting started
+++++++++++++++

Installation
------------

You can install `esnpy` using `pip`.

Either from PyPI:

    pip install esnpy

Or from GitHub:

    pip install git+https://github.com/NiziL/esnpy#egg=esnpy

Quickstart
----------

Here is a minimal example 

.. code-block:: python
    :linenos:
    :emphasize-lines: 3,5

    import esnpy

    reservoir_builder = createBuilder()
    trainer = createTrainer()
    warmup, data, target = loadData()

    # create the echo state network
    esn = esnpy.ESN(reservoir_builder.build(), trainer)
    # train it
    esn.fit(warmup, data, target)
    # test it
    predictions = esnpy.transform(data)
    print(f"error: {compute_err(target, predictions)}")

Package overview
----------------

**ESN and DeepESN**

You can create your ESN with `esnpy.ESN`. 
The constructor needs a `esnpy.Reservoir` and an implementation of `esnpy.train.Trainer`. 

`esnpy.DeepESN` doesn't differ a lot, it just expect a list of `Reservoir` and have an optional parameters `mask` to specify from which reservoirs the `Trainer` should learn. The size of `mask` and `reservoirs` must be the same. 

Then, simply call `fit` function by passing some warm up and training data with the related targets.  
Once trained, run predictions using `transform`.

**Reservoir and ReservoirBuilder**

A `Reservoir` can easily be initialized using the `ReservoirBuilder` dataclass.  
For convenience, the configuration class is also a builder, exposing a `build()` method.
This method has an optional `seed` parameter used to make deterministic initialization, and so to ease the comparaison of two identical reservoirs.

**Initializer and Tuner**

`esnpy.init.Initializer` and `esnpy.tune.Tuner` are the abstract base classes used to setup the input and internal weights of a reservoir.

`Initializer` is defined by a `init() -> Matrix` function. 
`esnpy` provides implementations of initializer for both uniform and gaussian distribution of weights, and for both dense and sparse matrix.

`Tuner` is defined by a `init(matrix : Matrix) -> Matrix` function, which can be used to modify the weights after initialization.
For example, `esnpy` provides a `SpectralRadiusTuner` to change the spectral radius of a weights matrix.

**Trainer**

`esnpy.train.Trainer` is responsible to create the output weights matrix from the training data and targets.  
It is defined by a `train(inputs: Matrix, data: Matrix, target: Matrix) -> Matrix` function.

`esnpy` provides a `RidgeTrainer` to compute the output weights using a ridge regression. 
This trainer has three parameters : one float, the regularization parameter's weight `alpha`, and two optionals boolean (default to true) `use_bias` and `use_input` to control if we should use a bias and the input to compute the readout weights.