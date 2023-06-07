# esnpy

`esnpy` is an out-of-the-box framework to experiment around ESN and DeepESN.  
Models have been implemented in pure NumPy/SciPy, so there is no need for a powerful GPU, or any esoteric requirements. 

Right now, the focus is on batch training, and feedback loops have not been taken into account.  
But feel free to open a ticket a discuss about anything you need, features you want, or even help !

The documentation for the latest release is available on [the github page](https://nizil.github.io/esnpy).

Note from the author: *`esnpy` is a small projet I initiated during my master intership, and have recently cleaned up. I might keep working on it for fun, but If you want/need a more robust framework, [ReservoirPy](https://github.com/reservoirpy/reservoirpy) might be the one you're searching for ;)*

## Getting Started

### Installation

**From PyPI**
```bash
pip install esnpy
```

**From source**
```bash
pip install git+https://github.com/NiziL/esnpy#egg=esnpy
```
Use `github.com/NiziL/esnpy@<tag or branch>#egg=esnpy` to install from a specific branch or tag instead of main.

### Quickstart

```python
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
```

#### `ESN` and `DeepESN`

You can create your ESN with `esnpy.ESN`. 
The constructor needs a `esnpy.Reservoir` and an implementation of `esnpy.train.Trainer`. 

`esnpy.DeepESN` doesn't differ a lot, it just expect a list of `Reservoir` and have an optional parameters `mask` to specify from which reservoirs the `Trainer` should learn. The size of `mask` and `reservoirs` must be the same. 

Then, simply call `fit` function by passing some warm up and training data with the related targets.  
Once trained, run predictions using `transform`.

#### `Reservoir` and `ReservoirBuilder`

A `Reservoir` can easily be initialized using the `ReservoirBuilder` dataclass.  
For convenience, the configuration class is also a builder, exposing a `build()` method.
This method has an optional `seed` parameter used to make deterministic initialization, and so to ease the comparaison of two identical reservoirs.

| Parameters    | Type                     | Description                                  | Default   |
|---------------|--------------------------|----------------------------------------------|-----------|
| input_size    | `int`                    | Size of input vectors                        |           |
| size          | `int`                    | Number of units in the reservoir             |           |
| leaky         | `float`                  | Leaky parameter of the reservoir             |           |
| fn            | `Callable`               | Activation function of the reservoir         | `np.tanh` |
| input_bias    | `bool`                   | Enable the usage of a bias in the input      | `True`    |
| input_init    | `esnpy.init.Initializer` | Define how to initialize the input weights   |           |
| input_tuners  | `list[esnpy.tune.Tuner]` | Define how to tune the input weights         | `[]`      |
| intern_init   | `esnpy.init.Initializer` | Define how to intialize the internal weights |           |
| intern_tuners | `list[esnpy.init.Tuner]` | Define how to tune the internal weights      | `[]`      |

#### `Initializer` and `Tuner` 

`esnpy.init.Initializer` and `esnpy.tune.Tuner` are the abstract base classes used to setup the input and internal weights of a reservoir.

`Initializer` is defined by a `init() -> Matrix` function. 
`esnpy` provides implementations of initializer for both uniform and gaussian distribution of weights, and for both dense and sparse matrix.

`Tuner` is defined by a `init(matrix : Matrix) -> Matrix` function, which can be used to modify the weights after initialization.
For example, `esnpy` provides a `SpectralRadiusTuner` to change the spectral radius of a weights matrix.

#### `Trainer` and `Reader`

`esnpy.train.Trainer` is responsible to create the output weights matrix from the training data and targets.  
It is defined by a `train(inputs: Matrix, data: Matrix, target: Matrix) -> Reader` function.

The `Reader` is then responsible for computing the final result from the reservoir activations through `__call__`.

`esnpy` provides a `RidgeTrainer` to compute the output weights using a ridge regression, and a `SklearnTrainer` (more on this one later). 
This trainer has three parameters : one float, the regularization parameter's weight `alpha`, and two optionals boolean (default to true) `use_bias` and `use_input` to control if we should use a bias and the input to compute the readout weights.

`SklearnTrainer` is an adapter wrapping scikit-learn model, relying on methods `fit` and `predict`. The feature is still experimental, chaos might ensure. 

## Code Examples 

Want to see some code in action ? Take a look at the `examples/` directory:
- `mackey_glass.py` demonstrates how to learn to predict a time series,
- `trajectory_classification/` demonstrates how to learn to classify 2D trajectories.

## Bibliography

Based on:
- *The "echo state" approach to analysing and training recurrent neural networks* by Herbert Jaeger ([pdf](https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf)),
- *A pratical guide to applying Echo State Networks* by Mantas Lukoševičius ([pdf](https://www.ai.rug.nl/minds/uploads/PracticalESN.pdf)),
- *Design of deep echo state networks* by Claudio Gallicchio and al ([link](https://www.sciencedirect.com/science/article/pii/S0893608018302223)),
- *Deep echo state network (DeepESN): A brief survey* by Claudio Gallicchio and Alessio Micheli ([pdf](https://arxiv.org/pdf/1712.04323.pdf)).

Special thanks to Mantas Lukoševičius for his [minimal ESN example](https://mantas.info/wp/wp-content/uploads/simple_esn/minimalESN.py), which greatly helped me to get started with reservoir computing.
