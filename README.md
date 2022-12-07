# esnpy

`esnpy` is an out-of-the-box framework to experiment around ESN and DeepESN.  
Models has been implemented in pure NumPy/SciPy, so there's no need for a powerful GPU, or some esoteric requirements. 

Right now, the focus is on batch training, and I haven't take into account the feedback loop.  
But feel free to open a ticket a discuss about anything you need, or features you want !

The documentation is coming soon.  

## Getting Started

### Installation

#### From Pypi

```bash
pip install esnpy
```

#### From source

```bash
pip install git+https://github.com/NiziL/esnpy#egg=esnpy
```
Use `github.com/NiziL/esnpy@<tag or branch>#egg=esnpy` to install from a specific branch or tag instead of main.

### Code Examples 

Don't want to read anything except code ? Take a look at the `examples/` folder for a quickstart !  
- `MackeyGlass/` demonstrates how to learn to predict a chaotic time series
- `TrajectoryClassification/` demonstrates how to learn to classify 2D trajectories

### Quickstart

You can create your ESN with `esnpy.ESN`. The constructor need a `esnpy.ReservoirConfig` and an implementation of `esnpy.train.Trainer`.  
Then, simply call `fit` function passing some warm up and training data with the related targets.  
Once trained, do predictions using `transform`.

```python
import esnpy

config = createConfig()
trainer = createTrainer()
warmup, data, target = loadData()

esnpy.ESN(config, trainer)
esnpy.fit(warmup, data, target)
predictions = esnpy.transform(data)

print(f"error: {compute_err(target, predictions)}")
```

#### `ReservoirConfig` parameters

| Parameters    | Type                     | Info                                         |
|---------------|--------------------------|----------------------------------------------|
| input_size    | `int`                    | Size of input vectors                        |
| size          | `int`                    | Number of units in the reservoir             |
| leaky         | `float`                  | Leaky parameter of the reservoir             |
| fn            | `Callable`               | Activation function of the reservoir         |
| input_bias    | `bool`                   | Enable the usage of a bias in the input      |
| input_init    | `esnpy.init.Initializer` | Define how to initialize the input weights   |
| input_tuners  | `list[esnpy.tune.Tuner]` | Define how to tune the input weights         |
| intern_init   | `esnpy.init.Initializer` | Define how to intialize the internal weights |
| intern_tuners | `list[esnpy.init.Tuner]` | Define how to tune the internal weights      |

#### `esnpy.init.Initializer` and `esnpy.tune.Tuner` 

`esnpy.init.Initializer` and `esnpy.tune.Tuner` are the abstract base classes used to setup the input and internal weights of a reservoir.

`Initializer` is defined by a `init() -> Matrix` function. 
`esnpy` provides implementations of initializer for both uniform and gaussian distribution of weights, and for both dense and sparse matrix.

`Tuner` is defined by a `init(matrix : Matrix) -> Matrix` function, which can be used to modify the weights after initialization.
For example, `esnpy` provides a `SpectralRadiusTuner` to change the spectral radius of a weights matrix.

#### `Trainer`

`esnpy.train.Trainer` is responsible to create the output weights matrix from the training data and targets.

It is defined by a `train(data: Matrix, target: Matrix) -> Matrix` function.
Beware, the `data` parameter here is not the input data but the reservoir states recorded during the training.

`esnpy` provides a `RidgeTrainer` to compute the output weights using a ridge regression.

## Tips & Tricks

- Sparse matrix are usually way faster than dense matrix
- If you want to also use the input vector to compute the output (as in original paper), you'll have to use a `esnpy.DeepESN` with a `None` as the first element of the reservoir config list. It will create a simple identity function as the first layer, and so allow a `Trainer` to get access to these data.
- Use `numpy.random.seed(seed)` before creating a each ESN if you want to compare two indentical reservoir.

## Features & Roadmap

- "core" features
  - [x] ESN (one reservoir)
  - [x] DeepESN (stacked reservoir)
  - [x] Initializer: random or normal distribution, dense or sparse matrix
  - [x] Tuner: spectral radius setter
  - [x] Trainer: basic ridge regression
- "nice to have" features
  - [ ] Trainer adapter for sklearn model
  - [ ] [Intrinsic plasticity](https://www.sciencedirect.com/science/article/pii/S0925231208000519) as a tuner
- "maybe later" features
  - [ ] better handling of feeback loop
  - [ ] online training (have to find papers about it)

## Bibliography

Based on:
- *The "echo state" approach to analysing and training recurrent neural networks* by Herbert Jaeger ([pdf](https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf)),
- *A pratical guide to applying Echo State Networks* by Mantas Lukoševičius ([pdf](https://www.ai.rug.nl/minds/uploads/PracticalESN.pdf)),
- *Design of deep echo state networks* by Claudio Gallicchio and al ([link](https://www.sciencedirect.com/science/article/pii/S0893608018302223)),
- *Deep echo state network (DeepESN): A brief survey* by Claudio Gallicchio and Alessio Micheli ([pdf](https://arxiv.org/pdf/1712.04323.pdf)).

Special thanks to Mantas Lukoševičius for his [minimal ESN example](https://mantas.info/wp/wp-content/uploads/simple_esn/minimalESN.py), which greatly helped me to get started with reservoir computing.
