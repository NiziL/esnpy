# esnpy

`esnpy` is an out-of-the-box framework to experiment around ESN and DeepESN.  
Models has been implemented using pure NumPy/SciPy, so there's no need for a powerful GPU, or some esoteric installations.  

## Installation

The project provides a working `pyproject.toml`, but it's still not hosted on pypi, hence installation from source is the only provided method right now.

```bash
git clone https://github.com/NiziL/esnpy
pip install .
```
or 
```bash
pip install git+https://github.com/NiziL/esnpy#egg=esnpy
```

## Getting started



## Bibliography

Based on:
- *The "echo state" approach to analysing and training recurrent neural networks* by Herbert Jaeger ([pdf](https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf)),
- *A pratical guide to applying Echo State Networks* by Mantas Lukoševičius ([pdf](https://www.ai.rug.nl/minds/uploads/PracticalESN.pdf)),
- *Design of deep echo state networks* by Claudio Gallicchio and al ([link](https://www.sciencedirect.com/science/article/pii/S0893608018302223)),
- *Deep echo state network (DeepESN): A brief survey* by Claudio Gallicchio and Alessio Micheli ([pdf](https://arxiv.org/pdf/1712.04323.pdf)).