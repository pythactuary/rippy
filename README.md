# rippy
Reinsurance Modeling and Pricing in Python!

## Getting Started

Install rippy from pypi:

```pip install rippy```

## Introduction

Rippy is a simple, fast and lightweight simulation-based reinsurance modeling package. It has two main components:

1. Frequency-Severity, otherwise known as compound distribution simulation

    Rippy contains the ```FrequencySeverityModel``` class to set up and simulate from compound distributions with a variety of frequency and severity distributions

    ```python
    from rippy import FrequencySeverityModel, Distributions
    claims = FrequencySeverityModel(Distributions.Poisson(5),Distributions.Pareto(0.5,100000)).generate()
    ```

2. Reinsurance contract calculation
   
   Rippy can then calculate the recoveries of a number of types of reinsurance contract for the simulated claims.

   ```python
   from rippy import XoL
   xol_layer = XoL(limit=100000,excess=100000)
   xol_layer.apply(claims)
   xol_layer.print_summary()
    ```


Rippy is based on the scientific python stack of numpy and scipy. It is designed for interoperability with numpy and ndarrays, so for example the simulated claims can be operated on with
numpy ufuncs:

```python
import numpy as np
capped_claims = np.minimum(claims,2000000)
```

Under the hood/bonnet, rippy represents the simulations of the Frequency-Severity model in sparse storage, keeping two lists of simulation indices and values: 

|sim_index | values|
|-----|-----|
| 0 | 13231.12
| 0 | 432.7 |
| 2 | 78935.12 |
| 3 | 3213.9 |
| 3 | 43843.1 |
| ...| ...|

Frequency-Severity simulations can be aggregated (summed within a sim_index), which results in a standard ```np.ndarray```

```python
aggregate_claims = claims.aggregate()

```

```python
np.array([13663.82,0,78935,12,47957.0,....])
```

### Configuring the simulation settings

The global number of simulations can be changed from the ```config``` class (the default is 100,000 simulations)

```python
from rippy import config
config.n_sims = 1000000
```

The global random seed can also be configured from the ```config``` class

```python
config.set_random_seed(123456)
```

Rippy uses the ```default_rng``` class of the ```numpy.random``` module. This can also be configured using the ```config.rng``` property.



## Project Status

Rippy is currently a proof of concept. There are a limited number of supported distributions and reinsurance contracts. We are working on:

* Adding more distributions and loss generation types
* Adding support for Catastrophe loss generation and reinsurance contracts
* Adding support for more reinsurance contract types (Surplus, Stop Loss etc)
* Grouping reinsurance contracts into programs and structures
* Stratified sampling and Quasi-Monte Carlo methods
* GPU support
* Reporting dashboards

## Issues

Please log issues in github

## Contributing

You are welcome to contribute pull requests

