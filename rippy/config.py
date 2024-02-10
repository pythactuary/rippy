import os

use_gpu = os.environ.get("RIPPY_USE_GPU")=="1"

if use_gpu:
    import cupy as xp
    print("Using GPU")
else:
    import numpy as xp
    xp.seterr(divide='ignore')



class config:
    n_sims = 10000
    seed = 123456789
    rng = xp.random.default_rng(seed)


def set_default_n_sims(n):
    config.n_sims = n


def set_random_seed(seed):
    config.rng.bit_generator.state = type(config.rng.bit_generator)(seed).state
