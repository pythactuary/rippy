import numpy as np


class config:
    n_sims = 10000
    seed = 123456789
    rng = np.random.default_rng(seed)


def set_default_n_sims(n):
    config.n_sims = n


def set_seed(seed):
    config.rng.bit_generator.state = type(config.rng.bit_generator)(seed).state
