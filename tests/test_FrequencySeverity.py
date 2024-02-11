import numpy as np
import pytest
from rippy import FreqSevSims

def test_aggregate():
    sim_index = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    n_sims = 3
    fs = FreqSevSims(sim_index, values, n_sims)
    assert np.array_equal(fs.aggregate(), np.array([3., 12., 30.]))

def test_occurrence():
    sim_index = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    n_sims = 3
    fs = FreqSevSims(sim_index, values, n_sims)
    assert np.array_equal(fs.occurrence(), np.array([2., 5., 9.]))

def test_copy():
    sim_index = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    n_sims = 3
    fs = FreqSevSims(sim_index, values, n_sims)
    fs_copy = fs.copy()
    assert np.array_equal(fs_copy.sim_index, fs.sim_index)
    assert np.array_equal(fs_copy.values, fs.values)
    assert fs_copy.n_sims == fs.n_sims

def test_apply():
    sim_index = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    n_sims = 3
    fs = FreqSevSims(sim_index, values, n_sims)
    fs_squared = fs.apply(lambda x: x**2)
    assert np.array_equal(fs_squared.values, np.array([1, 4, 9, 16, 25, 36, 49, 64, 81]))

def test_math_operations():
    sim_index = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    n_sims = 3
    fs1 = FreqSevSims(sim_index, values, n_sims)
    fs2 = FreqSevSims(sim_index, values, n_sims)
    fs_sum = fs1 + fs2
    assert np.array_equal(fs_sum.values, np.array([2, 4, 6, 8, 10, 12, 14, 16, 18]))
    fs_plus_one = fs1 + 1
    assert np.array_equal(fs_plus_one.values, np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]))
    fs_times_five = fs1 * 5
    assert np.array_equal(fs_times_five.values, np.array([5, 10, 15, 20, 25, 30, 35, 40, 45]))

import numpy as np
import pytest
from rippy import FreqSevSims, FrequencySeverityModel

def test_generate():
    freq_dist = MockDistribution([2, 3, 1])  # Replace MockDistribution with your actual frequency distribution
    sev_dist = MockDistribution([1, 2, 3, 4, 5])  # Replace MockDistribution with your actual severity distribution
    model = FrequencySeverityModel(freq_dist, sev_dist)
    n_sims = 5
    rng = np.random.default_rng(42)  # Replace with your desired random number generator
    sims = model.generate(n_sims, rng)
    assert len(sims.sim_index) == n_sims
    assert len(sims.values) == sum(freq_dist.generate(n_sims, rng))
    assert sims.n_sims == n_sims

class MockDistribution:
    def __init__(self, values):
        self.values = values

    def generate(self, size, rng):
        return np.random.choice(self.values, size=size)