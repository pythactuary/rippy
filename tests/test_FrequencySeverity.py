import numpy as np
import pytest
from rippy import FreqSevSims


def test_aggregate():
    sim_index = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    n_sims = 3
    fs = FreqSevSims(sim_index, values, n_sims)
    assert np.array_equal(fs.aggregate(), np.array([3.0, 12.0, 30.0]))


def test_occurrence():
    sim_index = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    n_sims = 3
    fs = FreqSevSims(sim_index, values, n_sims)
    assert np.array_equal(fs.occurrence(), np.array([2.0, 5.0, 9.0]))


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
    assert np.array_equal(
        fs_squared.values, np.array([1, 4, 9, 16, 25, 36, 49, 64, 81])
    )


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
    assert np.array_equal(
        fs_times_five.values, np.array([5, 10, 15, 20, 25, 30, 35, 40, 45])
    )
    fs1_less1 = -1 + fs1
    assert np.array_equal(fs1_less1.values, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]))
    fs_divide2 = fs1 / 2
    assert np.array_equal(
        fs_divide2.values, np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5])
    )
    fs_squared = fs1**2
    assert np.array_equal(
        fs_squared.values, np.array([1, 4, 9, 16, 25, 36, 49, 64, 81])
    )
    fs1_times_fs2 = fs1 * fs2
    assert np.array_equal(
        fs1_times_fs2.values, np.array([1, 4, 9, 16, 25, 36, 49, 64, 81])
    )
    fs1_divide_fs2 = fs1 / fs2
    assert np.array_equal(fs1_divide_fs2.values, np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]))
    two_to_power_fs1 = 2**fs1
    assert np.array_equal(
        two_to_power_fs1.values, np.array([2, 4, 8, 16, 32, 64, 128, 256, 512])
    )
    # test with numpy array
    arr = np.array([1, 2, 3])
    fs1_times_arr = fs1 * arr
    assert np.array_equal(
        fs1_times_arr.values, np.array([1, 2, 6, 8, 10, 18, 21, 24, 27])
    )
    fs1_divide_arr = fs1 / arr
    assert np.array_equal(
        fs1_divide_arr.values, np.array([1, 2, 1.5, 2, 2.5, 2, 7 / 3, 8 / 3, 3])
    )
    fs1_plus_arr = fs1 + arr
    assert np.array_equal(fs1_plus_arr.values, np.array([2, 3, 5, 6, 7, 9, 10, 11, 12]))
    fs1_minus_arr = fs1 - arr
    assert np.array_equal(fs1_minus_arr.values, np.array([0, 1, 1, 2, 3, 3, 4, 5, 6]))
    # test reverse operations
    fs1_times_arr = arr * fs1
    assert np.array_equal(
        fs1_times_arr.values, np.array([1, 2, 6, 8, 10, 18, 21, 24, 27])
    )
    fs1_divide_arr = arr / fs1
    assert np.array_equal(
        fs1_divide_arr.values,
        np.array([1, 0.5, 2 / 3, 0.5, 0.4, 0.5, 3 / 7, 0.375, 1 / 3]),
    )
    fs1_plus_arr = arr + fs1
    assert np.array_equal(fs1_plus_arr.values, np.array([2, 3, 5, 6, 7, 9, 10, 11, 12]))
    fs1_minus_arr = arr - fs1
    assert np.array_equal(
        fs1_minus_arr.values, np.array([0, -1, -1, -2, -3, -3, -4, -5, -6])
    )
    fs1_to_power_arr = fs1**arr
    assert np.array_equal(
        fs1_to_power_arr.values, np.array([1, 2, 9, 16, 25, 216, 343, 512, 729])
    )
    arr_to_power_fs1 = arr**fs1
    assert np.array_equal(
        arr_to_power_fs1.values, np.array([1, 1, 8, 16, 32, 729, 2187, 6561, 19683])
    )
