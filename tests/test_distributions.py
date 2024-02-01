from rippy import Distributions
import pytest
import math
import scipy.special


def test_GPD():
    shape = 0.25
    scale = 100000
    threshold = 1000000
    dist = Distributions.GPD(shape, scale, threshold)
    assert dist.cdf(1000000) == 0.0
    assert dist.cdf(1500000) == pytest.approx(0.960981557689, 1e-4)
    assert dist.invcdf(0) == 1000000
    assert dist.invcdf(0.960981557689) == pytest.approx(1500000, 1e-4)

    sims = dist.generate(100000000)
    assert sims.mean() == pytest.approx(scale / (1 - shape) + threshold, 1e-3)
    assert sims.std() == pytest.approx(
        scale / (1 - shape) / math.sqrt(1 - 2 * shape), 1e-3
    )


def test_Burr():
    power = 2
    shape = 3
    scale = 100000
    loc = 1000000
    dist = Distributions.Burr(power, shape, scale, loc)
    assert dist.cdf(1000000) == 0.0
    assert dist.cdf(1500000) == pytest.approx(0.9999431042330451, 1e-8)
    assert dist.invcdf(0) == 1000000
    assert dist.invcdf(0.9999431042330451) == pytest.approx(1500000, 1e-8)

    sims = dist.generate(10000000)
    assert sims.mean() == pytest.approx(
        shape * scipy.special.beta(shape - 1 / power, 1 + 1 / power) * scale + loc, 1e-3
    )
    assert sims.std() == pytest.approx(
        math.sqrt(
            shape * scipy.special.beta(shape - 2 / power, 1 + 2 / power)
            - shape**2 * scipy.special.beta(shape - 1 / power, 1 + 1 / power) ** 2
        )
        * scale,
        1e-3,
    )


def test_LogLogistic():
    shape = 4
    scale = 100000
    loc = 1000000
    dist = Distributions.LogLogistic(shape, scale, loc)
    assert dist.cdf(1000000) == 0.0
    assert dist.cdf(1500000) == pytest.approx(0.9984025559105432, 1e-8)
    assert dist.invcdf(0) == 1000000
    assert dist.invcdf(0.5) == scale + loc
    assert dist.invcdf(0.9984025559105432) == pytest.approx(1500000, 1e-8)

    sims = dist.generate(100000000)
    assert sims.mean() == pytest.approx(
        scipy.special.beta(1 - 1 / shape, 1 + 1 / shape) * scale + loc, 1e-3
    )
    assert sims.std() == pytest.approx(
        math.sqrt(
            scipy.special.beta(1 - 2 / shape, 1 + 2 / shape)
            - scipy.special.beta(1 - 1 / shape, 1 + 1 / shape) ** 2
        )
        * scale,
        1e-3,
    )
