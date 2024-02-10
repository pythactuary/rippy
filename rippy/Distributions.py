"""The Distributions module contains a number of classes for simulating from statistical distributions"""
from .config import config, xp as np, use_gpu
if use_gpu:
    import cupyx.scipy.special as special
else:
    import scipy.special as special
from abc import ABC, abstractmethod



class Distribution(ABC):
    """An abstract base class for statistical distributions"""
    def __init__(self):
        pass

    def cdf(self, x):
        pass

    def invcdf(self, u):
        pass

    def generate(
        self, n_sims=None, rng: np.random.Generator = config.rng
    ) -> np.ndarray:
        if n_sims is None:
            n_sims = config.n_sims

        return self.invcdf(rng.uniform(size=n_sims))

class DiscreteDistribution(Distribution):
    """An abstract base class for discrete distributions"""
    def __init__(self):
        pass

    def cdf(self, x):
        pass

    def invcdf(self, u):
        pass

class Poisson(DiscreteDistribution):
    """Poisson Distribution"""

    def __init__(self, mean):
        self.mean = mean

    def cdf(self, x):
        """Calculates the cumulative distribution function of the Poisson distribution"""
        return special.pdtr(x, self.mean)

    def invcdf(self, u) -> np.ndarray | float:
        """Calculates the inverse cumulative distribution function of the Poisson distribution"""
        return special.pdtri(u, self.mean)

    def generate(self, n_sims=None, rng: np.random.Generator = config.rng):
        """Generates random samples from the Poisson distribution"""
        if n_sims is None:
            n_sims = config.n_sims
        return rng.poisson(self.mean, n_sims)

class NegBinomial(DiscreteDistribution):
    """NegBinomial Distribution"""

    def __init__(self, n, p):
        self.n = n
        self.p = p

    def cdf(self, x):
        """Calculates the cumulative distribution function of the Negative Binomial distribution"""
        return special.nbdtr(x, self.n, self.p)

    def invcdf(self, u) -> np.ndarray | float:
        """Calculates the inverse cumulative distribution function of the Negative Binomial  distribution"""
        return special.nbdtri(u, self.n, self.p)

    def generate(self, n_sims=None, rng: np.random.Generator = config.rng):
        """Generates random samples from the Negative Binomial  distribution"""
        if n_sims is None:
            n_sims = config.n_sims
        return rng.negative_binomial(self.n, self.p, n_sims)

class GPD(Distribution):
    r"""The Generalised Pareto distribution is defined as through the cumulative distribution function:

        .. math::

            F(x) = 1 - (1+\frac{\xi(x-\mu)}{\sigma})^{-1/\xi}, \xi!=0 \\
            F(x) = 1 - e^{-(x-\mu)/\sigma}, \xi=0 \\
            
        where :math:`\xi` is the shape parameter, :math:`\sigma` is the scale parameter and :math:`\mu` is the location (or threshold) parameter. 

    """

    def __init__(self, shape, scale, loc):
        """Initializes a new instance of the Generalised Pareto distribution with the specified scale, shape and location"""
        self.shape = shape
        self.scale = scale
        self.loc = loc

    def cdf(self, x):
        r"""Calculates the cdf of the Generalized Pareto distribution.

        The cdf Generalised Pareto distribution is defined as

        .. math::

            F(x) = 1 - (1+\frac{\xi(x-\mu)}{\sigma})^{-1/\xi}, \xi!=0 \\
            F(x) = 1 - e^{-(x-\mu)/\sigma}, \xi=0 \\

        """
        return (
            1 - (1 + self.shape * (x - self.loc) / self.scale) ** (-1 / self.shape)
            if self.shape != 0
            else 1 - np.exp(-(x - self.loc) / self.scale)
        )

    def invcdf(self, u) -> np.ndarray | float:
        """Calculates the inverse cdf of the Generalized Pareto distribution"""
        xi = self.shape
        sigma = self.scale
        mu = self.loc
        return ((1 - u) ** (-xi) - 1) * sigma / xi + mu

class Burr(Distribution):
    """Burr Distribution"""

    def __init__(self, power, shape, scale, loc):
        self.power = power
        self.shape = shape
        self.scale = scale
        self.loc = loc

    def cdf(self, x):
        """Calculates the cdf of the Burr distribution"""
        return 1 - (1 + ((x - self.loc) / self.scale) ** self.power) ** (
            -self.shape
        )

    def invcdf(self, u) -> np.ndarray | float:
        """Calculates the inverse cdf of the Burr distribution"""
        return (
            self.scale
            * (((1 / (1 - u)) ** (1 / self.shape) - 1) ** (1 / self.power))
            + self.loc
        )

class LogLogistic(Distribution):
    """Log Logistic Distribution"""

    def __init__(self, shape, scale, loc):
        self.shape = shape
        self.scale = scale
        self.loc = loc

    def cdf(self, x):
        """Calculates the cdf of the Log Logistic distribution"""
        y = ((x - self.loc) / self.scale) ** (self.shape)
        return y / (1 + y)

    def invcdf(self, u) -> np.ndarray | float:
        """Calculates the inverse cdf of the Log Logistic distribution"""
        return self.scale * ((u / (1 - u)) ** (1 / self.shape)) + self.loc

class Normal(Distribution):
    """Normal distribution"""

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def cdf(self, x):
        """Calculates the cdf of the Normal Distribution"""
        return special.ndtr( (x-self.mu)/self.sigma)

    def invcdf(self, u) -> np.ndarray | float:
        """Calculates the inverse cdf of the Normal Distribution"""
        return special.ndtr( u)*self.sigma+self.mu

class Pareto(Distribution):
    """Paretp Distribution"""

    def __init__(self, shape, scale):
        self.shape = shape
        self.scale = scale

    def cdf(self, x):
        """Calculates the cdf of the Pareto distribution"""
        return 1-(x/self.scale)**(-self.shape)

    def invcdf(self, u) -> np.ndarray | float:
        """Calculates the inverse cdf of the Pareto distribution"""
        return (1-u)**(-1/self.shape)*self.scale
