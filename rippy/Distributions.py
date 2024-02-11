"""The Distributions module contains a number of classes for simulating from statistical distributions. The distributions mainly follow the convention of Klugman"""
from .config import config, xp as np, _use_gpu
if _use_gpu:
    import cupyx.scipy.special as special
else:
    import scipy.special as special
from abc import ABC, abstractmethod



class Distribution(ABC):
    """An abstract base class for statistical distributions"""
    def __init__(self):
        pass
    
    @abstractmethod
    def cdf(self, x):
        pass
    
    @abstractmethod
    def invcdf(self, u):
        pass

    def generate(
            self, n_sims=None, rng: np.random.Generator = config.rng
        ) -> np.ndarray:
        """
        Generate random samples from the distribution.

        Parameters:
            n_sims (int, optional): Number of simulations to generate. If not provided, the default value from the config will be used.
            rng (np.random.Generator, optional): Random number generator. Defaults to the value from the config.

        Returns:
            np.ndarray: Array of random samples generated from the distribution.
        """
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
    """Poisson Distribution

    The Poisson distribution is a discrete probability distribution that expresses the probability of a given number of events occurring in a fixed interval of time or space.

    Parameters:
    - mean (float): The mean or average number of events in the interval.

    Methods:
    - cdf(x): Calculates the cumulative distribution function of the Poisson distribution.
    - invcdf(u): Calculates the inverse cumulative distribution function of the Poisson distribution.
    - generate(n_sims=None, rng=config.rng): Generates random samples from the Poisson distribution.

    """

    def __init__(self, mean):
        self.mean = mean

    def cdf(self, x: np.ndarray | float)-> np.ndarray | float:
        """
        Calculates the cumulative distribution function (CDF) of the Poisson distribution.

        Parameters:
        - x: The value at which to evaluate the CDF.

        Returns:
        The probability that a random variable from the Poisson distribution is less than or equal to x.
        """
        return special.pdtr(x, self.mean)
    

    def invcdf(self, u: np.ndarray | float ) -> np.ndarray | float:
        """
        Calculates the inverse cumulative distribution function of the Poisson distribution.

        Parameters:
            u (float or ndarray): The probability value(s) for which to calculate the inverse cumulative distribution.

        Returns:
            float or ndarray: The value(s) corresponding to the given probability value(s) in the Poisson distribution.
        """
        return special.pdtri(u, self.mean)
    

    def generate(self, n_sims=None, rng: np.random.Generator = config.rng):
        """Generates random samples from the Poisson distribution"""
        if n_sims is None:
            n_sims = config.n_sims
        return rng.poisson(self.mean, n_sims)

class NegBinomial(DiscreteDistribution):
    """NegBinomial Distribution

    This class represents the Negative Binomial distribution.

    Parameters:
    - n (float): The number of failures until the experiment is stopped.
    - p (float): The probability of success in each trial.

    Note that the parameter n, although having an interpretation as an integer can actually be a float

    Methods:
    - cdf(x): Calculates the cumulative distribution function of the Negative Binomial distribution.
    - invcdf(u): Calculates the inverse cumulative distribution function of the Negative Binomial distribution.
    - generate(n_sims=None, rng=config.rng): Generates random samples from the Negative Binomial distribution.
    """

    def __init__(self, n: float, p: float):
        """
        Create a new NegBinomial distribution with set parameters.

        Args:
            n (float): The number of failures until the experiment is stopped.
            p (float): The probability of success in each trial.

        Returns:
            None
        """
        self.n = n
        self.p = p

    def cdf(self, x):
        """
        Calculates the cumulative distribution function of the Negative Binomial distribution.

        Parameters:
            x (float): The value at which to evaluate the cumulative distribution function.

        Returns:
            float: The cumulative distribution function value at the given value.
        """
        return special.nbdtr(x, self.n, self.p)
    
    def invcdf(self, u) -> np.ndarray | float:
        """
        Calculates the inverse cumulative distribution function of the Negative Binomial distribution.

        Parameters:
            u (float or ndarray): The probability value(s) for which to calculate the inverse cumulative distribution.


        Returns:
            np.ndarray or float: The inverse cumulative distribution function values corresponding to the given probability values.
        """
        return special.nbdtri(u, self.n, self.p)

    def generate(self, n_sims=None, rng: np.random.Generator = config.rng):
        """
        Generates random samples from the Negative Binomial distribution.

        Parameters:
            n_sims (int, optional): Number of simulations to generate. If not provided, it uses the default value from the config.
            rng (numpy.random.Generator, optional): Random number generator. If not provided, it uses the default generator from the config.

        Returns:
            numpy.ndarray: Array of random samples from the Negative Binomial distribution.
        """
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
    r"""The Burr Distribution is defined through the cumulative distribution function:
    
    .. math::

        F(x) =    1 - \left[1 + \left(\frac{(x - \mu)}{\sigma}\right) ^k\right] ^
            {-c}, x>\mu \\

    where :math:`\mu` is the location parameter, :math:`\sigma` is the scale parameter, :math:`c` is the power parameter and :math:`k` is the shape parameter.
    
    """

    def __init__(self, power, shape, scale, loc):
        """
        Creates a new Burr distribution.

        Args:
            power (float): The power parameter.
            shape (float): The shape parameter.
            scale (float): The scale parameter.
            loc (float): The location parameter.
        """
        self.power = power
        self.shape = shape
        self.scale = scale
        self.loc = loc

    def cdf(self, x) -> np.ndarray | float:
        """
        Calculates the cumulative distribution function (CDF) of the Burr distribution.

        Parameters:
            x: The value at which to evaluate the CDF.

        Returns:
            u: The CDF value at the given x.
        """
        return 1 - (1 + ((x - self.loc) / self.scale) ** self.power) ** (
            -self.shape
        )

    def invcdf(self, u) -> np.ndarray | float:
        """
        Calculates the inverse cumulative distribution function (CDF) of the Burr distribution.

        Parameters:
            u (float or np.ndarray): The probability value(s) for which to calculate the inverse CDF.

        Returns:
            float or np.ndarray: The corresponding quantile(s) for the given probability value(s).
        """
        return (
            self.scale
            * (((1 / (1 - u)) ** (1 / self.shape) - 1) ** (1 / self.power))
            + self.loc
        )

class LogLogistic(Distribution):
    r"""The Log Logistic Distribution is defined through the cumulative distribution function:

    .. math::

        F(x) =    1 - (1 + (\frac{(x - \mu)}{\sigma}) ^ {k}) ^ (
            -1
        ), x>\mu \\

    where :math:`\mu` is the location parameter, :math:`\sigma` is the scale parameter and :math:`k` is the shape parameter.
    
    """

    def __init__(self, shape, scale, loc=0):
        """
        Create a LogLogistic distribution.

        Args:
            shape (float): The shape parameter of the distribution.
            scale (float): The scale parameter of the distribution.
            loc (float): The location parameter of the distribution.
        """
        self.shape = shape
        self.scale = scale
        self.loc = loc

    def cdf(self, x):
        """
        Calculates the cumulative distribution function (CDF) of the Log Logistic distribution.

        Parameters:
            x (float): The input value.

        Returns:
            float: The CDF value at the given input.
        """
        y = ((x - self.loc) / self.scale) ** (self.shape)
        return y / (1 + y)

    def invcdf(self, u: np.ndarray | float) -> np.ndarray | float:
            """
            Calculates the inverse cdf of the Log Logistic distribution.

            Parameters:
                u (float or np.ndarray): The probability value(s) for which to calculate the inverse cdf.

            Returns:
                float or np.ndarray: The corresponding inverse cdf value(s).
            """
            return self.scale * ((u / (1 - u)) ** (1 / self.shape)) + self.loc

class Normal(Distribution):
    """Normal distribution"""

    def __init__(self, mu, sigma):
        """
        Create a Normal distribution.

        Parameters:
        - mu (float): The mean of the distribution.
        - sigma (float): The standard deviation of the distribution.
        """
        self.mu = mu
        self.sigma = sigma

    def cdf(self, x):
        """
        Calculates the cumulative distribution function (CDF) of the Normal Distribution.

        Parameters:
        - x: The value at which to evaluate the CDF.

        Returns:
        The probability that a random variable from the Normal Distribution is less than or equal to x.
        """
        return special.ndtr((x - self.mu) / self.sigma)

    def invcdf(self, u) -> np.ndarray | float:
        """Calculates the inverse cdf of the Normal Distribution
        
        Parameters:
        u (float or ndarray): The probability value(s) for which to calculate the inverse cdf.
        
        Returns:
        float or ndarray: The corresponding value(s) from the inverse cdf of the Normal Distribution.
        """
        return special.ndtri(u) * self.sigma + self.mu

class Pareto(Distribution):
    r"""Pareto Distribution

    Represents a Pareto distribution with given shape and scale parameters.

    The Pareto distribution is a power-law probability distribution that is frequently used to model the distribution of wealth, income, and other quantities. It is defined by the following probability density function (PDF):

    ..math ::

        f(x) = \frac{a * x_m^a}{x^{a+1}}

    where :math:`a` is the shape parameter and :math:`x_m` is the scale parameter.

    The cumulative distribution function (CDF) of the Pareto distribution is given by:

    ,,math ::
        F(x) = 1 - (\frac{x_m}{x})^a

    Args:
        shape (float): The shape parameter of the Pareto distribution.
        scale (float): The scale parameter of the Pareto distribution.
    """

    def __init__(self, shape, scale):
        self.shape = shape
        self.scale = scale

    def cdf(self, x):
        """Calculates the cumulative distribution function (CDF) of the Pareto distribution.

        Args:
            x (float): The value at which to evaluate the CDF.

        Returns:
            float: The CDF value at the given x.
        """
        return 1 - (x / self.scale) ** (-self.shape)

    def invcdf(self, u) -> np.ndarray | float:
        """Calculates the inverse cumulative distribution function (inverse CDF) of the Pareto distribution.

        Args:
            u (float): The probability value at which to evaluate the inverse CDF.

        Returns:
            np.ndarray or float: The inverse CDF value(s) at the given u.
        """
        return (1 - u) ** (-1 / self.shape) * self.scale


class ParaLogistic(Distribution):
    r"""ParaLogistic Distribution

    Represents a ParaLogistic distribution with given shape and scale parameters.

    The cumulative distribution function (CDF) of the ParaLogistic distribution is given by:

    .. math::
        F(x) = 1 - \left[1+\left(\frac{x-\mu}{\sigma}\right)^\alpha\right]^{-\alpha}, x>\mu

    where the shape parameter :math:`\alpha` and the scale parameter :math:`\sigma` are both positive, and the location parameter :math:`\mu` is any real number.
        
    Args:
        shape (float): The shape parameter of the ParaLogistic distribution.
        scale (float): The scale parameter of the ParaLogistic distribution.
    """

    def __init__(self, shape, scale,loc = 0):
        self.shape = shape
        self.scale = scale
        self.loc = loc

    def cdf(self, x):
        """Calculates the cumulative distribution function (CDF) of the ParaLogistic distribution.

        Args:
            x (float): The value at which to evaluate the CDF.

        Returns:
            float: The CDF value at the given x.
        """
        y = 1/(1+((x-self.loc)/self.scale)**(self.shape))
        return 1 - (y) ** self.shape

    def invcdf(self, u: np.ndarray | float) -> np.ndarray | float:
        """Calculates the inverse cumulative distribution function (inverse CDF) of the ParaLogistic distribution.

        Args:
            u (float): The probability value at which to evaluate the inverse CDF.

        Returns:
            np.ndarray or float: The inverse CDF value(s) at the given u.
        """
        return  self.loc + self.scale * ((1 - u) ** (-1 / self.shape) - 1) ** (1 / self.shape)