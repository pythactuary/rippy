from typing import Union
import numpy

from .config import config, xp as np
from . import Distributions


def _get_sims_of_events(n_events_by_sim: np.ndarray):
    """Given the number of events in each simulation, returns the simulation index for each event.

    >>> n_events_by_sim = np.array([1, 0, 3])
    >>> _get_sims_of_events(n_events_by_sim)
    array([0, 2, 2, 2])

    Parameters:
    - n_events_by_sim (np.ndarray): Array of the number of events in each simulation.

    Returns:
    - np.ndarray: Array of simulation indices for each event.



    """
    cumulative_n_events = n_events_by_sim.cumsum()
    total_events = cumulative_n_events[-1]
    event_no = np.arange(total_events)
    return cumulative_n_events.searchsorted(event_no + 1)


class FrequencySeverityModel:
    """A class for constructing and simulating from Frequency-Severity, or Compound distributions"""

    def __init__(
        self,
        freq_dist: Distributions.Distribution,
        sev_dist: Distributions.Distribution,
    ):
        self.freq_dist = freq_dist
        self.sev_dist = sev_dist

    def generate(self, n_sims=None, rng: np.random.Generator = config.rng):
        """
        Generate simulations from the Frequency-Severity model.

        Parameters:
        - n_sims (int): Number of simulations to generate. If None, uses the default value from the config.
        - rng (np.random.Generator): Random number generator. Defaults to the value from the config.

        Returns:
        - FreqSevSims: Object containing the generated simulations.
        """
        if n_sims is None:
            n_sims = config.n_sims
        n_events = self.freq_dist.generate(n_sims, rng)
        total_events = n_events.sum()
        sev = self.sev_dist.generate(int(total_events), rng)
        return FreqSevSims(_get_sims_of_events(n_events), sev, n_sims)


class FreqSevSims:
    """A class for storing and manipulating Frequency-Severity simulations.
    FreqSevSims objects provide convenience methods for aggregating and summarizing the simulations.

    >>> sim_index = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
    >>> values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> n_sims = 3
    >>> fs = FreqSevSims(sim_index, values, n_sims)
    >>> fs.aggregate()
    array([ 3., 12., 30.])
    >>> fs.occurrence()
    array([2., 5., 9.])

    They can be operated on using standard mathematical operations, as well as as numpy ufuncs and functions.

    >>> fs + 1
    FreqSevSims(array([0, 0, 1, 1, 1, 2, 2, 2, 2]), array([ 2,  3,  4,  5,  6,  7,  8,  9, 10]), 3)
    >>> np.maximum(fs, 5)
    FreqSevSims(array([0, 0, 1, 1, 1, 2, 2, 2, 2]), array([5, 5, 5, 5, 5, 6, 7, 8, 9]), 3)
    >>> np.where(fs > 5, 1, 0)
    FreqSevSims(array([0, 0, 1, 1, 1, 2, 2, 2, 2]), array([0, 0, 0, 0, 0, 1, 1, 1, 1]), 3)

    FreqSevSims objects can be multiplied, added, subtracted, divided, and compared with other FreqSevSims objects,
    provided that the simulation indices match.

    >>> fs1 = FreqSevSims(sim_index, values, n_sims)
    >>> fs2 = FreqSevSims(sim_index, values, n_sims)
    >>> fs1 + fs2
    FreqSevSims(array([0, 0, 1, 1, 1, 2, 2, 2, 2]), array([ 2,  4,  6,  8, 10, 12, 14, 16, 18]), 3)




    """

    def __init__(self, sim_index: np.ndarray, values: np.ndarray, n_sims: int):
        """
        Create a new FreqSevSims object out the list of simulation indices, and the list of values corresponding to
        each simulation index. Note, the simulation indices are assumed to be ordered and 0-indexed.


        Parameters:
        sim_index (np.ndarray): Array of simulation indices.
        values (np.ndarray): Array of values.
        n_sims (int): Number of simulations.

        Raises:
        AssertionError: If values and sim_index are not of type np.ndarray or if their sizes don't match.


        """
        self.sim_index = sim_index
        self.values = values
        self.n_sims = n_sims
        assert isinstance(self.values, np.ndarray)
        assert isinstance(self.sim_index, np.ndarray)
        assert self.sim_index.size == self.values.size

    def __str__(self):
        return (
            "Simulation Index\n"
            + str(self.sim_index)
            + "\n Values\n"
            + str(self.values)
        )

    def aggregate(self):
        """Calculates the aggregate loss for each simulation.

        >>> sim_index = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
        >>> values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> n_sims = 3
        >>> fs = FreqSevSims(sim_index, values, n_sims)
        >>> fs.aggregate()
        array([ 3., 12., 30.])

        Returns:
            numpy.ndarray: An array containing the aggregate loss for each simulation.
        """
        result = np.zeros(self.n_sims)
        np.add.at(result, self.sim_index, self.values)
        return result

    def occurrence(self):
        """Calculates the maximum occurrence loss for each simulation.

        >>> sim_index = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
        >>> values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> n_sims = 3
        >>> fs = FreqSevSims(sim_index, values, n_sims)
        >>> fs.occurrence()
        array([2., 5., 9.])

        Returns:
            numpy.ndarray: An array containing the aggregate loss for each simulation.
        """
        result = np.zeros(self.n_sims)
        np.maximum.at(result, self.sim_index, self.values)
        return result

    def copy(self) -> "FreqSevSims":
        """Creates a copy of the FreqSevSims object."""
        return FreqSevSims(self.sim_index, self.values.copy(), self.n_sims)

    def apply(self, func) -> "FreqSevSims":
        """Applies a function to the values of the FreqSevSims object."""
        return FreqSevSims(self.sim_index, func(self.values), self.n_sims)

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs, **kwargs
    ) -> "FreqSevSims":
        inputs = tuple(x.values if isinstance(x, FreqSevSims) else x for x in inputs)
        out = kwargs.get("out", ())
        if out:
            kwargs["out"] = tuple(x.values for x in out)
        result = getattr(ufunc, method)(*inputs, **kwargs)
        return FreqSevSims(self.sim_index, result, self.n_sims)

    def __array_function__(self, func: np.ufunc, types, args, kwargs):
        if func not in (numpy.where, numpy.maximum, numpy.sum):
            raise NotImplementedError
        args = tuple(x.values if isinstance(x, FreqSevSims) else x for x in args)
        result = func(*args, **kwargs)
        if isinstance(result, np.number):
            return result
        return FreqSevSims(self.sim_index, result, self.n_sims)

    def __add__(self, x: Union["FreqSevSims", int, float | np.ndarray]):
        if self._is_compatible(x):
            return FreqSevSims(self.sim_index, self.values + x.values, self.n_sims)
        elif isinstance(x, int) or isinstance(x, float):
            return FreqSevSims(self.sim_index, self.values + x, self.n_sims)
        elif isinstance(x, np.ndarray):
            return FreqSevSims(
                self.sim_index, self.values + x[self.sim_index], self.n_sims
            )
        else:
            raise NotImplementedError

    def __radd__(self, x: Union["FreqSevSims", int, float | np.ndarray]):
        return self.__add__(x)

    def __sub__(self, x: Union["FreqSevSims", int, float | np.ndarray]):
        if self._is_compatible(x):
            return FreqSevSims(self.sim_index, self.values - x.values, self.n_sims)
        elif isinstance(x, int) or isinstance(x, float):
            return FreqSevSims(self.sim_index, self.values - x, self.n_sims)
        elif isinstance(x, np.ndarray):
            return FreqSevSims(
                self.sim_index, self.values - x[self.sim_index], self.n_sims
            )
        else:
            raise NotImplementedError

    def __rsub__(self, x: Union["FreqSevSims", int, float | np.ndarray]):
        return -self.__sub__(x)

    def __neg__(self):
        return FreqSevSims(self.sim_index, -self.values, self.n_sims)

    def __mul__(self, other: Union["FreqSevSims", int, float | np.ndarray]):
        if self._is_compatible(other):
            return FreqSevSims(self.sim_index, self.values * other.values, self.n_sims)
        elif isinstance(other, int) or isinstance(other, float):
            return FreqSevSims(self.sim_index, self.values * other, self.n_sims)
        elif isinstance(other, np.ndarray):
            return FreqSevSims(
                self.sim_index, self.values * other[self.sim_index], self.n_sims
            )
        else:
            raise NotImplementedError

    def __truediv__(self, other: Union["FreqSevSims", int, float | np.ndarray]):
        if self._is_compatible(other):
            return FreqSevSims(self.sim_index, self.values / other.values, self.n_sims)
        elif isinstance(other, int) or isinstance(other, float):
            return FreqSevSims(self.sim_index, self.values / other, self.n_sims)
        elif isinstance(other, np.ndarray):
            return FreqSevSims(
                self.sim_index, self.values / other[self.sim_index], self.n_sims
            )
        else:
            raise NotImplementedError

    def __rtruediv__(self, other: Union["FreqSevSims", int, float | np.ndarray]):
        return self.__mul__(1 / other)

    def __rmul__(self, other: Union["FreqSevSims", int, float | np.ndarray]):
        return self.__mul__(other)

    def __lt__(self, other: Union["FreqSevSims", int, float | np.ndarray]):
        if self._is_compatible(other):
            return FreqSevSims(self.sim_index, self.values < other.values, self.n_sims)
        elif isinstance(other, int) or isinstance(other, float):
            return FreqSevSims(self.sim_index, self.values < other, self.n_sims)
        elif isinstance(other, np.ndarray):
            return FreqSevSims(
                self.sim_index, self.values < other[self.sim_index], self.n_sims
            )
        else:
            raise NotImplementedError

    def __le__(self, other: Union["FreqSevSims", int, float | np.ndarray]):
        if self._is_compatible(other):
            return FreqSevSims(self.sim_index, self.values <= other.values, self.n_sims)
        elif isinstance(other, int) or isinstance(other, float):
            return FreqSevSims(self.sim_index, self.values <= other, self.n_sims)
        elif isinstance(other, np.ndarray):
            return FreqSevSims(
                self.sim_index, self.values <= other[self.sim_index], self.n_sims
            )
        else:
            raise NotImplementedError

    def __gt__(self, other: Union["FreqSevSims", int, float | np.ndarray]):
        if self._is_compatible(other):
            return FreqSevSims(self.sim_index, self.values > other.values, self.n_sims)
        elif isinstance(other, int) or isinstance(other, float):
            return FreqSevSims(self.sim_index, self.values > other, self.n_sims)
        elif isinstance(other, np.ndarray):
            return FreqSevSims(
                self.sim_index, self.values > other[self.sim_index], self.n_sims
            )
        else:
            raise NotImplementedError

    def __ge__(self, other: Union["FreqSevSims", int, float | np.ndarray]):
        if self._is_compatible(other):
            return FreqSevSims(self.sim_index, self.values >= other.values, self.n_sims)
        elif isinstance(other, int) or isinstance(other, float):
            return FreqSevSims(self.sim_index, self.values >= other, self.n_sims)
        elif isinstance(other, np.ndarray):
            return FreqSevSims(
                self.sim_index, self.values >= other[self.sim_index], self.n_sims
            )
        else:
            raise NotImplementedError

    def __eq__(self, other: Union["FreqSevSims", int, float | np.ndarray]):
        if self._is_compatible(other):
            return FreqSevSims(self.sim_index, self.values == other.values, self.n_sims)
        elif isinstance(other, int) or isinstance(other, float):
            return FreqSevSims(self.sim_index, self.values == other, self.n_sims)
        elif isinstance(other, np.ndarray):
            return FreqSevSims(
                self.sim_index, self.values == other[self.sim_index], self.n_sims
            )
        else:
            raise NotImplementedError

    def __and__(self, other: Union["FreqSevSims", int, float | np.ndarray]):
        if self._is_compatible(other):
            return FreqSevSims(
                self.sim_index, (self.values) & (other.values), self.n_sims
            )
        elif (
            isinstance(other, int)
            or isinstance(other, float)
            or isinstance(other, bool)
        ):
            return FreqSevSims(self.sim_index, (self.values) & other, self.n_sims)
        elif isinstance(other, np.ndarray):
            return FreqSevSims(
                self.sim_index, (self.values) & (other[self.sim_index]), self.n_sims
            )
        else:
            raise NotImplementedError

    def __rand__(self, other):
        return self.__and__(other)

    def __or__(self, other: Union["FreqSevSims", int, float | np.ndarray]):
        if self._is_compatible(other):
            return FreqSevSims(
                self.sim_index, (self.values) | (other.values), self.n_sims
            )
        elif (
            isinstance(other, int)
            or isinstance(other, float)
            or isinstance(other, bool)
        ):
            return FreqSevSims(self.sim_index, (self.values) | other, self.n_sims)
        elif isinstance(other, np.ndarray):
            return FreqSevSims(
                self.sim_index, (self.values) | (other[self.sim_index]), self.n_sims
            )
        else:
            raise NotImplementedError

    def __ror__(self, other: Union["FreqSevSims", int, float | np.ndarray]):
        return self.__or__(other)

    def __repr__(self):
        return "%s(%r)" % (type(self).__name__, self.values)

    def _is_compatible(self, other: Union["FreqSevSims", int, float | np.ndarray]):
        """Check if two FreqSevSims objects are compatible for mathematical operations."""
        return isinstance(other, type(self)) and self.sim_index is other.sim_index
