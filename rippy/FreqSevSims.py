from collections.abc import Iterable
import numpy as np


def _is_compatible(self, other):
    return isinstance(other, FreqSevSims) and self.sim_index is other.sim_index


class FreqSevSims:
    def __init__(self, sim_index: np.ndarray, values: np.ndarray, n_sims: int):
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
        """Calculates the aggregate loss for each simulation"""
        return np.bincount(self.sim_index, self.values, self.n_sims)

    def occurrence(self):
        """Calculates the maximum occurrence loss for each simulation"""
        result = np.zeros(self.n_sims)
        np.maximum.at(result, self.sim_index, self.values)
        return result

    def copy(self):
        return FreqSevSims(self.sim_index, self.values.copy(), self.n_sims)

    def apply(self, func):
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
        if func not in (np.where,):
            raise NotImplementedError
        args = tuple(x.values if isinstance(x, FreqSevSims) else x for x in args)
        result = func(*args, **kwargs)
        print("trying array function")
        print(args)
        return FreqSevSims(self.sim_index, result, self.n_sims)

    def __add__(self, x):
        if _is_compatible(self, x):
            return FreqSevSims(self.sim_index, self.values + x.values, self.n_sims)
        elif isinstance(x, int) or isinstance(x, float):
            return FreqSevSims(self.sim_index, self.values + x, self.n_sims)
        elif isinstance(x, np.ndarray):
            print(self.sim_index)
            return FreqSevSims(
                self.sim_index, self.values + x[self.sim_index], self.n_sims
            )
        else:
            raise NotImplementedError

    def __radd__(self, x):
        return self.__add__(x)

    def __sub__(self, x):
        if _is_compatible(self, x):
            return FreqSevSims(self.sim_index, self.values - x.values, self.n_sims)
        elif isinstance(x, int) or isinstance(x, float):
            return FreqSevSims(self.sim_index, self.values - x, self.n_sims)
        elif isinstance(x, np.ndarray):
            return FreqSevSims(
                self.sim_index, self.values + x[self.sim_index], self.n_sims
            )
        else:
            raise NotImplementedError

    def __rsub__(self, x):
        return -self.__sub__(x)

    def __neg__(self):
        return FreqSevSims(self.sim_index, -self.values, self.n_sims)

    def __mul__(self, other):
        if _is_compatible(self, other):
            return FreqSevSims(self.sim_index, self.values * other.values, self.n_sims)
        elif isinstance(other, int) or isinstance(other, float):
            return FreqSevSims(self.sim_index, self.values * other, self.n_sims)
        elif isinstance(other, np.ndarray):
            return FreqSevSims(
                self.sim_index, self.values * other[self.sim_index], self.n_sims
            )
        else:
            raise NotImplementedError

    def __truediv__(self, other):
        if _is_compatible(self, other):
            return FreqSevSims(self.sim_index, self.values / other.values, self.n_sims)
        elif isinstance(other, int) or isinstance(other, float):
            return FreqSevSims(self.sim_index, self.values / other, self.n_sims)
        elif isinstance(other, np.ndarray):
            return FreqSevSims(
                self.sim_index, self.values / other[self.sim_index], self.n_sims
            )
        else:
            raise NotImplementedError

    def __rtruediv__(self, other):
        return self.__mul__(1 / other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __lt__(self, other):
        if _is_compatible(self, other):
            return FreqSevSims(self.sim_index, self.values < other.values, self.n_sims)
        elif isinstance(other, int) or isinstance(other, float):
            return FreqSevSims(self.sim_index, self.values < other, self.n_sims)
        elif isinstance(other, np.ndarray):
            return FreqSevSims(
                self.sim_index, self.values < other[self.sim_index], self.n_sims
            )
        else:
            raise NotImplementedError

    def __le__(self, other):
        if _is_compatible(self, other):
            return FreqSevSims(self.sim_index, self.values <= other.values, self.n_sims)
        elif isinstance(other, int) or isinstance(other, float):
            return FreqSevSims(self.sim_index, self.values <= other, self.n_sims)
        elif isinstance(other, np.ndarray):
            return FreqSevSims(
                self.sim_index, self.values <= other[self.sim_index], self.n_sims
            )
        else:
            raise NotImplementedError

    def __gt__(self, other):
        if _is_compatible(self, other):
            return FreqSevSims(self.sim_index, self.values > other.values, self.n_sims)
        elif isinstance(other, int) or isinstance(other, float):
            return FreqSevSims(self.sim_index, self.values > other, self.n_sims)
        elif isinstance(other, np.ndarray):
            return FreqSevSims(
                self.sim_index, self.values > other[self.sim_index], self.n_sims
            )
        else:
            raise NotImplementedError

    def __ge__(self, other):
        if _is_compatible(self, other):
            return FreqSevSims(self.sim_index, self.values >= other.values, self.n_sims)
        elif isinstance(other, int) or isinstance(other, float):
            print("trying ge")
            print(self.values >= other)
            print(FreqSevSims(self.sim_index, self.values >= other, self.n_sims))
            return FreqSevSims(self.sim_index, self.values >= other, self.n_sims)
        elif isinstance(other, np.ndarray):
            return FreqSevSims(
                self.sim_index, self.values >= other[self.sim_index], self.n_sims
            )
        else:
            raise NotImplementedError

    def __eq__(self, other):
        if _is_compatible(self, other):
            return FreqSevSims(self.sim_index, self.values == other.values, self.n_sims)
        elif isinstance(other, int) or isinstance(other, float):
            return FreqSevSims(self.sim_index, self.values == other, self.n_sims)
        elif isinstance(other, np.ndarray):
            return FreqSevSims(
                self.sim_index, self.values == other[self.sim_index], self.n_sims
            )
        else:
            raise NotImplementedError

    def __and__(self, other):
        if _is_compatible(self, other):
            print("testing and")
            return FreqSevSims(
                self.sim_index, (self.values) & (other.values), self.n_sims
            )
        elif (
            isinstance(other, int)
            or isinstance(other, float)
            or isinstance(other, bool)
        ):
            print("testing and with int")
            return FreqSevSims(self.sim_index, (self.values) & other, self.n_sims)
        elif isinstance(other, np.ndarray):
            print("testing and with ndarray")
            return FreqSevSims(
                self.sim_index, (self.values) & (other[self.sim_index]), self.n_sims
            )
        else:
            raise NotImplementedError

    def __rand__(self, other):
        return self.__and__(other)

    def __or__(self, other):
        if _is_compatible(self, other):
            print("testing and")
            return FreqSevSims(
                self.sim_index, (self.values) | (other.values), self.n_sims
            )
        elif (
            isinstance(other, int)
            or isinstance(other, float)
            or isinstance(other, bool)
        ):
            print("testing and with int")
            return FreqSevSims(self.sim_index, (self.values) | other, self.n_sims)
        elif isinstance(other, np.ndarray):
            print("testing and with ndarray")
            return FreqSevSims(
                self.sim_index, (self.values) | (other[self.sim_index]), self.n_sims
            )
        else:
            raise NotImplementedError

    def __ror__(self, other):
        return self.__or__(other)

    def __repr__(self):
        return "%s(%r)" % (type(self).__name__, self.values)
