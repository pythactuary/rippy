from __future__ import annotations
from typing import Union
from .FrequencySeverity import FreqSevSims
from .stochastic_scalar import StochasticScalar
import numpy as np
import scipy.stats
import plotly.graph_objects as go


class ProteusVariable:
    """A class to hold a multivariate variable in a simulation.

    A Proteus Variable is a hierarchical structure that can hold multiple
    scalar variables. The purpose of this class is to allow
    for the creation of more complex variables that can be used in
    simulations.

    Each level of a Proteus Variable can be a list or dictionary of scalar variables or other ProteusVariable objects. Each level can have a different number of elements.
    Each level has a name that can be used to access the level in the hierarchy.

    Sub elements of a ProteusVariable can be accessed using the [] notation.

    """

    def __len__(self):
        return len(self.values)

    def __init__(
        self,
        dim_name: str,
        values: (
            list[Union[ProteusVariable, StochasticScalar | FreqSevSims | float | int]]
            | dict[
                str,
                Union[ProteusVariable, StochasticScalar | FreqSevSims | float | int],
            ]
        ),
    ):
        self.dim_name: str = dim_name
        self.values = values
        self.dimensions = [dim_name]
        self._dimension_set = set(self.dimensions)
        # check the number of simulations in each variable
        self.n_sims = None
        for value in (
            self.values.values() if isinstance(self.values, dict) else self.values
        ):
            if isinstance(value, ProteusVariable):
                if (
                    self._dimension_set.intersection(value._dimension_set)
                    or self.dim_name == value.dim_name
                ):
                    raise ValueError(
                        "Duplicate dimension names in ProteusVariable hierarchy."
                    )
                self._dimension_set.intersection_update(value.dimensions)
                self.dimensions.extend(value.dimensions)

            if self.n_sims is None:
                if isinstance(value, ProteusVariable) or isinstance(
                    value, StochasticScalar
                ):
                    self.n_sims = value.n_sims
                self.n_sims = 1
            elif isinstance(value, ProteusVariable) or isinstance(
                value, StochasticScalar
            ):
                if value.n_sims != self.n_sims:
                    if self.n_sims == 1:
                        self.n_sims == value.n_sims
                    else:
                        raise ValueError("Number of simulations do not match.")

    def sum(self, dimensions: list[str] = []) -> ProteusVariable | StochasticScalar:
        """Sum the variables across the specified dimensions. Returns a new ProteusVariable with the summed values."""
        if dimensions is None or dimensions == []:
            result: StochasticScalar = sum(self)
            return result
        if self.dimensions in dimensions:
            result = ProteusVariable(dim_name=self.values[0].dimensions, values=0)
            for value in self.values:
                if isinstance(value, ProteusVariable | StochasticScalar):
                    result = result + value.sum(dimensions)
                else:
                    result = result + value
            return result
        else:
            return self

    def __iter__(self):
        if isinstance(self.values, dict):
            return iter(self.values.values())
        else:
            return iter(self.values)

    def _binary_operation(self, other, operation):
        if isinstance(other, ProteusVariable):
            if self.dimensions != other.dimensions:
                raise ValueError("Dimensions of the two variables do not match.")
        if isinstance(self.values, dict):
            if isinstance(other, ProteusVariable):
                return ProteusVariable(
                    dim_name=self.dim_name,
                    values={
                        key: operation(value, other.values[key])
                        for key, value in self.values.items()
                    },
                )
            return ProteusVariable(
                dim_name=self.dim_name,
                values={
                    key: operation(value, other) for key, value in self.values.items()
                },
            )
        elif isinstance(self.values, list):
            if isinstance(other, ProteusVariable):
                return ProteusVariable(
                    dim_name=self.dim_name,
                    values=[
                        operation(value, other.values[i])
                        for i, value in enumerate(self.values)
                    ],
                )
            return ProteusVariable(
                dim_name=self.dim_name,
                values=[operation(value, other) for i, value in enumerate(self.values)],
            )

    def __add__(
        self, other: ProteusVariable | StochasticScalar | float | int
    ) -> ProteusVariable:
        """Add two ProteusVariable objects together. Returns a new ProteusVariable object with the summed values."""
        return self._binary_operation(other, lambda a, b: a + b)

    def __radd__(self, other) -> ProteusVariable:
        return self.__add__(other)

    def __mul__(self, other) -> ProteusVariable:
        return self._binary_operation(other, lambda a, b: a * b)

    def __rmul__(self, other) -> ProteusVariable:
        return self.__mul__(other)

    def __getitem__(self, key: str | int):
        if isinstance(self.values, dict):
            if isinstance(key, int):
                return self.values[list(self.values.keys())[key]]
            else:
                return self.values[key]
        else:
            if isinstance(key, int):
                return self.values[key]
            else:
                raise ValueError("Key must be an integer for a list.")

    def get_value_at_sim(self, sim_no: int | StochasticScalar):
        _get_value = lambda x: (
            x.get_value_at_sim(sim_no) if isinstance(x, ProteusVariable) else x[sim_no]
        )
        if isinstance(self.values, dict):
            result = ProteusVariable(
                dim_name=self.dim_name,
                values={k: _get_value(v) for k, v in self.values.items()},
            )
        elif isinstance(self.values, list):
            result = ProteusVariable(
                dim_name=self.dim_name,
                values=[_get_value(v) for v in self.values],
            )
        return result

    def correlation_matrix(self, correlation_type="spearman") -> list[list[float]]:
        # validate type
        correlation_type = correlation_type.lower()
        assert correlation_type in ["linear", "spearman", "kendall"]
        assert hasattr(self[0], "values")
        n = len(self.values)
        result: list[list[float]] = [[0.0] * n] * n
        values = [self[i] for i in range(len(self.values))]
        if correlation_type.lower() in ["spearman", "kendall"]:
            # rank the variables first
            for i, value in enumerate(values):
                values[i] = scipy.stats.rankdata(value.values)

        if correlation_type == "kendall":
            for i, value1 in enumerate(values):
                for j, value2 in enumerate(values):
                    result[i][j] = scipy.stats.kendalltau(value1, value2)
        else:
            result = list(np.corrcoef(values))

        return result

    def show_histogram(self):

        fig = go.Figure()
        labels = (
            self.values.keys()
            if isinstance(self.values, dict)
            else range(len(self.values))
        )
        for value, label in zip(self.values.values(), labels):
            fig.add_trace(go.Histogram(x=value.values, name=label))
        fig.show()

    def show_cdf(self):

        fig = go.Figure()
        labels = (
            self.values.keys()
            if isinstance(self.values, dict)
            else range(len(self.values))
        )
        for value, label in zip(self.values.values(), labels):
            fig.add_trace(
                go.Scatter(
                    x=np.sort(value.values),
                    y=np.arange(value.n_sims) / value.n_sims,
                    name=label,
                )
            )
        fig.show()
