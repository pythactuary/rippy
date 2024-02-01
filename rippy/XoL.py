import numpy as np
from .FreqSevSims import FreqSevSims
from dataclasses import dataclass


@dataclass
class ContractResults:
    def __init__(self, recoveries: FreqSevSims, reinstatement_premium: np.ndarray):
        self.recoveries = recoveries
        self.reinstatement_premium = reinstatement_premium


class XoL:
    def __init__(
        self,
        name: str,
        limit: float,
        excess: float,
        premium: float,
        reinstatement_cost: list[float] | None = None,
        aggregate_limit: float = None,
        aggregate_deductible: float = None,
        franchise: float = None,
        reverse_franchise: float = None,
    ):
        """
        Initialize a new XoL layer.

        Args:
            name (str): The name of the XoL layer.
            limit (float): The limit of coverage.
            excess (float): The excess amount.
            premium (float): The premium amount.
            reinstatement_cost (list[float] | None, optional): The reinstatement cost as a fraction of the base premium, by reinstatement. Defaults to None.
            aggregate_limit (float, optional): The aggregate limit. Defaults to None.
            aggregate_deductible (float, optional): The aggregate deductible. Defaults to None.
            franchise (float, optional): The franchise amount. Defaults to None.
            reverse_franchise (float, optional): The reverse franchise amount. Defaults to None.
        """
        self.name = name
        self.limit = limit
        self.excess = excess
        self.aggregate_limit = aggregate_limit
        self.premium = premium
        self.aggregate_deductible = aggregate_deductible
        self.franchise = franchise
        self.reverse_franchise = reverse_franchise
        self.summary = None
        self.num_reinstatements = (
            aggregate_limit / limit - 1 if aggregate_limit is not None else None
        )
        self.reinstatement_premium_cost = (
            np.array(reinstatement_cost) if reinstatement_cost is not None else None
        )

    def apply(self, claims: FreqSevSims) -> ContractResults:
        """
        Apply the XoL contract to a set of claims.

        Args:
            claims (FreqSevSims): The claims to apply the contract to.

        Returns:
            ContractResults: The results of applying the contract.

        Calculation of the recoveries from the excess of loss contract:

        Firstly, the effect of any franchise or reverse franchise is calculated on the individual losses.

        losses post franchise = loss if loss >= franchise and loss<= reverse franchise

        Next the individual losses to the layer are calculated:

        layer_loss = min(max(losses post franchise - excess, 0), limit)

        Then the aggregate layer losses before aggregate limit and deductible are calculated.
        The aggregate limit and deductible are then applied to get the aggregate recoveries for the layer:

        aggregate_recoveries = min(max(aggregate_layer_losses - aggregate_deductible, 0), aggregate_limit)

        The aggregate recoveries are then allocated back to the individual losses in proportion to the individual recoveries before aggregate limit and deductible.

        The reinstatement premium is calculated as the sum of the reinstatement premium cost multiplied by the number of reinstatements used.
        The number of reinstatements used is calculated as the minimum of the aggregate recoveries divided by the limit and the number of reinstatements available (which is the aggregate limit divided by the occurrence limit, less one).
        """
        # apply franchise
        if self.franchise is not None or self.reverse_franchise is not None:
            franchise = self.franchise if self.franchise is not None else 0
            reverse_franchise = (
                self.reverse_franchise if self.reverse_franchise is not None else np.inf
            )
            claims = np.where(
                (claims >= franchise) & (claims < reverse_franchise), claims, 0
            )

        individual_recoveries_pre_aggregate = np.minimum(
            np.maximum(claims - self.excess, 0), self.limit
        )
        if self.aggregate_limit is None and self.aggregate_deductible is None:
            return ContractResults(individual_recoveries_pre_aggregate, None)
        aggregate_limit = (
            self.aggregate_limit if self.aggregate_limit is not None else np.inf
        )
        aggregate_deductible = (
            self.aggregate_deductible if self.aggregate_deductible is not None else 0
        )
        aggregate_recoveries_pre_agg = individual_recoveries_pre_aggregate.aggregate()

        aggregate_recoveries = np.minimum(
            np.maximum(aggregate_recoveries_pre_agg - aggregate_deductible, 0),
            aggregate_limit,
        )
        ratio = np.where(
            aggregate_recoveries_pre_agg == 0,
            1,
            np.divide(
                aggregate_recoveries,
                aggregate_recoveries_pre_agg,
                where=aggregate_recoveries_pre_agg != 0,
            ),
        )
        recoveries = individual_recoveries_pre_aggregate * ratio
        results = ContractResults(recoveries, None)
        if self.reinstatement_premium_cost is not None:
            cumulative_reinstatement_cost = np.cumsum(self.reinstatement_premium_cost)
            limits_used = aggregate_recoveries / self.limit
            reinstatements_used = np.minimum(limits_used, self.num_reinstatements)
            reinstatements_used_full = np.floor(reinstatements_used).astype(int)
            reinstatements_used_fraction = (
                reinstatements_used - reinstatements_used_full
            )
            reinstatement_number = np.maximum(reinstatements_used_full - 1, 0)
            reinstatement_premium_proportion = self.reinstatement_premium_cost[
                reinstatement_number
            ] * reinstatements_used_fraction + np.where(
                reinstatements_used_full > 0,
                cumulative_reinstatement_cost[reinstatement_number],
                0,
            )

            reinstatement_premium = reinstatement_premium_proportion * self.premium
            results.reinstatement_premium = reinstatement_premium

        self.calc_summary(claims, recoveries, aggregate_recoveries)
        return results

    def print_summary(self):
        """Print a summary of the losses to the layer"""
        print("Layer Name : {}".format(self.name))
        print("Mean Recoveries: ", self.summary["mean"])
        print("SD Recoveries: ", self.summary["std"])
        print("Probability of Attachment: ", self.summary["prob_attach"]),

        print(
            "Probability of Vertical Exhaustion: ", self.summary["prob_vert_exhaust"]
        ),
        print(
            "Probability of Horizontal Exhaustion: ",
            self.summary["prob_horizonal_exhaust"],
        ),
        print("")


class XoLTower:
    def __init__(
        self,
        limit: list,
        excess: list,
        premium: float,
        reinstatement_cost: list[float] | None = None,
        aggregate_deductible: float = None,
        aggregate_limit: float = None,
        franchise: float = None,
        reverse_franchise: float = None,
    ):
        self.limit = limit
        self.excess = excess
        self.aggregate_limit = aggregate_limit
        self.aggregate_deductible = aggregate_deductible
        self.franchise = franchise
        self.reverse_franchise = reverse_franchise
        self.n_layers = len(limit)
        self.layers = [
            XoL(
                "Layer {}".format(i + 1),
                limit[i],
                excess[i],
                premium[i],
                reinstatement_cost[i],
                aggregate_limit[i] if aggregate_limit is not None else None,
                aggregate_deductible[i] if aggregate_deductible is not None else None,
                franchise[i] if franchise is not None else None,
                reverse_franchise[i] if reverse_franchise is not None else None,
            )
            for i in range(self.n_layers)
        ]

    def apply(self, claims: FreqSevSims) -> FreqSevSims:
        """Apply the XoL contract to a set of claims"""
        recoveries = claims.copy() * 0
        reinstatement_premium = np.zeros(claims.n_sims)
        for layer in self.layers:
            layer_results = layer.apply(claims)
            recoveries += layer_results.recoveries
            reinstatement_premium += (
                layer_results.reinstatement_premium
                if layer_results.reinstatement_premium is not None
                else 0
            )

        return ContractResults(recoveries, reinstatement_premium)

    def print_summary(self):
        """Print a summary of the program losses"""
        for layer in self.layers:
            layer.print_summary()
