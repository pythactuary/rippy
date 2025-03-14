import scipy.special
from .config import config, xp as np, _use_gpu
from .variables import StochasticScalar, ProteusVariable
import scipy.stats

if _use_gpu:
    import cupyx.scipy.special as special
else:
    import scipy.special as special
from abc import ABC, abstractmethod
from line_profiler import profile


class Copula(ABC):
    """A class to represent a copula"""

    @abstractmethod
    def generate(self, n_sims=None, rng=config.rng) -> ProteusVariable:
        """Generate samples from the copula"""
        pass

    def apply(self, variables: ProteusVariable | list[StochasticScalar]):
        """Apply the copula to a list of variables"""
        # generate the copula samples
        copula_samples = self.generate(n_sims=variables[0].n_sims)
        if len(variables) != len(copula_samples):
            raise ValueError("Number of variables and copula samples do not match.")
        # apply the copula to the variables
        apply_copula(variables, copula_samples)


class EllipticalCopula(Copula, ABC):
    """A base class to represent an Elliptical copula"""

    matrix: np.ndarray
    chol: np.ndarray

    def __init__(
        self, matrix: np.typing.ArrayLike, *args, matrix_type: str = "linear", **kwargs
    ):
        if matrix_type == "linear":
            self.correlation_matrix = np.asarray(matrix)
            # check that the correlation matrix is positive definite
            try:
                self.chol = np.linalg.cholesky(self.correlation_matrix)
            except np.linalg.LinAlgError:
                raise ValueError("Correlation matrix is not positive definite")
        elif matrix_type == "chol":
            self.chol = matrix
        self.matrix = matrix


class GaussianCopula(EllipticalCopula):
    """A class to represent a Gaussian copula"""

    def __init__(self, matrix: np.typing.ArrayLike, matrix_type: str = "linear"):
        super().__init__(matrix, matrix_type=matrix_type)

    @profile
    def generate(self, n_sims=None, rng=config.rng) -> ProteusVariable:
        """Generate samples from the copula"""
        if n_sims is None:
            n_sims = config.n_sims
        n_vars = self.correlation_matrix.shape[0]
        # Generate samples from a multivariate normal distribution
        samples = self.chol.dot(
            rng.multivariate_normal(
                mean=np.zeros(n_vars), cov=np.eye(n_vars), size=n_sims
            ).T
        )
        uniform_samples = special.ndtr(samples)
        result = ProteusVariable(
            "dim1", [StochasticScalar(sample) for sample in uniform_samples]
        )
        for val in result:
            val.coupled_variable_group.merge(result[0].coupled_variable_group)
        return result


class StudentsTCopula(EllipticalCopula):
    """A class to represent a Student's T copula"""

    def __init__(
        self, matrix: np.typing.ArrayLike, dof: float, matrix_type: str = "linear"
    ):
        super().__init__(matrix, matrix_type=matrix_type)
        if dof <= 0:
            raise ValueError("Degrees of Freedom cannot be negative")
        self.dof = dof

    def generate(self, n_sims=None, rng=config.rng):
        if n_sims is None:
            n_sims = config.n_sims
        n_vars = len(self.matrix)
        normal_samples = self.chol.dot(
            rng.multivariate_normal(
                mean=np.zeros(n_vars), cov=np.eye(n_vars), size=n_sims
            ).T
        )
        chi_samples = np.sqrt(rng.gamma(self.dof / 2, 2 / self.dof, size=n_sims))
        t_samples = normal_samples / chi_samples[np.newaxis, :]
        uniform_samples = scipy.stats.distributions.t(self.dof).cdf(t_samples)

        return ProteusVariable(
            "dim1", [StochasticScalar(sample) for sample in uniform_samples]
        )


class ArchimedeanCopula(Copula, ABC):
    """A class to represent an Archimedean copula"""

    @abstractmethod
    def generator_inv(self, t: np.typing.ArrayLike) -> np.typing.ArrayLike:
        """The inverse generator function of the copula"""
        pass

    @abstractmethod
    def generate_latent_distribution(self, n_sims, rng) -> np.typing.ArrayLike:
        """Generate samples from the latent distribution of the copula"""
        pass

    def __init__(self, n: int):
        self.n = n

    def generate(self, n_sims=None, rng=config.rng) -> ProteusVariable:
        if n_sims is None:
            n_sims = config.n_sims
        """Generate samples from an Archimedean copula"""
        n_vars = self.n
        # Generate samples from a uniform distribution
        u = rng.uniform(size=(n_vars, n_sims))
        # Generate samples from the latent distribution
        latent_samples = self.generate_latent_distribution(n_sims, rng)
        # Calculate the copula samples
        copula_samples = self.generator_inv(-np.log(u) / latent_samples[np.newaxis])
        result = ProteusVariable(
            "dim1", [StochasticScalar(sample) for sample in copula_samples]
        )
        for val in result:
            val.coupled_variable_group.merge(result[0].coupled_variable_group)
        return result


class ClaytonCopula(ArchimedeanCopula):
    """A class to represent a Clayton copula"""

    def __init__(self, theta: float, n: int):
        # Check that the parameter is positive
        if theta < 0:
            raise ValueError("Theta cannot be negative")
        self.theta = theta
        self.n = n

    def generator_inv(self, t: np.typing.ArrayLike) -> np.typing.ArrayLike:
        if self.theta == 0:
            return np.exp(-t)
        return (1 + t) ** (-1 / self.theta)

    def generate_latent_distribution(
        self, n_sims, rng: np.random.Generator
    ) -> np.typing.ArrayLike:
        if self.theta == 0:
            return np.array([1])
        return rng.gamma(1 / self.theta, size=n_sims)


class GumbelCopula(ArchimedeanCopula):
    """A class to represent a Gumbel copula"""

    def __init__(self, theta: float, n: int):
        # Check that the parameter is not less than one
        if theta < 1:
            raise ValueError("Theta must be positive")
        self.theta = theta
        self.n = n

    def generator_inv(self, t: np.typing.ArrayLike) -> np.typing.ArrayLike:
        return np.exp(-(t ** (1 / self.theta)))

    def generate_latent_distribution(self, n_sims, rng) -> np.typing.ArrayLike:
        return scipy.stats.distributions.levy_stable.rvs(
            1.0 / self.theta,
            1.0,
            0,
            np.cos(np.pi / (2 * self.theta)) ** self.theta,
            size=n_sims,
            random_state=rng,
        )


class FrankCopula(ArchimedeanCopula):
    """A class to represent a Frank copula"""

    def __init__(self, theta: float, n: int):
        # Check that the parameter is not less than one
        if theta < 0:
            raise ValueError("Theta must be positive")
        self.theta = theta
        self.n = n

    def generator_inv(self, t: np.typing.ArrayLike) -> np.typing.ArrayLike:
        return -np.log1p(np.exp(-t) * (np.expm1(-self.theta))) / self.theta

    def generate_latent_distribution(
        self, n_sims, rng: np.random.Generator
    ) -> np.typing.ArrayLike:
        """Generate samples from the distribution of the latent variable of the Frank copula,
        which is a logseries distribution."""
        return rng.logseries(1 - np.exp(-self.theta), size=n_sims)


class JoeCopula(ArchimedeanCopula):
    """A class to represent a Joe copula"""

    def __init__(self, theta: float, n: int):
        # Check that the parameter is not less than one
        if theta < 1:
            raise ValueError("Theta must be in the range [1, inf)")
        self.theta = theta
        self.n = n

    def generator_inv(self, t: np.typing.ArrayLike) -> np.typing.ArrayLike:
        return 1 - (1 - np.exp(-t)) ** (1 / self.theta)

    def generate_latent_distribution(
        self, n_sims, rng: np.random.Generator
    ) -> np.typing.ArrayLike:
        """Generate samples from the distribution of the latent variable of the Joe copula,
        which is a Sibuya distribution."""
        return _sibuya_gen(1 / self.theta, n_sims, rng)


def _sibuya_gen(alpha, size, rng: np.random.Generator):
    """Generate samples from a Sibuya distribution. This is a distribution that is a mixture of Poisson distributions with rate given
    by a mixture of an exponential distribution multiplied by the ratio of two Gamma distributions.
    """

    g1 = rng.gamma(alpha, 1, size=size)
    g2 = rng.gamma(1 - alpha, 1, size=size)
    r = g2 / g1  # this could be represented as a beta prime distribution
    e = rng.exponential(1, size=size)
    u = r * e

    return 1 + rng.poisson(u, size=size)


@profile
def apply_copula(
    variables: list[StochasticScalar],
    copula_samples: list[StochasticScalar],
):
    """Apply a re-ordering from a copula to a list of variables."""
    variables = list(variables)
    # Check that the number of variables and copula samples match.
    if len(variables) != len(copula_samples):
        raise ValueError("Number of variables and copula samples do not match.")
    # Check that the dependency groups of the variables are different.
    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables[i + 1 :]):
            if var1.coupled_variable_group is var2.coupled_variable_group:
                raise ValueError(
                    f"Cannot apply copula as the variables at positions {i} and {j+i+1} are not independent"
                )
    # calculate the rank of the copula samples
    copula_sort_indices = np.argsort(
        np.array([cs.values for cs in copula_samples]), axis=1
    )
    # calculate the reordering of the variables required to match the rank
    copula_ranks = np.argsort(copula_sort_indices, axis=1)
    # calculate the sort_indices of the variables
    variable_sort_indices = np.argsort(
        np.array([var.values for var in variables]), axis=1
    )
    first_variable_rank = np.argsort(variable_sort_indices[0])
    # rearrange the copula ranks to fit with the ranks of the first variable
    copula_ranks = copula_ranks[:, copula_sort_indices[0, first_variable_rank]]
    # apply the reordering to all of the variables
    for i, var in enumerate(variables):
        if i == 0:
            continue  # do not reorder the first variable
        re_ordering = variable_sort_indices[i, copula_ranks[i]]
        for var2 in var.coupled_variable_group.variables:
            var2._reorder_sims(re_ordering)
    # merge the dependency groups
    for i, var in enumerate(variables):
        var.coupled_variable_group.merge(variables[0].coupled_variable_group)
