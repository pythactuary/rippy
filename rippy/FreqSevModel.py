from .FreqSevSims import FreqSevSims
import rippy.Distributions as Distributions
from .config import config, xp as np


def _get_sims_of_events(n_events: np.ndarray):
    cumulative = n_events.cumsum()
    total_events = cumulative[-1]
    event_no = np.arange(total_events)
    return cumulative.searchsorted(event_no + 1)


class FrequencySeverityModel:
    def __init__(
        self,
        freq_dist: Distributions.Distribution,
        sev_dist: Distributions.Distribution,
    ):
        self.freq_dist = freq_dist
        self.sev_dist = sev_dist

    def generate(self, n_sims=None, rng: np.random.Generator = config.rng):
        if n_sims is None:
            n_sims = config.n_sims
        n_events = self.freq_dist.generate(n_sims, rng)
        total_events = n_events.sum()
        sev = self.sev_dist.generate(int(total_events), rng)
        return FreqSevSims(_get_sims_of_events(n_events), sev, n_sims)
