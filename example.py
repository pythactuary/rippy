from rippy import Distributions, config, XoLTower
from rippy.FrequencySeverity import FrequencySeverityModel
import numpy as np

config.n_sims = 1000000

sev_dist = Distributions.GPD(shape=0.33, scale=100000, loc=1000000)
freq_dist = Distributions.Poisson(mean=5)

losses_pre_cap = FrequencySeverityModel(freq_dist, sev_dist).generate()
policy_limit = 5000000
# you can apply standard numpy ufuncs to the losses
losses_post_cap = np.minimum(losses_pre_cap, policy_limit)

# you can apply standard numerical operations to the losses
losses_with_LAE = losses_post_cap * 1.05
stochastic_inflation = Distributions.Normal(0.05, 0.02).generate()

# you can multiply frequency severity losses with other standard simulations
gross_losses = losses_with_LAE * (1 + stochastic_inflation)

prog = XoLTower(
    limit=[1000000, 1000000, 1000000, 1000000, 1000000],
    excess=[1000000, 2000000, 3000000, 4000000, 5000000],
    aggregate_limit=[3000000, 2000000, 1000000, 1000000, 1000000],
    premium=[5000, 4000, 3000, 2000, 1000],
    reinstatement_cost=[[1, 0.5], [1], None, None, None],
)

recoveries = prog.apply(gross_losses)

prog.print_summary()
