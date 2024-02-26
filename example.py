from rippy import Distributions, config, XoLTower
from rippy.FrequencySeverity import FrequencySeverityModel
import numpy as np

config.n_sims = 1000000

elt = {"EventId":np.array([1234,1235,1236,1237,1238,1239]),
                          "ExpValue":np.array([100000000,200000000,300000000,400000000,500000000,600000000]),
"PerspValue":np.array([10000000,20000000,30000000,40000000,50000000,60000000]),
"StdDevI":np.array([1000000,2000000,3000000,4000000,5000000,6000000]),
"StdDevC":np.array([1000000,2000000,3000000,4000000,5000000,6000000]),
}

mean_dr = elt["PerspValue"]/elt["ExpValue"]
sd_total = elt["StdDevI"]+elt["StdDevC"]
cov_loss = sd_total / elt["PerspValue"]

alpha = (1-mean_dr)/cov_loss**2 - mean_dr
beta = alpha/mean_dr*(1-mean_dr)

rate = {"EventId":np.array([1239,1235,1236,1237,1238,1234]),
        "rate":np.array([0.01,0.02,0.03,0.04,0.05,0.06])}

rate_for_elt = rate["rate"][elt["EventId"].searchsorted(rate["EventId"])]

n_events = Distributions.Poisson(sum(rate_for_elt)).generate()
cumulative_normalised_rate= rate_for_elt.cumsum()/sum(rate_for_elt)
simulated_row = cumulative_normalised_rate.searchsorted(Distributions.Uniform().generate(n_sims = sum(n_events)))

simulated_alpha = alpha[simulated_row]
simulated_beta = beta[simulated_row]
simulated_exposure = elt["ExpValue"][simulated_row]

severity_uniform = Distributions.Uniform().generate(n_sims = sum(n_events))

from scipy.special import betaincinv

loss = betaincinv(simulated_alpha,simulated_beta,severity_uniform)*simulated_exposure


sev_dist = Distributions.GPD(0.33, 100000, 1000000)
freq_dist = Distributions.Poisson(5)

losses_pre_cap = FrequencySeverityModel(freq_dist, sev_dist).generate()
policy_limit = 5000000
losses_post_cap = np.minimum(
    losses_pre_cap, policy_limit
)  # you can apply standard numpy ufuncs to the losses
losses_with_LAE = (
    losses_post_cap * 1.05
)  # you can apply standard numerical operations to the losses
stochastic_inflation = Distributions.Normal(0.05, 0.02).generate()

gross_losses = losses_with_LAE * (
    1 + stochastic_inflation
)  # you can multiply frequency severity losses with other standard simulations


prog = XoLTower(
    limit=[1000000, 1000000, 1000000, 1000000, 1000000],
    excess=[1000000, 2000000, 3000000, 4000000, 5000000],
    aggregate_limit=[3000000, 2000000, 1000000, 1000000, 1000000],
    aggregate_deductible=[0, 0, 0, 0, 0],
    premium=[5, 4, 3, 2, 1],
    reinstatement_cost=[[1, 0.5], [1], None, None, None],
)

recoveries = prog.apply(gross_losses)

prog.print_summary()
