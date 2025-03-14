from rippy import Distributions, config
from rippy.FrequencySeverity import FrequencySeverityModel
from rippy import copulas
from rippy.variables import ProteusVariable

config.n_sims = 100000

lobs = ["Motor", "Property", "Liability", "Marine", "Aviation"]

individual_large_losses_by_lob = ProteusVariable(
    dim_name="class",
    values={
        name: FrequencySeverityModel(
            Distributions.Poisson(mean=5),
            Distributions.GPD(shape=0.33, scale=100000, loc=1000000),
        ).generate()
        for name in lobs
    },
)

attritional_losses_by_lob = ProteusVariable(
    dim_name="class",
    values={lob: Distributions.Gamma(alpha=2, theta=1000).generate() for lob in lobs},
)

losses_with_LAE = individual_large_losses_by_lob * 1.05

# create the aggregate losses by class
aggregate_large_losses_by_class = ProteusVariable(
    "class", {name: losses_with_LAE[name].aggregate() for name in lobs}
)
# correlate the attritional and large losses
for lob in lobs:
    copulas.GumbelCopula(1.2, 2).apply(
        [aggregate_large_losses_by_class[lob], attritional_losses_by_lob[lob]]
    )
# calculate the total losses
total_losses_by_lob = aggregate_large_losses_by_class + attritional_losses_by_lob

# apply a copula to the total losses by lob
copulas.GumbelCopula(1.2, 5).apply(total_losses_by_lob)

# create the total losses
total_losses = total_losses_by_lob.sum()

total_losses.show_histogram()
