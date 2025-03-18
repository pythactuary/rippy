from rippy import config, distributions
from rippy.config import xp as np
from rippy.variables import ProteusVariable
from rippy.catastrophes import SimEventId, SimEventLossTable
import pandas as pd  # noqa

config.n_sims = 100000

yet_df = pd.read_csv("data/master_ylt_full.csv")
master_yet = SimEventId(
    yet_df["SimNo"].values, yet_df["SimEventId"].values, n_sims=config.n_sims
)
ylt = SimEventLossTable.from_dataframe(pd.read_csv("data/cat_ylt_full.csv"), master_yet)
ylt2 = SimEventLossTable.from_dataframe(
    pd.read_csv("data/cat_ylt2_full.csv"), master_yet
)

ylts = ProteusVariable("class", values={"Property": ylt, "Reinsurance": ylt2})
total_ylt = sum(ylts)
inflation_rate = ProteusVariable(
    dim_name="year",
    values={
        str(2025 + year): distributions.Normal(0.05, 0.02).generate()
        for year in range(10)
    },
)
cumulative_inflation_index = np.cumprod(1 + inflation_rate)

scaled_ylt = total_ylt * cumulative_inflation_index

y: ProteusVariable = np.minimum(np.maximum(scaled_ylt - 1000000, 0), 100000)

z = ProteusVariable(
    dim_name="year",
    values={yr: y[yr].aggregate() for yr in y.values.keys()},
)

z.show_cdf()
