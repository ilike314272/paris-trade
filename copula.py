from copulas.bivariate import Clayton, Frank, Gumbel
from copulas.univariate import GaussianKDE
import numpy as np
import itertools

from data_init import *

dd_pct_change = dd.from_pandas(df.pct_change().dropna(), npartitions=int(config['nparitions']))

def fit_copula(x, y):
    copula = Frank()
    data = np.column_stack(x, y)
    copula.fit(data)
    return copula

def tail_prob(copula, x, y):
    # Calculate empirical CDFs
    u = pd.Series(x).rank(pct=True).values
    v = pd.Series(y).rank(pct=True).values
    tail_probs = copula.cumulative_distribution(np.column_stack([u, v]))
    return tail_probs

def set_pair(pair):
    a, b = pair
    x = returns[a].values
    y = returns[b].values

    if len(x) != len(y): return None

    try:

    except Exception as e:
        return None
