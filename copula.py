import itertools
import numpy as np
import dask
from copulas.bivariate import Frank
from copulas.univariate import GaussianKDE

from data_init import *

# Assume df is already created with yfinance adjusted close data
returns = df.pct_change().dropna()

# === Fit KDE marginals and transform to uniform [0,1] ===
def fit_marginals(x, y):
    kde_x = GaussianKDE()
    kde_y = GaussianKDE()
    kde_x.fit(x)
    kde_y.fit(y)
    u = kde_x.cdf(x)
    v = kde_y.cdf(y)
    # Ensure values are clipped to avoid copula domain errors
    u = np.clip(u, 1e-6, 1 - 1e-6)
    v = np.clip(v, 1e-6, 1 - 1e-6)
    return u, v

# === Fit copula ===
def fit_copula(u, v):
    copula = Frank()
    data = np.column_stack([u, v])
    copula.fit(data)
    return copula

# === Get tail probabilities from copula ===
def tail_prob(copula, u, v):
    return copula.cumulative_distribution(np.column_stack([u, v]))

# === Process each ticker pair ===
def process_pair(pair):
    a, b = pair
    x = returns[a].dropna().values
    y = returns[b].dropna().values
    if len(x) != len(y):
        print(f"Length mismatch: {a}-{b}")
        return None
    try:
        u, v = fit_marginals(x, y)
        copula = fit_copula(u, v)
        probs = tail_prob(copula, u, v)
        return {
            'type': 'copula',
            'pair': (a, b),
            'tail_prob': probs,
            'copula': copula
        }
    except Exception as e:
        print(f"Error processing pair {a}-{b}: {e}")
        return None

# === Build and run Dask graph ===
pairs = list(itertools.combinations(config['tickers'], 2))
tasks = [dask.delayed(process_pair)(pair) for pair in pairs]
results = dask.compute(*tasks)
copula_models = [r for r in results if r is not None]
