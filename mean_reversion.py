import numpy as np
from pykalman import KalmanFilter

from data_init import *

# Strategies to implement based on correlations
strat = []
for i in range(len(config['tickers'])):
    for j in range(i+1, len(config['tickers'])):
        corr = m_corr.iloc[i,j]
        if corr > config['threshold+']:
            strat.append(('paris', config['tickers'][i], config['tickers'][j], corr))
        elif corr < config['threshold-']:
            strat.append(('all weather', config['tickers'][i], config['tickers'][j], corr))

# Hedge ratio with Kalman Filter
def hedge_ratio(x, y):
    delta = 1e-5
    trans_cov = delta / (1-delta) * np.eye(2)
    obs_mat = np.expand_dims(np.vstack([x, np.ones(len(x))]).T, axis=1)

    kf = KalmanFilter(
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        initial_state_mean=[0,0],
        observation_covariance=1.0,
        transition_covariance=trans_cov,
    )

    state_means, _ = kf.filter(y)
    return state_means[:, 0]

# Spread and Z_Score
def spread_z(x, y):
    spread = y - hedge_ratio(x, y) * x
    return spread, (spread - spread.mean()) / spread.std()

# Signal generation
def generate_signals(z_score, entry=1.0, exit=0.2):
    signals = pd.Series(index=z_score.index, data=0)
    signals[z_score > entry] = -1
    signals[z_score < -entry] = 1
    signals[abs(z_score) < exit] = 0
    return signals.ffill()

# Backtest against historical data
def backtest(signals, x, y):
    y_ret = y.pct_change().shift(-1)
    x_ret = x.pct_change().shift(-1)
    spread_ret = signals * (y_ret - hedge_ratio(x,y) * x_ret)
    return spread_ret.cumsum()