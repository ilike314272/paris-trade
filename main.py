import matplotlib.pyplot as plt
from copula import *
from mean_reversion import *

# Strategy Selection
mode = config.get('mode', 'copula')
strategies = []

print("\nStrategy Suggestions:")

if mode in ['copula', 'both']:
    for model in copula_models:
        a, b = model['pair']
        tail_prob_mean = model['tail_prob'].mean()
        if tail_prob_mean < config['threshold-']:
            strategies.append(('all weather', a, b, tail_prob_mean, 'copula'))
        elif tail_prob_mean > config['threshold+']:
            strategies.append(('paris', a, b, tail_prob_mean, 'copula'))
        else:
            print('Neither:', tail_prob_mean)

if mode in ['mean_reversion', 'both']:
    for i in range(len(config['tickers'])):
        for j in range(i + 1, len(config['tickers'])):
            a, b = config['tickers'][i], config['tickers'][j]
            corr = m_corr.loc[a, b]
            if corr > config['threshold+']:
                strategies.append(('paris', a, b, corr, 'mean'))
            elif corr < config['threshold-']:
                strategies.append(('all weather', a, b, corr, 'mean'))

import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Print All Strategy Candidates
for strat in strategies:
    print(f"Strategy: {strat[0]:<12} | Pair: {strat[1]} - {strat[2]} | Metric: {strat[3]:.4f} ({strat[4]})")

# Visualize All Strategies
if strategies:
    for idx, (stype, a, b, metric, source) in enumerate(strategies):
        print(f"\nBacktesting Strategy #{idx + 1} on {a}-{b} ({stype}, source={source})")

        x, y = returns[a], returns[b]

        if source == 'mean':
            spread, z = spread_z(x, y)
            z.index = spread.index
            signals = generate_signals(z)
            pnl = backtest(signals, x, y)
            print(f"  -> Mean Reversion Metrics:")
            print(f"     Spread Length: {len(spread)}")
            print(f"     Z-Score Length: {len(z)}")
            print(f"     Signals Length: {len(signals)}")
            print(f"     PnL Length: {len(pnl)}")
            plt.figure(figsize=(14, 8))
            plt.subplot(3, 1, 1)
            plt.plot(spread.index, spread, label='Spread')
            plt.title(f'Spread: {b} - {a} | Source: {source}')
            plt.legend()
            plt.subplot(3, 1, 2)
            plt.plot(z.index, z, label='Z-Score')
            plt.axhline(1, color='r', linestyle='--')
            plt.axhline(-1, color='g', linestyle='--')
            plt.axhline(0, color='black', linestyle='-')
            plt.title('Z-Score')
            plt.legend()
            plt.subplot(3, 1, 3)
            plt.plot(pnl.index, pnl, label='Cumulative Return')
            plt.title('Backtest Performance')
            plt.legend()
            plt.tight_layout()
            plt.show()
        elif source == 'copula':
            model = next((m for m in copula_models if m['pair'] == (a, b) or m['pair'] == (b, a)), None)
            if model is None:
                print(f"No copula model found for pair {a}-{b}")
                continue
            # Retrieve copula and tail probabilities
            copula = model['copula']
            tail_probs = pd.Series(model['tail_prob'])
            time_index = returns.index if len(tail_probs) == len(returns) else pd.date_range(start=config['start'], periods=len(tail_probs))
            # Begin plotting
            fig, axs = plt.subplots(1, 1, figsize=(28, 10))
            fig.suptitle(f'Tail Probability Diagnostics: {a} vs {b}', fontsize=16)
            # Smoothed Tail Probabilities
            smoothed = gaussian_filter1d(tail_probs, sigma=2)
            window_size = 10  # or any period you prefer
            moving_avg = pd.Series(smoothed).rolling(window=window_size, center=True).mean()
            x_vals = np.arange(len(smoothed))
            coeffs = np.polyfit(x_vals, smoothed, deg=1)
            line_fit = np.polyval(coeffs, x_vals)
            axs.plot(time_index, smoothed, color='blue', label='Smoothed')
            axs.plot(time_index, moving_avg, color='orange', linestyle='--', label=f'{window_size}-Point MA')
            axs.plot(time_index, line_fit, color='red', linestyle=':', label='Best Fit Line')
            axs.set_xlabel('Date')
            axs.set_title('Tail Probabilities')
            axs.set_ylabel('Probability')
            axs.set_xlabel('Date')
            axs.legend()
        else:
            print(f"Unknown source type: {source}")
else:
    print("No valid strategy pairs found.")
