import matplotlib.pyplot as plt
from copula import *
from mean_reversion import *

# === Strategy Selection ===
mode = config.get('mode', 'copula')
strategies = []

print("\nStrategy Suggestions:")

if mode in ['copula', 'both']:
    print("\nCopula-Based Strategies:")
    for model in copula_models:
        a, b = model['pair']
        tail_prob_mean = model['tail_prob'].mean()
        if tail_prob_mean < config['threshold-']:
            strategies.append(('all weather', a, b, tail_prob_mean, 'copula'))
        elif tail_prob_mean > config['threshold+']:
            strategies.append(('paris', a, b, tail_prob_mean, 'copula'))

if mode in ['mean_reversion', 'both']:
    print("\nMean-Reversion Strategies (Kalman):")
    for i in range(len(config['tickers'])):
        for j in range(i + 1, len(config['tickers'])):
            a, b = config['tickers'][i], config['tickers'][j]
            corr = m_corr.loc[a, b]
            if corr > config['threshold+']:
                strategies.append(('paris', a, b, corr, 'mean'))
            elif corr < config['threshold-']:
                strategies.append(('all weather', a, b, corr, 'mean'))

# === Print All Strategy Candidates ===
for strat in strategies:
    print(f"Strategy: {strat[0]:<12} | Pair: {strat[1]} - {strat[2]} | Metric: {strat[3]:.4f} ({strat[4]})")

# === Visualize All Strategies ===
if strategies:
    for idx, (stype, a, b, metric, source) in enumerate(strategies):
        print(f"\nBacktesting Strategy #{idx + 1} on {a}-{b} ({stype}, source={source})")

        x, y = returns[a], returns[b]
        spread, z = spread_z(x, y)
        z.index = spread.index

        signals = generate_signals(z)
        pnl = backtest(signals, x, y)

        print(f"  -> Lengths: Spread={len(spread)}, Z={len(z)}, Signals={len(signals)}, PnL={len(pnl)}")

        # === Plot ===
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
else:
    print("No valid strategy pairs found.")