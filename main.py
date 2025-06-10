import yfinance as yf
import pandas as pd
import numpy as np
import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import coint
from itertools import combinations
from pykalman import KalmanFilter
from dask import delayed, compute


# ==== CONFIGURATION ====
def run_analysis(
        tickers=['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META'],
        start='2022-01-01',
        end=None,
        coint_threshold=0.05,
        plot_top_n=3
):
    # 1. Download historical data
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    data = data.dropna()

    # 2. Compute all possible pairs
    pairs = list(combinations(data.columns, 2))

    # 3. Define computation per pair
    @delayed
    def analyze_pair(ticker1, ticker2):
        series1 = data[ticker1]
        series2 = data[ticker2]

        # Correlation
        corr = series1.corr(series2)

        # Cointegration
        score, pval, _ = coint(series1, series2)

        # Kalman Filter estimation
        kf = KalmanFilter(
            transition_matrices=np.eye(2),
            observation_matrices=np.vstack([series1.values, np.ones(len(series1))]).T[:, np.newaxis, :],
            initial_state_mean=np.zeros(2),
            em_vars=['transition_covariance', 'observation_covariance']
        )
        try:
            kf = kf.em(series2.values, n_iter=5)
            state_means, _ = kf.filter(series2.values)
            spread = series2 - (state_means[:, 0] * series1 + state_means[:, 1])
            spread_std = np.std(spread)
        except Exception as e:
            spread_std = np.nan

        return {
            'pair': (ticker1, ticker2),
            'correlation': corr,
            'cointegration_p': pval,
            'spread_std': spread_std
        }

    # 4. Run computations with Dask
    results = compute(*[analyze_pair(t1, t2) for t1, t2 in pairs])
    result_df = pd.DataFrame(results)

    # 5. Sort and display summary
    result_df.sort_values('cointegration_p', inplace=True)
    print("Top pairs by cointegration p-value:")
    print(result_df[['pair', 'correlation', 'cointegration_p', 'spread_std']].head(plot_top_n))

    # 6. Plot correlation vs cointegration heatmap
    pivot_corr = pd.DataFrame(index=data.columns, columns=data.columns, dtype=float)
    pivot_pval = pd.DataFrame(index=data.columns, columns=data.columns, dtype=float)

    for r in results:
        t1, t2 = r['pair']
        pivot_corr.loc[t1, t2] = r['correlation']
        pivot_corr.loc[t2, t1] = r['correlation']
        pivot_pval.loc[t1, t2] = r['cointegration_p']
        pivot_pval.loc[t2, t1] = r['cointegration_p']

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(pivot_corr.astype(float), ax=axs[0], annot=True, cmap='coolwarm')
    axs[0].set_title('Pairwise Correlation')
    sns.heatmap(pivot_pval.astype(float), ax=axs[1], annot=True, cmap='viridis_r')
    axs[1].set_title('Cointegration p-values')
    plt.tight_layout()
    plt.show()

    # 7. Visualize spread for top pairs
    top_pairs = result_df.nsmallest(plot_top_n, 'cointegration_p')['pair']
    for t1, t2 in top_pairs:
        y1 = data[t1]
        y2 = data[t2]

        kf = KalmanFilter(
            transition_matrices=np.eye(2),
            observation_matrices=np.vstack([y1.values, np.ones(len(y1))]).T[:, np.newaxis, :],
            initial_state_mean=np.zeros(2),
            em_vars=['transition_covariance', 'observation_covariance']
        )
        kf = kf.em(y2.values, n_iter=5)
        state_means, _ = kf.filter(y2.values)
        spread = y2 - (state_means[:, 0] * y1 + state_means[:, 1])

        plt.figure(figsize=(12, 4))
        plt.plot(spread, label=f'Spread {t2} - {t1}')
        plt.axhline(spread.mean(), color='r', linestyle='--')
        plt.fill_between(range(len(spread)), spread.mean() - spread.std(), spread.mean() + spread.std(),
                         color='gray', alpha=0.2)
        plt.title(f'Kalman Spread for {t1}-{t2}')
        plt.legend()
        plt.show()


# === Run with defaults or override ===
if __name__ == "__main__":
    run_analysis()
