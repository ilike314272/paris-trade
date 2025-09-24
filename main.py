import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
from dask import dataframe as dd
import itertools
import dask
from scipy.ndimage import gaussian_filter1d
from copulas.bivariate import Frank
from copulas.univariate import GaussianKDE

# from mean_reversion
from mean_reversion import spread_z, generate_signals, backtest

# -----------------------------
# Helper: build copula models dynamically
# -----------------------------
def fit_marginals(x, y):
    kde_x = GaussianKDE()
    kde_y = GaussianKDE()
    kde_x.fit(x)
    kde_y.fit(y)
    u = kde_x.cdf(x)
    v = kde_y.cdf(y)
    u = np.clip(u, 1e-6, 1 - 1e-6)
    v = np.clip(v, 1e-6, 1 - 1e-6)
    return u, v

def fit_copula(u, v):
    copula = Frank()
    data = np.column_stack([u, v])
    copula.fit(data)
    return copula

def tail_prob(copula, u, v):
    return copula.cumulative_distribution(np.column_stack([u, v]))

def process_pair(pair, returns):
    a, b = pair
    x = returns[a].dropna().values
    y = returns[b].dropna().values
    if len(x) != len(y):
        return None
    try:
        u, v = fit_marginals(x, y)
        copula = fit_copula(u, v)
        probs = tail_prob(copula, u, v)
        return {
            "pair": (a, b),
            "tail_prob": probs,
            "copula": copula,
        }
    except Exception:
        return None


# -----------------------------
# Sidebar Config Controls
# -----------------------------
st.sidebar.header("Strategy Configuration")

mode = st.sidebar.selectbox("Select Strategy Mode", ["copula", "mean_reversion", "both"], index=0)

if "tickers" not in st.session_state:
    st.session_state["tickers"] = ["AAPL", "MSFT", "GOOG"]

ticker_input = st.sidebar.text_input("Add a ticker")
if st.sidebar.button("Add") and ticker_input.strip():
    if ticker_input.strip().upper() not in st.session_state["tickers"]:
        st.session_state["tickers"].append(ticker_input.strip().upper())

st.sidebar.write("### Current Tickers:")
for t in st.session_state["tickers"]:
    col1, col2 = st.sidebar.columns([3, 1])
    col1.write(t)
    if col2.button("❌", key=f"remove_{t}"):
        st.session_state["tickers"].remove(t)
        st.rerun()

threshold_minus = st.sidebar.slider("Threshold -", -1.0, 0.0, -0.2, 0.01)
threshold_plus = st.sidebar.slider("Threshold +", 0.0, 1.0, 0.5, 0.01)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))

go = st.sidebar.button("Go")


# -----------------------------
# Run strategies only when Go is pressed
# -----------------------------
if go:
    st.title("Strategy Suggestions")

    # ---- Fetch fresh data ----
    prices = [yf.download(t, start_date, end_date, auto_adjust=False)["Close"] for t in st.session_state["tickers"]]
    df = pd.concat(prices, axis=1)
    df.columns = st.session_state["tickers"]
    returns = df.pct_change().dropna()

    # Correlation matrix
    ddf = dd.from_pandas(returns, npartitions=min(8, len(returns)))
    m_corr = ddf.corr().compute()

    # Build copula models fresh
    pairs = list(itertools.combinations(st.session_state["tickers"], 2))
    tasks = [dask.delayed(process_pair)(pair, returns) for pair in pairs]
    results = dask.compute(*tasks)
    copula_models = [r for r in results if r is not None]

    # ---- Strategy Selection ----
    strategies = []

    if mode in ["copula", "both"]:
        for model in copula_models:
            a, b = model["pair"]
            tail_prob_mean = np.mean(model["tail_prob"])
            if tail_prob_mean < threshold_minus:
                strategies.append(("all weather", a, b, tail_prob_mean, "copula"))
            elif tail_prob_mean > threshold_plus:
                strategies.append(("paris", a, b, tail_prob_mean, "copula"))

    if mode in ["mean_reversion", "both"]:
        for i in range(len(st.session_state["tickers"])):
            for j in range(i + 1, len(st.session_state["tickers"])):
                a, b = st.session_state["tickers"][i], st.session_state["tickers"][j]
                corr = m_corr.loc[a, b]
                if corr > threshold_plus:
                    strategies.append(("paris", a, b, corr, "mean"))
                elif corr < threshold_minus:
                    strategies.append(("all weather", a, b, corr, "mean"))

    # ---- Display ----
    if strategies:
        st.subheader("Strategy Candidates")
        strat_df = pd.DataFrame(strategies, columns=["Strategy", "Ticker A", "Ticker B", "Metric", "Source"])
        st.dataframe(strat_df)

        for idx, (stype, a, b, metric, source) in enumerate(strategies):
            st.markdown(f"### Strategy #{idx+1}: {stype} ({a}-{b}, {source})")

            x, y = returns[a], returns[b]

            if source == "mean":
                spread, z = spread_z(x, y)
                z.index = spread.index
                signals = generate_signals(z)
                pnl = backtest(signals, x, y)

                fig, axs = plt.subplots(3, 1, figsize=(14, 8))
                axs[0].plot(spread.index, spread, label="Spread")
                axs[0].legend()
                axs[1].plot(z.index, z, label="Z-Score")
                axs[1].axhline(1, color="r", linestyle="--")
                axs[1].axhline(-1, color="g", linestyle="--")
                axs[1].legend()
                axs[2].plot(pnl.index, pnl, label="PnL")
                axs[2].legend()
                st.pyplot(fig)

            elif source == "copula":
                tail_probs = pd.Series(model["tail_prob"])
                time_index = returns.index if len(tail_probs) == len(returns) else pd.date_range(start=start_date, periods=len(tail_probs))
                smoothed = gaussian_filter1d(tail_probs, sigma=2)
                window_size = 10
                moving_avg = pd.Series(smoothed).rolling(window=window_size, center=True).mean()
                x_vals = np.arange(len(smoothed))
                coeffs = np.polyfit(x_vals, smoothed, deg=1)
                line_fit = np.polyval(coeffs, x_vals)

                fig, ax = plt.subplots(figsize=(14, 6))
                ax.plot(time_index, smoothed, label="Smoothed")
                ax.plot(time_index, moving_avg, linestyle="--", label=f"{window_size}-MA")
                ax.plot(time_index, line_fit, linestyle=":", label="Fit")
                ax.legend()
                st.pyplot(fig)

    else:
        st.warning("No valid strategy pairs found.")
else:
    st.info("Adjust config and click **Go** to run strategies.")
