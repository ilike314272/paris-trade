import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d

# --- Import your project modules ---
from data_init import config as base_config, m_corr, df
from copula import copula_models, returns
from mean_reversion import spread_z, generate_signals, backtest

# -----------------------------
# Sidebar Config Controls
# -----------------------------
st.sidebar.header("Strategy Configuration")

# Mode selection
mode = st.sidebar.selectbox("Select Strategy Mode", ["copula", "mean_reversion", "both"], index=0)

# --- Dynamic ticker management ---
if "tickers" not in st.session_state:
    st.session_state["tickers"] = base_config["tickers"].copy()

ticker_input = st.sidebar.text_input("Add a ticker")
add_ticker = st.sidebar.button("Add")

if add_ticker and ticker_input.strip():
    if ticker_input.strip() not in st.session_state["tickers"]:
        st.session_state["tickers"].append(ticker_input.strip().upper())

st.sidebar.write("### Current Tickers:")
for t in st.session_state["tickers"]:
    col1, col2 = st.sidebar.columns([3, 1])
    col1.write(t)
    if col2.button("❌", key=f"remove_{t}"):
        st.session_state["tickers"].remove(t)
        st.rerun()

# Threshold sliders
threshold_minus = st.sidebar.slider("Threshold -", -1.0, 0.0, float(base_config.get("threshold-", -0.2)), 0.01)
threshold_plus = st.sidebar.slider("Threshold +", 0.0, 1.0, float(base_config.get("threshold+", 0.5)), 0.01)

# Date range for copula plots
start_date = st.sidebar.date_input("Start Date", pd.to_datetime(base_config["start"]))

# "Go" button
go = st.sidebar.button("Go")


# -----------------------------
# Run strategies only when Go is pressed
# -----------------------------
if go:
    # Updated config
    config = base_config.copy()
    config["mode"] = mode
    config["tickers"] = st.session_state["tickers"]
    config["threshold-"] = threshold_minus
    config["threshold+"] = threshold_plus
    config["start"] = start_date.strftime("%Y-%m-%d")

    # -----------------------------
    # Strategy Selection
    # -----------------------------
    st.title("Strategy Suggestions")
    strategies = []

    if mode in ["copula", "both"]:
        for model in copula_models:
            a, b = model["pair"]
            tail_prob_mean = np.mean(model["tail_prob"])
            if tail_prob_mean < config["threshold-"]:
                strategies.append(("all weather", a, b, tail_prob_mean, "copula"))
            elif tail_prob_mean > config["threshold+"]:
                strategies.append(("paris", a, b, tail_prob_mean, "copula"))

    if mode in ["mean_reversion", "both"]:
        for i in range(len(config["tickers"])):
            for j in range(i + 1, len(config["tickers"])):
                a, b = config["tickers"][i], config["tickers"][j]
                corr = m_corr.loc[a, b]
                if corr > config["threshold+"]:
                    strategies.append(("paris", a, b, corr, "mean"))
                elif corr < config["threshold-"]:
                    strategies.append(("all weather", a, b, corr, "mean"))

    # -----------------------------
    # Display Strategies
    # -----------------------------
    if strategies:
        st.subheader("Strategy Candidates")
        strat_df = pd.DataFrame(strategies, columns=["Strategy", "Ticker A", "Ticker B", "Metric", "Source"])
        st.dataframe(strat_df)

        # Visualization for each strategy
        for idx, (stype, a, b, metric, source) in enumerate(strategies):
            st.markdown(f"### Backtesting Strategy #{idx+1}: {stype} ({a}-{b}, source={source})")
            x, y = returns[a], returns[b]

            if source == "mean":
                spread, z = spread_z(x, y)
                z.index = spread.index
                signals = generate_signals(z)
                pnl = backtest(signals, x, y)

                fig, axs = plt.subplots(3, 1, figsize=(14, 8))

                axs[0].plot(spread.index, spread, label="Spread")
                axs[0].set_title(f"Spread: {b} - {a}")
                axs[0].legend()

                axs[1].plot(z.index, z, label="Z-Score")
                axs[1].axhline(1, color="r", linestyle="--")
                axs[1].axhline(-1, color="g", linestyle="--")
                axs[1].axhline(0, color="black", linestyle="-")
                axs[1].set_title("Z-Score")
                axs[1].legend()

                axs[2].plot(pnl.index, pnl, label="Cumulative Return")
                axs[2].set_title("Backtest Performance")
                axs[2].legend()

                plt.tight_layout()
                st.pyplot(fig)

            elif source == "copula":
                model = next((m for m in copula_models if m["pair"] == (a, b) or m["pair"] == (b, a)), None)
                if model:
                    tail_probs = pd.Series(model["tail_prob"])
                    time_index = returns.index if len(tail_probs) == len(returns) else pd.date_range(start=config["start"], periods=len(tail_probs))
                    smoothed = gaussian_filter1d(tail_probs, sigma=2)
                    window_size = 10
                    moving_avg = pd.Series(smoothed).rolling(window=window_size, center=True).mean()
                    x_vals = np.arange(len(smoothed))
                    coeffs = np.polyfit(x_vals, smoothed, deg=1)
                    line_fit = np.polyval(coeffs, x_vals)

                    fig, ax = plt.subplots(figsize=(14, 6))
                    ax.plot(time_index, smoothed, color="blue", label="Smoothed")
                    ax.plot(time_index, moving_avg, color="orange", linestyle="--", label=f"{window_size}-Point MA")
                    ax.plot(time_index, line_fit, color="red", linestyle=":", label="Best Fit Line")
                    ax.set_title(f"Tail Probability Diagnostics: {a} vs {b}")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Probability")
                    ax.legend()
                    st.pyplot(fig)
    else:
        st.warning("No valid strategy pairs found.")
else:
    st.info("Adjust config and click **Go** to run strategies.")
