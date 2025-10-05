import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from arch import arch_model
from scipy import stats
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Kalman Filter Implementation
class KalmanFilter:
    def __init__(self, transition_matrices, observation_matrices, 
                 initial_state_mean, initial_state_covariance,
                 transition_covariance, observation_covariance):
        self.transition_matrices = transition_matrices
        self.observation_matrices = observation_matrices
        self.transition_covariance = transition_covariance
        self.observation_covariance = observation_covariance
        self.state_mean = initial_state_mean
        self.state_covariance = initial_state_covariance
        
    def filter(self, observations):
        n = len(observations)
        state_means = np.zeros((n, len(self.state_mean)))
        state_covariances = np.zeros((n, len(self.state_mean), len(self.state_mean)))
        
        for t in range(n):
            # Predict
            state_mean_pred = self.transition_matrices @ self.state_mean
            state_cov_pred = (self.transition_matrices @ self.state_covariance @ 
                             self.transition_matrices.T + self.transition_covariance)
            
            # Update
            residual = observations[t] - self.observation_matrices @ state_mean_pred
            residual_cov = (self.observation_matrices @ state_cov_pred @ 
                           self.observation_matrices.T + self.observation_covariance)
            
            kalman_gain = state_cov_pred @ self.observation_matrices.T @ np.linalg.inv(residual_cov)
            
            self.state_mean = state_mean_pred + kalman_gain @ residual
            self.state_covariance = state_cov_pred - kalman_gain @ self.observation_matrices @ state_cov_pred
            
            state_means[t] = self.state_mean
            state_covariances[t] = self.state_covariance
            
        return state_means, state_covariances

# Statistical Tests
def adf_test(series, threshold=0.05):
    # Ensure series is 1D
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    series_clean = series.dropna()
    
    # Convert to 1D numpy array if needed
    if hasattr(series_clean, 'values'):
        series_clean = series_clean.values.flatten()
    
    result = adfuller(series_clean)
    return {
        'statistic': result[0],
        'p_value': result[1],
        'passed': result[1] < threshold
    }

def kpss_test(series, threshold=0.05):
    # Ensure series is 1D
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    series_clean = series.dropna()
    
    # Convert to 1D numpy array if needed
    if hasattr(series_clean, 'values'):
        series_clean = series_clean.values.flatten()
    
    result = kpss(series_clean, regression='c')
    return {
        'statistic': result[0],
        'p_value': result[1],
        'passed': result[1] > threshold
    }

def engle_granger_test(y, x, threshold=0.05):
    # Ensure 1D series
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    
    result = coint(y, x)
    return {
        'statistic': result[0],
        'p_value': result[1],
        'passed': result[1] < threshold
    }

def johansen_test(data, threshold=0.05):
    result = coint_johansen(data, det_order=0, k_ar_diff=1)
    trace_stat = result.lr1[0]
    critical_value = result.cvt[0, 1]  # 5% critical value
    return {
        'trace_statistic': trace_stat,
        'critical_value': critical_value,
        'passed': trace_stat > critical_value
    }

def variance_ratio_test(series, lag=2, threshold=0.05):
    # Ensure series is 1D
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    
    series = series.dropna()
    
    # Convert to numpy array
    if hasattr(series, 'values'):
        series = series.values.flatten()
    
    n = len(series)
    mu = np.mean(series)
    
    var_1 = np.var(series - mu, ddof=1)
    var_lag = np.var(series[lag:] - series[:-lag], ddof=1) / lag
    
    vr = var_lag / var_1
    
    # Asymptotic standard error
    se = np.sqrt(2 * (2 * lag - 1) * (lag - 1) / (3 * lag * n))
    z_stat = (vr - 1) / se
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    
    return {
        'variance_ratio': vr,
        'z_statistic': z_stat,
        'p_value': p_value,
        'passed': abs(vr - 1) < 0.5 and p_value < threshold
    }

def run_all_tests(y, x, thresholds):
    results = {}
    
    # Ensure we're working with 1D arrays/series
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    
    # ADF test on spread
    spread = y - x
    
    # Ensure spread is 1D
    if isinstance(spread, pd.DataFrame):
        spread = spread.iloc[:, 0]
    
    results['adf'] = adf_test(spread, thresholds['adf'])
    results['kpss'] = kpss_test(spread, thresholds['kpss'])
    results['engle_granger'] = engle_granger_test(y, x, thresholds['engle_granger'])
    
    data = pd.DataFrame({'y': y, 'x': x}).dropna()
    results['johansen'] = johansen_test(data, thresholds['johansen'])
    results['variance_ratio'] = variance_ratio_test(spread, threshold=thresholds['variance_ratio'])
    
    return results

def apply_kalman_filter(y, x):
    # Simple linear regression to initialize
    y_clean = y.dropna()
    x_clean = x.dropna()
    common_idx = y_clean.index.intersection(x_clean.index)
    y_clean = y_clean.loc[common_idx]
    x_clean = x_clean.loc[common_idx]
    
    # Create observations matrix
    observations = np.column_stack([y_clean.values, x_clean.values])
    
    # Initialize Kalman Filter for hedge ratio estimation
    delta = 1e-5
    transition_matrices = np.eye(2)
    observation_matrices = np.array([[1, 0], [0, 1]])
    initial_state_mean = np.array([0, 1])
    initial_state_covariance = np.eye(2)
    transition_covariance = delta * np.eye(2)
    observation_covariance = np.eye(2)
    
    kf = KalmanFilter(
        transition_matrices, observation_matrices,
        initial_state_mean, initial_state_covariance,
        transition_covariance, observation_covariance
    )
    
    state_means, _ = kf.filter(observations)
    
    # Calculate spread using Kalman-estimated hedge ratio
    hedge_ratios = state_means[:, 1]
    spread = y_clean.values - hedge_ratios * x_clean.values
    z_score = (spread - np.mean(spread)) / np.std(spread)
    
    return pd.Series(spread, index=common_idx), pd.Series(z_score, index=common_idx), pd.Series(hedge_ratios, index=common_idx)

def fit_copula(u, v):
    # Gaussian copula
    from scipy.stats import kendalltau
    tau = kendalltau(u, v)[0]
    rho = np.sin(tau * np.pi / 2)
    
    # Transform to uniform
    u_rank = stats.rankdata(u) / (len(u) + 1)
    v_rank = stats.rankdata(v) / (len(v) + 1)
    
    # Transform to normal
    u_norm = norm.ppf(u_rank)
    v_norm = norm.ppf(v_rank)
    
    # Tail probabilities
    threshold = 0.05
    lower_tail = np.mean((u_rank < threshold) & (v_rank < threshold))
    upper_tail = np.mean((u_rank > 1 - threshold) & (v_rank > 1 - threshold))
    
    return {
        'correlation': rho,
        'lower_tail_prob': lower_tail,
        'upper_tail_prob': upper_tail
    }

def fit_garch_and_simulate(returns, n_simulations=1000, n_periods=30):
    try:
        # Ensure returns is 1D numpy array
        if returns is None:
            return None, {'var_95': 0, 'var_99': 0, 'volatility': 0, 'error': 'No data provided'}
            
        if isinstance(returns, pd.DataFrame):
            returns = returns.iloc[:, 0].values
        elif isinstance(returns, pd.Series):
            returns = returns.values
        
        if returns is None or len(returns) == 0:
            return None, {'var_95': 0, 'var_99': 0, 'volatility': 0, 'error': 'Empty data'}
        
        returns = np.array(returns).flatten()
        
        # Remove any NaN or infinite values
        returns = returns[np.isfinite(returns)]
        
        if len(returns) < 50:
            return None, {'var_95': 0, 'var_99': 0, 'volatility': 0, 'error': f'Insufficient data: {len(returns)} points'}
        
        # Check for zero variance
        if np.std(returns) < 1e-10:
            return None, {'var_95': 0, 'var_99': 0, 'volatility': 0, 'error': 'Zero variance in returns'}
        
        # Scale returns to avoid numerical issues (convert to percentage if needed)
        if np.abs(returns).mean() < 0.01:
            returns = returns * 100
        
        # Fit GARCH(1,1) model with better parameters
        model = arch_model(returns, vol='Garch', p=1, q=1, mean='Zero', rescale=True)
        fitted = model.fit(disp='off', show_warning=False)
        
        # Get conditional volatility
        cond_vol = fitted.conditional_volatility
        if cond_vol is None or len(cond_vol) == 0:
            return None, {'var_95': 0, 'var_99': 0, 'volatility': 0, 'error': 'Could not estimate volatility'}
        
        last_vol = cond_vol.iloc[-1] if isinstance(cond_vol, pd.Series) else cond_vol[-1]
        
        # Simulate future paths using fitted parameters
        try:
            forecasts = fitted.forecast(horizon=n_periods, simulations=n_simulations, method='simulation')
            
            # Extract simulated values with careful handling
            if hasattr(forecasts, 'simulations') and forecasts.simulations is not None:
                sim_dict = forecasts.simulations
                
                # Get the variance forecasts
                if 'variance' in sim_dict:
                    variance_sims = sim_dict['variance'].values
                    if variance_sims is not None and variance_sims.size > 0:
                        # Generate returns from variance forecasts
                        # variance_sims shape should be (n_simulations, horizon) or (horizon, n_simulations)
                        if variance_sims.ndim == 3:
                            variance_sims = variance_sims[0]  # Remove first dimension if present
                        
                        # Ensure correct shape: (horizon, n_simulations)
                        if variance_sims.shape[0] != n_periods:
                            variance_sims = variance_sims.T
                        
                        # Generate returns from variance (std = sqrt(variance))
                        std_sims = np.sqrt(np.abs(variance_sims))
                        simulated_returns = np.random.normal(0, 1, variance_sims.shape) * std_sims
                    else:
                        raise ValueError("Empty variance simulations")
                else:
                    raise ValueError("No variance key in simulations")
            else:
                raise ValueError("No simulations attribute")
                
        except Exception as sim_error:
            # Fallback: manual simulation using fitted parameters
            alpha = fitted.params.get('alpha[1]', 0.1)
            beta = fitted.params.get('beta[1]', 0.85)
            omega = fitted.params.get('omega', 0.01)
            
            simulated_returns = np.zeros((n_periods, n_simulations))
            h = np.ones(n_simulations) * last_vol ** 2  # Initialize variance
            
            for t in range(n_periods):
                # Generate standardized residuals
                z = np.random.standard_normal(n_simulations)
                
                # Calculate returns
                simulated_returns[t, :] = np.sqrt(h) * z
                
                # Update variance for next period using GARCH(1,1) equation
                h = omega + alpha * simulated_returns[t, :] ** 2 + beta * h
        
        # Calculate VaR on final period
        final_returns = simulated_returns[-1, :]
        var_95 = np.percentile(final_returns, 5)
        var_99 = np.percentile(final_returns, 1)
        
        return simulated_returns, {
            'var_95': var_95,
            'var_99': var_99,
            'volatility': last_vol
        }
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return None, {'var_95': 0, 'var_99': 0, 'volatility': 0, 'error': error_msg}

# Streamlit App
st.set_page_config(layout="wide", page_title="Pairs Trading Analysis")

st.title("Pairs Trading Statistical Analysis")

# Sidebar inputs
st.sidebar.header("Configuration")

# Tickers input
tickers_input = st.sidebar.text_area("Enter tickers (one per line)", "AAPL\nMSFT\nGOOG")
tickers = [t.strip() for t in tickers_input.split('\n') if t.strip()]

# Feature selection
feature_type = st.sidebar.selectbox("Feature Type", ["Price Change", "Volume Change"])

# Date range
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start Date", datetime(2023, 1, 1))
end_date = col2.date_input("End Date", datetime.now())

# CSV upload
uploaded_files = st.sidebar.file_uploader("Upload CSV files (time series)", accept_multiple_files=True, type=['csv'])

# Statistical test thresholds
st.sidebar.subheader("Statistical Test Thresholds")
adf_threshold = st.sidebar.slider("ADF p-value", 0.01, 0.1, 0.05, 0.01)
kpss_threshold = st.sidebar.slider("KPSS p-value", 0.01, 0.1, 0.05, 0.01)
eg_threshold = st.sidebar.slider("Engle-Granger p-value", 0.01, 0.1, 0.05, 0.01)
johansen_threshold = st.sidebar.slider("Johansen p-value", 0.01, 0.1, 0.05, 0.01)
vr_threshold = st.sidebar.slider("Variance Ratio p-value", 0.01, 0.1, 0.05, 0.01)

thresholds = {
    'adf': adf_threshold,
    'kpss': kpss_threshold,
    'engle_granger': eg_threshold,
    'johansen': johansen_threshold,
    'variance_ratio': vr_threshold
}

# Load data
@st.cache_data
def load_yfinance_data(tickers, feature_type, start, end):
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if not df.empty:
                if feature_type == "Price Change":
                    # Use Adj Close for price change calculation
                    if 'Adj Close' in df.columns:
                        price_col = 'Adj Close'
                    elif 'Close' in df.columns:
                        price_col = 'Close'
                    else:
                        continue
                    
                    series = df[price_col]
                    if isinstance(series, pd.DataFrame):
                        series = series.iloc[:, 0]
                    
                    # Calculate percentage change
                    change = series.pct_change().dropna()
                    data[f"{ticker}_PriceChange"] = change
                    
                elif feature_type == "Volume Change":
                    if 'Volume' not in df.columns:
                        st.warning(f"No volume data for {ticker}")
                        continue
                    
                    volume = df['Volume']
                    if isinstance(volume, pd.DataFrame):
                        volume = volume.iloc[:, 0]
                    
                    # Calculate percentage change in volume
                    change = volume.pct_change().dropna()
                    data[f"{ticker}_VolumeChange"] = change
                    
        except Exception as e:
            st.warning(f"Could not load {ticker}: {str(e)}")
    return data

@st.cache_data
def load_csv_data(file, feature_type):
    try:
        df = pd.read_csv(file)
        
        # Try to identify date column (first column or column with 'date' in name)
        date_col = None
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                date_col = col
                break
        
        if date_col is None:
            date_col = df.columns[0]
        
        # Try to parse dates
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df.set_index(date_col, inplace=True)
        df = df.sort_index()
        
        result = {}
        for col in df.columns:
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                # Calculate change
                if feature_type == "Price Change":
                    change = df[col].pct_change().dropna()
                    result[f"{file.name}_{col}_PriceChange"] = change
                elif feature_type == "Volume Change":
                    change = df[col].pct_change().dropna()
                    result[f"{file.name}_{col}_VolumeChange"] = change
        
        return result
    except Exception as e:
        st.warning(f"Error loading {file.name}: {str(e)}")
        return {}

# Load all data
all_data = {}

if st.sidebar.button("Load Data"):
    with st.spinner("Loading yfinance data..."):
        yf_data = load_yfinance_data(tickers, feature_type, start_date, end_date)
        all_data.update(yf_data)
    
    if uploaded_files:
        with st.spinner("Loading CSV files..."):
            for file in uploaded_files:
                csv_data = load_csv_data(file, feature_type)
                all_data.update(csv_data)
    
    st.session_state['all_data'] = all_data
    st.session_state['pairs_analyzed'] = False

if 'all_data' in st.session_state and st.session_state['all_data']:
    all_data = st.session_state['all_data']
    
    # Generate all pairs
    if not st.session_state.get('pairs_analyzed', False):
        with st.spinner("Analyzing pairs..."):
            pairs_results = []
            data_keys = list(all_data.keys())
            
            for i in range(len(data_keys)):
                for j in range(i + 1, len(data_keys)):
                    key1, key2 = data_keys[i], data_keys[j]
                    
                    # Align data
                    s1 = all_data[key1]
                    s2 = all_data[key2]
                    common_idx = s1.index.intersection(s2.index)
                    
                    if len(common_idx) > 30:  # Minimum data points
                        s1_aligned = s1.loc[common_idx].dropna()
                        s2_aligned = s2.loc[common_idx].dropna()
                        
                        # Align again after dropping NAs
                        final_idx = s1_aligned.index.intersection(s2_aligned.index)
                        
                        if len(final_idx) > 30:
                            s1_aligned = s1_aligned.loc[final_idx]
                            s2_aligned = s2_aligned.loc[final_idx]
                            
                            # Run tests
                            test_results = run_all_tests(s1_aligned, s2_aligned, thresholds)
                            
                            pairs_results.append({
                                'pair': f"{key1} / {key2}",
                                'asset1': key1,
                                'asset2': key2,
                                'tests': test_results,
                                'data1': s1_aligned,
                                'data2': s2_aligned
                            })
            
            st.session_state['pairs_results'] = pairs_results
            st.session_state['pairs_analyzed'] = True
    
    # Filter controls
    st.sidebar.subheader("Filter by Tests Passed")
    filter_adf = st.sidebar.checkbox("ADF", value=False)
    filter_kpss = st.sidebar.checkbox("KPSS", value=False)
    filter_eg = st.sidebar.checkbox("Engle-Granger", value=False)
    filter_johansen = st.sidebar.checkbox("Johansen", value=False)
    filter_vr = st.sidebar.checkbox("Variance Ratio", value=False)
    
    # Filter pairs
    filtered_pairs = []
    for pair in st.session_state.get('pairs_results', []):
        tests = pair['tests']
        if ((not filter_adf or tests['adf']['passed']) and
            (not filter_kpss or tests['kpss']['passed']) and
            (not filter_eg or tests['engle_granger']['passed']) and
            (not filter_johansen or tests['johansen']['passed']) and
            (not filter_vr or tests['variance_ratio']['passed'])):
            filtered_pairs.append(pair)
    
    # Display pairs list
    st.sidebar.subheader(f"Pairs ({len(filtered_pairs)})")
    
    selected_pair = st.sidebar.radio(
        "Select pair to analyze:",
        options=range(len(filtered_pairs)),
        format_func=lambda x: filtered_pairs[x]['pair'] if x < len(filtered_pairs) else "",
        key='pair_selector'
    )
    
    # Main display area
    if filtered_pairs and selected_pair is not None:
        pair_data = filtered_pairs[selected_pair]
        
        st.header(f"Analysis: {pair_data['pair']}")
        
        # Statistical tests results
        st.subheader("Statistical Test Results")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        tests = pair_data['tests']
        
        with col1:
            st.metric("ADF Test", 
                     "✓ Passed" if tests['adf']['passed'] else "✗ Failed",
                     f"p={tests['adf']['p_value']:.4f}")
        
        with col2:
            st.metric("KPSS Test",
                     "✓ Passed" if tests['kpss']['passed'] else "✗ Failed",
                     f"p={tests['kpss']['p_value']:.4f}")
        
        with col3:
            st.metric("Engle-Granger",
                     "✓ Passed" if tests['engle_granger']['passed'] else "✗ Failed",
                     f"p={tests['engle_granger']['p_value']:.4f}")
        
        with col4:
            st.metric("Johansen",
                     "✓ Passed" if tests['johansen']['passed'] else "✗ Failed",
                     f"trace={tests['johansen']['trace_statistic']:.2f}")
        
        with col5:
            st.metric("Variance Ratio",
                     "✓ Passed" if tests['variance_ratio']['passed'] else "✗ Failed",
                     f"VR={tests['variance_ratio']['variance_ratio']:.3f}")
        
        # Time series plot
        st.subheader("Change Series")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           subplot_titles=('Cumulative Changes', 'Daily Changes'))
        
        # Cumulative changes
        s1_cum = (1 + pair_data['data1']).cumprod()
        s2_cum = (1 + pair_data['data2']).cumprod()
        
        fig.add_trace(go.Scatter(x=s1_cum.index, y=s1_cum.values, name=pair_data['asset1']), row=1, col=1)
        fig.add_trace(go.Scatter(x=s2_cum.index, y=s2_cum.values, name=pair_data['asset2']), row=1, col=1)
        
        # Daily changes
        fig.add_trace(go.Scatter(x=pair_data['data1'].index, y=pair_data['data1'].values, 
                                name=f"{pair_data['asset1']}"), row=2, col=1)
        fig.add_trace(go.Scatter(x=pair_data['data2'].index, y=pair_data['data2'].values, 
                                name=f"{pair_data['asset2']}"), row=2, col=1)
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Kalman Filter Analysis
        st.subheader("Kalman Filter Mean Reversion")
        spread, z_score, hedge_ratios = apply_kalman_filter(pair_data['data1'], pair_data['data2'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Current Z-Score", f"{z_score.iloc[-1]:.3f}")
            st.metric("Mean Spread", f"{spread.mean():.4f}")
        
        with col2:
            st.metric("Spread Std Dev", f"{spread.std():.4f}")
            st.metric("Current Hedge Ratio", f"{hedge_ratios.iloc[-1]:.4f}")
        
        # Spread plot
        fig_spread = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   subplot_titles=('Spread', 'Z-Score'))
        
        fig_spread.add_trace(go.Scatter(x=spread.index, y=spread.values, name='Spread'), row=1, col=1)
        fig_spread.add_hline(y=spread.mean(), line_dash="dash", line_color="red", row=1, col=1)
        
        fig_spread.add_trace(go.Scatter(x=z_score.index, y=z_score.values, name='Z-Score'), row=2, col=1)
        fig_spread.add_hline(y=2, line_dash="dash", line_color="red", row=2, col=1)
        fig_spread.add_hline(y=-2, line_dash="dash", line_color="red", row=2, col=1)
        fig_spread.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        fig_spread.update_layout(height=500)
        st.plotly_chart(fig_spread, use_container_width=True)
        
        # Copula Analysis
        st.subheader("Copula Tail Probability Analysis")
        copula_results = fit_copula(pair_data['data1'].values, pair_data['data2'].values)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Correlation", f"{copula_results['correlation']:.4f}")
        with col2:
            st.metric("Lower Tail Prob (5%)", f"{copula_results['lower_tail_prob']:.4f}")
        with col3:
            st.metric("Upper Tail Prob (5%)", f"{copula_results['upper_tail_prob']:.4f}")
        
        # GARCH Analysis
        st.subheader("GARCH Volatility and Value at Risk")
        
        spread_clean = spread.dropna()
        if len(spread_clean) > 50:
            # Use the spread directly (already a change series)
            simulated_paths, var_results = fit_garch_and_simulate(spread_clean, n_simulations=500)
            
            if simulated_paths is not None and 'error' not in var_results:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Volatility", f"{var_results['volatility']:.4f}")
                with col2:
                    st.metric("VaR (95%)", f"{var_results['var_95']:.4f}")
                with col3:
                    st.metric("VaR (99%)", f"{var_results['var_99']:.4f}")
                
                # Generate future dates based on the last date in the data
                last_date = spread_clean.index[-1]
                
                # Infer the frequency of the data
                if len(spread_clean.index) > 1:
                    # Calculate median time difference
                    time_diffs = np.diff(spread_clean.index.values.astype('int64'))
                    median_diff = np.median(time_diffs)
                    
                    # Convert to pandas frequency
                    if median_diff < 1e9 * 60 * 60 * 2:  # Less than 2 hours in nanoseconds
                        freq = 'H'  # Hourly
                    elif median_diff < 1e9 * 60 * 60 * 24 * 1.5:  # Less than 1.5 days
                        freq = 'D'  # Daily
                    elif median_diff < 1e9 * 60 * 60 * 24 * 10:  # Less than 10 days
                        freq = 'W'  # Weekly
                    else:
                        freq = 'M'  # Monthly
                else:
                    freq = 'D'  # Default to daily
                
                # Generate future dates
                future_dates = pd.date_range(start=last_date, periods=simulated_paths.shape[0] + 1, freq=freq)[1:]
                
                # Plot simulated paths
                fig_garch = go.Figure()
                
                # Plot subset of simulations
                n_display = min(100, simulated_paths.shape[1])
                for i in range(n_display):
                    fig_garch.add_trace(go.Scatter(
                        x=future_dates,
                        y=simulated_paths[:, i],
                        mode='lines',
                        line=dict(width=0.5, color='lightblue'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                # Add mean path
                mean_path = simulated_paths.mean(axis=1)
                fig_garch.add_trace(go.Scatter(
                    x=future_dates,
                    y=mean_path,
                    mode='lines',
                    line=dict(width=3, color='red'),
                    name='Mean Path'
                ))
                
                # Add percentile bands
                percentile_5 = np.percentile(simulated_paths, 5, axis=1)
                percentile_95 = np.percentile(simulated_paths, 95, axis=1)
                
                fig_garch.add_trace(go.Scatter(
                    x=future_dates,
                    y=percentile_95,
                    mode='lines',
                    line=dict(width=2, color='orange', dash='dash'),
                    name='95th Percentile'
                ))
                
                fig_garch.add_trace(go.Scatter(
                    x=future_dates,
                    y=percentile_5,
                    mode='lines',
                    line=dict(width=2, color='orange', dash='dash'),
                    name='5th Percentile'
                ))
                
                # Add vertical line at current date
                fig_garch.add_vline(
                    x=last_date.timestamp() * 1000,  # Convert to milliseconds
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Today",
                    annotation_position="top"
                )
                
                fig_garch.update_layout(
                    title=f"GARCH Simulated Spread Paths (Next 30 {freq})",
                    xaxis_title="Date",
                    yaxis_title="Spread Value",
                    height=500,
                    xaxis=dict(
                        tickformat='%Y-%m-%d' if freq in ['D', 'W', 'M'] else '%Y-%m-%d %H:%M'
                    )
                )
                st.plotly_chart(fig_garch, use_container_width=True)
            else:
                error_detail = var_results.get('error', 'Unknown error')
                st.warning(f"Could not fit GARCH model: {error_detail}")
                with st.expander("Show error details"):
                    st.code(error_detail)
        else:
            st.warning(f"Insufficient data for GARCH analysis (need 50+ points, have {len(spread_clean)})")
        
else:
    st.info("Configure parameters in the sidebar and click 'Load Data' to begin analysis.")