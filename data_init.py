import yfinance as yf
import pandas as pd
from dask import dataframe as dd
import json

# Load config settings into dictionary
config = json.load(open('config.json'))

# Price adjacent to close
prices = [yf.download(ticker, config['start'], config['end'])['Adj Close'] for ticker in config['tickers']]

# Concatenate the array into a multi-ticker time series
df = pd.concat(prices, axis=1)

# Find the % Change from the time series, excluding the N/A elements
ddf = dd.from_pandas(df.pct_change().dropna(), npartitions=int(config['npartitions']))

# Compute correlation matrix
m_corr = ddf.corr().compute()