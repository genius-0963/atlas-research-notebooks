# %% [markdown]
# # Cryptocurrency Correlation Analysis
# Fetching top 20 cryptocurrencies by volume and analyzing their price correlations using CCXT

# %% [markdown]
# ## Install Required Packages
# 
# **Run this command first in a cell, then restart kernel:**
# 
# ```
# pip install -q ccxt pandas numpy matplotlib seaborn
# ```

# %% [markdown]
# ## Import Libraries

# %%
import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
from matplotlib.colors import LinearSegmentedColormap

# %% [markdown]
# ## Configuration

# %%
warnings.filterwarnings('ignore')

# %% [markdown]
# ## Connect to Exchange and Fetch Top 20 Cryptocurrencies

# %%
exchange = ccxt.binance()
tickers = exchange.fetch_tickers()

# %%
ticker_data = []
for symbol, ticker in tickers.items():
    if '/USDT' in symbol and ticker['quoteVolume'] is not None:
        ticker_data.append({
            'symbol': symbol,
            'volume': ticker['quoteVolume']
        })

df_tickers = pd.DataFrame(ticker_data)
df_tickers = df_tickers.sort_values('volume', ascending=False)
top_20_symbols = df_tickers.head(20)['symbol'].tolist()

print("Top 20 cryptocurrencies by trading volume:")
for i, symbol in enumerate(top_20_symbols, 1):
    print(f"{i}. {symbol}")

# %% [markdown]
# ## Fetch Historical Price Data

# %%


def fetch_ohlcv_data(exchange, symbol, timeframe='1d', limit=365):
    """
    Fetch OHLCV data for a given symbol from the exchange.

    Args:
        exchange: CCXT exchange instance
        symbol: Trading pair symbol (e.g., 'BTC/USDT')
        timeframe: Candle timeframe (default: '1d')
        limit: Number of candles to fetch (default: 365)

    Returns:
        pandas Series with close prices or None if error
    """
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        if not ohlcv or len(ohlcv) < 2:
            return None

        df = pd.DataFrame(
            ohlcv, columns=['timestamp', 'open',
                            'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df['close']
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None


# %%
price_data = {}
failed_symbols = []

for symbol in top_20_symbols:
    print(f"Fetching data for {symbol}...")
    data = fetch_ohlcv_data(exchange, symbol, timeframe='1d', limit=365)

    if data is not None and len(data) >= 30:  # Ensure sufficient data
        price_data[symbol.replace('/USDT', '')] = data
    else:
        failed_symbols.append(symbol)

    time.sleep(0.1)  # Rate limiting

print(f"\nSuccessfully fetched data for {len(price_data)} cryptocurrencies")
if failed_symbols:
    print(f"Failed to fetch sufficient data for: {failed_symbols}")

# %% [markdown]
# ## Calculate Returns and Correlation Matrix

# %%
# Create price DataFrame and calculate returns
price_df = pd.DataFrame(price_data)
price_df = price_df.dropna()
returns_df = price_df.pct_change().dropna()

print("Price data shape:", price_df.shape)
print("Returns data shape:", returns_df.shape)

# %%
# Calculate correlation matrix
correlation_matrix = returns_df.corr()
print("Correlation Matrix:")
print(correlation_matrix.round(3))

# %% [markdown]
# ## Visualize Correlation Matrix

# %%
# Set up the plot style
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(14, 12))
fig.patch.set_facecolor('#1a1a1a')
ax.set_facecolor('#1a1a1a')

# Create custom colormap
colors = ['#FF4500', '#2a2a2a', '#4A9D9A']  # Red -> Gray -> Teal
cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)

# Create mask for upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

# Create heatmap
sns.heatmap(correlation_matrix,
            mask=mask,
            annot=True,
            cmap=cmap,
            center=0,
            square=True,
            fmt='.2f',
            linewidths=0.5,
            linecolor='#333333',
            cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'},
            annot_kws={'size': 9, 'color': 'white', 'weight': 'bold'})

# Customize the plot
plt.title('Cryptocurrency Returns Correlation Matrix\n(Top 20 by Trading Volume)',
          fontsize=18, fontweight='bold', pad=25, color='white')
plt.xticks(rotation=45, ha='right', color='white', fontsize=10)
plt.yticks(rotation=0, color='white', fontsize=10)

# Customize colorbar
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(colors='white')
cbar.ax.yaxis.label.set_color('white')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Analysis Insights
#
# This correlation matrix shows the relationships between daily returns of the top 20 cryptocurrencies by trading volume.
#
# - **High positive correlations** (teal) indicate cryptocurrencies that tend to move together
# - **Low correlations** (gray) suggest relatively independent price movements
# - **Negative correlations** (red) indicate inverse relationships (rare in crypto markets)
#
# Most major cryptocurrencies show moderate to high positive correlations, reflecting the overall market tendency to move together.
