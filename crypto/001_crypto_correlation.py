# %% [markdown]
# Cryptocurrency Correlation Analysis
# Fetching top 20 cryptocurrencies by volume and analyzing their price correlations using CCXT

# %%
from matplotlib.colors import LinearSegmentedColormap
import warnings
import time
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ccxt
pip install ccxt pandas numpy matplotlib seaborn

# %%

# %%
warnings.filterwarnings('ignore')

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

# %%


def fetch_ohlcv_data(exchange, symbol, timeframe='1d', limit=365):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        if not ohlcv or len(ohlcv) < 2:
            return None
        df = pd.DataFrame(
            ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df['close']
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None


price_data = {}
failed_symbols = []

for symbol in top_20_symbols:
    print(f"Fetching data for {symbol}...")
    data = fetch_ohlcv_data(exchange, symbol, timeframe='1d', limit=365)
    if data is not None and len(data) >= 30:
        price_data[symbol.replace('/USDT', '')] = data
    else:
        failed_symbols.append(symbol)
    time.sleep(0.1)

print(f"\nSuccessfully fetched data for {len(price_data)} cryptocurrencies")
if failed_symbols:
    print(f"Failed to fetch sufficient data for: {failed_symbols}")

# %%
price_df = pd.DataFrame(price_data)
price_df = price_df.dropna()

returns_df = price_df.pct_change().dropna()

print("Price data shape:", price_df.shape)
print("Returns data shape:", returns_df.shape)

# %%
correlation_matrix = returns_df.corr()
print("Correlation Matrix:")
print(correlation_matrix.round(3))

# %%
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(14, 12))
fig.patch.set_facecolor('#1a1a1a')
ax.set_facecolor('#1a1a1a')

colors = ['#FF4500', '#2a2a2a', '#4A9D9A']
cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

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

plt.title('Cryptocurrency Returns Correlation Matrix\n(Top 20 by Trading Volume)',
          fontsize=18, fontweight='bold', pad=25, color='white')
plt.xticks(rotation=45, ha='right', color='white', fontsize=10)
plt.yticks(rotation=0, color='white', fontsize=10)

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(colors='white')
cbar.ax.yaxis.label.set_color('white')

plt.tight_layout()
plt.show()
