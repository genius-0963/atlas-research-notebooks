# %% [markdown]
# # Bitcoin Rainbow Chart with Log Regression and Halving Events
# Analysis of Bitcoin price using logarithmic regression bands to create the famous "Rainbow Chart" with halving markers

# %%
!pip install ccxt pandas numpy matplotlib seaborn scipy

# %%
import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set dark theme
plt.style.use('dark_background')

# %% [markdown]
# ## Fetch Bitcoin Historical Price Data from Binance using CCXT

# %%
# Initialize exchange and fetch Bitcoin data
exchange = ccxt.binance()
symbol = 'BTC/USDT'
timeframe = '1d'
since = exchange.parse8601('2017-01-01T00:00:00Z')

# Fetch OHLCV data
ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)

# Continue fetching to get all historical data
all_ohlcv = ohlcv
while len(ohlcv) == 1000:
    since = ohlcv[-1][0] + 86400000  # Add one day in milliseconds
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
    all_ohlcv.extend(ohlcv)

# Convert to DataFrame
df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('date', inplace=True)
df = df[df['close'] > 0]  # Remove any zero prices

# %% [markdown]
# ## Calculate Logarithmic Regression and Rainbow Bands

# %%
# Prepare data for log regression
df['days_since_start'] = (df.index - df.index[0]).days
df['log_price'] = np.log10(df['close'])

# Perform linear regression on log prices
x = df['days_since_start'].values
y = df['log_price'].values
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Calculate regression line
df['log_regression'] = slope * df['days_since_start'] + intercept
df['regression_price'] = 10 ** df['log_regression']

# Define rainbow bands (multipliers for the regression line)
bands = {
    'Maximum Bubble': 3.5,
    'Selling. Seriously, SELL!': 2.8,
    'FOMO intensifies': 2.2,
    'Is this a bubble?': 1.7,
    'HODL!': 1.3,
    'Still cheap': 1.0,
    'Accumulate': 0.7,
    'BUY!': 0.5,
    'Fire sale!': 0.35
}

# Calculate band prices
for band_name, multiplier in bands.items():
    df[band_name] = df['regression_price'] * multiplier

# %% [markdown]
# ## Define Bitcoin Halving Dates and Prepare Visualization

# %%
# Bitcoin halving dates
halvings = [
    datetime(2012, 11, 28),  # First halving
    datetime(2016, 7, 9),     # Second halving
    datetime(2020, 5, 11),    # Third halving
    datetime(2024, 4, 20)     # Fourth halving
]

# Filter halvings to only include those within our data range
halvings = [h for h in halvings if df.index[0] <= h <= df.index[-1]]

# Define rainbow colors for bands
colors = [
    '#FF0000',  # Red - Maximum Bubble
    '#FF4500',  # Orange Red - Selling
    '#FFA500',  # Orange - FOMO
    '#FFD700',  # Gold - Bubble?
    '#ADFF2F',  # Green Yellow - HODL
    '#00FF00',  # Green - Still cheap
    '#00CED1',  # Dark Turquoise - Accumulate
    '#0000FF',  # Blue - BUY!
    '#4B0082'   # Indigo - Fire sale!
]

# %% [markdown]
# ## Create the Bitcoin Rainbow Chart with Halving Markers

# %%
# Create the rainbow chart
fig, ax = plt.subplots(figsize=(16, 10))
fig.patch.set_facecolor('#0a0a0a')
ax.set_facecolor('#0a0a0a')

# Plot rainbow bands
band_names = list(bands.keys())
for i in range(len(band_names)):
    if i == 0:
        ax.fill_between(df.index, df[band_names[i]], df[band_names[i]]*2, 
                        color=colors[i], alpha=0.7, label=band_names[i])
    else:
        ax.fill_between(df.index, df[band_names[i-1]], df[band_names[i]], 
                        color=colors[i], alpha=0.7, label=band_names[i])

# Plot actual Bitcoin price
ax.plot(df.index, df['close'], color='white', linewidth=2, label='BTC Price', zorder=5)

# Add halving markers
for halving_date in halvings:
    ax.axvline(x=halving_date, color='yellow', linestyle='--', linewidth=1.5, alpha=0.8)
    # Get y position for label
    y_pos = ax.get_ylim()[1] * 0.7
    ax.text(halving_date, y_pos, 'Halving', rotation=90, 
            verticalalignment='bottom', color='yellow', fontsize=10, fontweight='bold')

# Set logarithmic scale for y-axis
ax.set_yscale('log')

# Formatting
ax.set_xlabel('Date', fontsize=14, color='white')
ax.set_ylabel('Bitcoin Price (USD)', fontsize=14, color='white')
ax.set_title('Bitcoin Rainbow Chart - Logarithmic Regression with Halving Events', 
             fontsize=18, fontweight='bold', color='white', pad=20)

# Format x-axis dates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator())

# Grid
ax.grid(True, alpha=0.2, color='gray', linestyle='-', linewidth=0.5)

# Legend
ax.legend(loc='upper left', fontsize=10, framealpha=0.9, facecolor='#1a1a1a')

# Adjust layout
plt.tight_layout()

# %% [markdown]
# ## Display Current Price Position and Statistics

# %%
# Calculate current position in the rainbow
current_price = df['close'].iloc[-1]
current_regression = df['regression_price'].iloc[-1]
current_multiplier = current_price / current_regression

# Determine current band
current_band = 'Below Fire sale!'
for band_name, multiplier in sorted(bands.items(), key=lambda x: x[1]):
    if current_multiplier >= multiplier:
        current_band = band_name

# Display statistics
print(f"Bitcoin Rainbow Chart Analysis")
print(f"=" * 50)
print(f"Current Date: {df.index[-1].strftime('%Y-%m-%d')}")
print(f"Current BTC Price: ${current_price:,.2f}")
print(f"Regression Price: ${current_regression:,.2f}")
print(f"Current Multiplier: {current_multiplier:.2f}x")
print(f"Current Band: {current_band}")
print(f"\nRegression Statistics:")
print(f"Slope (log scale): {slope:.6f}")
print(f"R-squared: {r_value**2:.4f}")
print(f"Days analyzed: {df['days_since_start'].iloc[-1]}")

plt.show()

# %% [markdown]
# ## Summary
# 
# The Bitcoin Rainbow Chart successfully visualizes:
# - **Logarithmic regression trend** showing Bitcoin's long-term growth trajectory
# - **Nine colored bands** indicating market sentiment zones from "Fire sale!" to "Maximum Bubble"
# - **Bitcoin halving events** marked with yellow vertical lines showing their historical impact
# - **Current price position** within the rainbow bands to assess market conditions
# 
# The chart uses a dark theme for better visibility and includes all historical halvings within the data range. The logarithmic scale helps visualize Bitcoin's exponential growth pattern over time.

