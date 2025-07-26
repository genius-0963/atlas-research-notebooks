# %% [markdown]
# # Cryptocurrency Dark-Themed Volume Profile & Market Regime Analysis
#
# ## 📚 Educational Visualization for Market Structure Analysis
#
# This notebook creates an educational visualization combining Volume Profile analysis with Market Regime identification for BTC/USDT trading data from Binance using CCXT.
#
# **⚠️ DISCLAIMER: This is for educational purposes only. Not financial advice.**

# %% [markdown]
# ## 📖 Key Terminology & Concepts
#
# ### Volume Profile
# - **Definition**: A charting tool that shows trading activity over a specified time period at specified price levels
# - **Purpose**: Identifies price levels where significant trading occurred, which often act as areas of interest
# - **Key Components**:
#   - **High Volume Nodes (HVN)**: Price levels with significant trading activity
#   - **Low Volume Nodes (LVN)**: Price levels with minimal trading activity
#   - **Point of Control (POC)**: The price level with the highest traded volume
#
# ### Market Regimes (Theoretical Framework)
# Based on Wyckoff's market cycle theory, markets theoretically move through four phases:
#
# 1. **Accumulation**: A theoretical phase where informed participants might be building positions while prices move sideways
# 2. **Markup**: A theoretical uptrend phase where prices rise as demand increases
# 3. **Distribution**: A theoretical phase where informed participants might be reducing positions while prices move sideways
# 4. **Markdown**: A theoretical downtrend phase where prices fall as supply increases
#
# ### Technical Indicators Used
# - **Simple Moving Average (SMA)**: Average price over a specified number of periods
# - **Relative Volume**: Current volume compared to average volume
# - **Price Momentum**: Rate of price change over time

# %% [markdown]
# ## Install Required Packages
# 
# **Run this command first in a cell, then restart kernel:**
# 
# ```
# pip install ccxt pandas numpy matplotlib seaborn
# ```

# %% [markdown]
# ## Import Libraries

# %%
# Import required libraries
import warnings
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ccxt

# %%
warnings.filterwarnings('ignore')

# Set dark theme for all plots
plt.style.use('dark_background')
np.random.seed(42)

# %% [markdown]
# ## 📊 Data Collection
#
# We'll fetch historical price data from Binance using the CCXT library:
# - **Symbol**: BTC/USDT (Bitcoin priced in US Dollar Tether)
# - **Timeframe**: 4-hour candles (6 candles per day)
# - **Period**: Last 60 days
#
# The OHLCV data includes:
# - **Open**: Opening price of the period
# - **High**: Highest price during the period
# - **Low**: Lowest price during the period
# - **Close**: Closing price of the period
# - **Volume**: Total trading volume during the period

# %%
# Initialize exchange connection
exchange = ccxt.binance({'enableRateLimit': True})
symbol = 'BTC/USDT'
timeframe = '4h'
days = 60

try:
    # Calculate timestamp for 60 days ago
    since = exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)

    # Fetch OHLCV data
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)

    # Convert to DataFrame
    df = pd.DataFrame(
        ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    print(f"✅ Live data fetched: {len(df)} candles")
    print(
        f"📅 Period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"💰 Price range: ${df['low'].min():,.0f} - ${df['high'].max():,.0f}")
    print(f"📊 Current price: ${df['close'].iloc[-1]:,.2f}")
except Exception as e:
    print(f"⚠️ Error fetching data: {e}")
    print("This could be due to network issues or API limitations.")

# %% [markdown]
# ## 🔍 Market Regime Classification Algorithm
#
# The regime classification uses multiple technical indicators to categorize market behavior:
#
# ### Indicators Calculated:
# 1. **20 & 50 Period Moving Averages**: Trend direction indicators
# 2. **Relative Volume**: Volume compared to 20-period average
# 3. **Price Momentum**: 10-period rate of change
# 4. **Rolling Highs/Lows**: 10-period highest high and lowest low
#
# ### Classification Logic:
# - **Markup**: Price > MA20 > MA50, making higher highs/lows, positive momentum
# - **Markdown**: Price < MA20 < MA50, making lower highs/lows, negative momentum
# - **Distribution**: Price above MA50, high volume, slowing momentum
# - **Accumulation**: Price near/below MA50, elevated volume, low volatility

# %%


def calculate_market_regime(df, window=20):
    """
    Educational market regime classification based on technical indicators.
    This is a theoretical model for academic study.
    """
    # Calculate technical indicators
    df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['volume_sma'] = df['volume'].rolling(window=window).mean()
    df['relative_volume'] = df['volume'] / df['volume_sma']
    df['price_momentum'] = df['close'].pct_change(10)
    df['high_10'] = df['high'].rolling(10).max()
    df['low_10'] = df['low'].rolling(10).min()

    regimes = []
    regime_strength = []

    for i in range(len(df)):
        if i < 50:  # Not enough data for indicators
            regimes.append('accumulation')
            regime_strength.append(50)
            continue

        # Current indicator values
        price = df['close'].iloc[i]
        sma_short = df[f'sma_{window}'].iloc[i]
        sma_50 = df['sma_50'].iloc[i]
        rel_vol = df['relative_volume'].iloc[i]
        price_mom = df['price_momentum'].iloc[i]

        # Compare highs and lows
        prev_high = df['high_10'].iloc[i-10] if i >= 10 else df['high'].iloc[0]
        prev_low = df['low_10'].iloc[i-10] if i >= 10 else df['low'].iloc[0]
        current_high = df['high_10'].iloc[i]
        current_low = df['low_10'].iloc[i]

        # Classify regime based on conditions
        if (price > sma_short > sma_50 and current_high > prev_high and
                current_low > prev_low and price_mom > 0.01):
            regime = 'markup'
            strength = min(100, 60 + price_mom * 500)
        elif (price < sma_short < sma_50 and current_high < prev_high and
              current_low < prev_low and price_mom < -0.01):
            regime = 'markdown'
            strength = min(100, 60 + abs(price_mom) * 500)
        elif (price > sma_50 and rel_vol > 1.3 and price_mom < 0.005 and
              current_high >= prev_high):
            regime = 'distribution'
            strength = min(100, 50 + rel_vol * 20)
        elif (price <= sma_50 and rel_vol > 1.2 and abs(price_mom) < 0.01):
            regime = 'accumulation'
            strength = min(100, 50 + rel_vol * 20)
        else:
            regime = regimes[-1] if regimes else 'accumulation'
            strength = 40

        regimes.append(regime)
        regime_strength.append(strength)

    df['regime'] = regimes
    df['regime_strength'] = regime_strength
    return df


# Apply regime classification
df = calculate_market_regime(df)
print(f"\n🎯 Current Market Regime: {df['regime'].iloc[-1].upper()}")
print(f"💪 Regime Strength: {df['regime_strength'].iloc[-1]:.0f}%")
print("\nNote: This is a theoretical classification for educational purposes only.")

# %% [markdown]
# ## 📈 Volume Profile Calculation
#
# The Volume Profile algorithm distributes the traded volume across price levels:
#
# ### Process:
# 1. **Define Price Range**: From lowest low to highest high in the dataset
# 2. **Create Price Bins**: Divide the range into 100 equal levels
# 3. **Distribute Volume**: For each candle, calculate how much volume traded at each price level
# 4. **Identify POC**: Find the price level with the highest cumulative volume
#
# ### Why Volume Profile Matters:
# - **High Volume Nodes**: Areas where significant trading occurred, often act as reference points
# - **Low Volume Nodes**: Areas with minimal trading, price often moves quickly through these
# - **Point of Control**: The "fairest" price where most trading occurred

# %%


def calculate_volume_profile(df, bins=100):
    """
    Calculate volume distribution across price levels.
    This shows where the most trading activity occurred.
    """
    price_min, price_max = df['low'].min(), df['high'].max()
    price_levels = np.linspace(price_min, price_max, bins)

    volume_profile = []
    regime_profile = []

    for i in range(len(price_levels) - 1):
        level_low = price_levels[i]
        level_high = price_levels[i + 1]

        volume_at_level = 0
        regime_counts = {'accumulation': 0,
                         'distribution': 0, 'markup': 0, 'markdown': 0}

        # Calculate volume for each price level
        for idx in range(len(df)):
            candle_high = df['high'].iloc[idx]
            candle_low = df['low'].iloc[idx]
            candle_volume = df['volume'].iloc[idx]

            # Check if candle traded in this price range
            if candle_low <= level_high and candle_high >= level_low:
                # Calculate overlap percentage
                overlap_high = min(candle_high, level_high)
                overlap_low = max(candle_low, level_low)
                candle_range = candle_high - candle_low

                if candle_range > 0:
                    overlap_pct = (overlap_high - overlap_low) / candle_range
                    volume_at_level += candle_volume * overlap_pct
                    regime_counts[df['regime'].iloc[idx]] += 1

        volume_profile.append(volume_at_level)

        # Determine dominant regime at this price level
        dominant_regime = max(regime_counts, key=regime_counts.get) if sum(
            regime_counts.values()) > 0 else 'accumulation'
        regime_profile.append(dominant_regime)

    # Find Point of Control
    poc_index = np.argmax(volume_profile)
    poc_price = (price_levels[poc_index] + price_levels[poc_index + 1]) / 2

    return price_levels[:-1], np.array(volume_profile), regime_profile, poc_price


# Calculate volume profile
price_levels, volume_profile, regime_profile, poc_price = calculate_volume_profile(
    df)
print(f"\n🎯 Point of Control (POC): ${poc_price:,.0f}")
print("   → This represents the price level with the highest trading volume")

# Identify high volume nodes
high_volume_threshold = np.percentile(volume_profile[volume_profile > 0], 70)
high_volume_nodes = [(price_levels[i], volume_profile[i])
                     for i in range(len(volume_profile))
                     if volume_profile[i] > high_volume_threshold]
high_volume_nodes.sort(key=lambda x: x[1], reverse=True)

print("\n📊 Top 3 High Volume Nodes (HVN):")
for i, (price, vol) in enumerate(high_volume_nodes[:3], 1):
    diff_from_current = (
        (price - df['close'].iloc[-1]) / df['close'].iloc[-1]) * 100
    print(f"   {i}. ${price:,.0f} ({diff_from_current:+.1f}% from current)")

# %% [markdown]
# ## 🎨 Visualization Components
#
# The visualization consists of three main panels:
#
# ### 1. Main Price Chart (Left Panel)
# - **Candlesticks**: Show OHLC data for each 4-hour period
# - **Background Colors**: Indicate the theoretical market regime
# - **Moving Averages**: 20 and 50 period SMAs for trend reference
# - **POC Line**: Yellow dashed line showing Point of Control
# - **Regime Labels**: Show transitions between market phases
#
# ### 2. Volume Profile (Right Panel)
# - **Horizontal Bars**: Volume traded at each price level
# - **Bar Colors**: Indicate dominant regime when price was at that level
# - **HVN Markers**: Dotted lines showing high volume nodes
#
# ### 3. Regime Timeline (Bottom Panel)
# - **Colored Bands**: Show regime evolution over time
# - **Intensity**: Darker colors indicate stronger regime classification
# - **Transitions**: White vertical lines mark regime changes

# %%
# EDUCATIONAL VISUALIZATION - NOT FINANCIAL ADVICE
# This is for educational and research purposes only
# Market regimes are theoretical concepts for academic study

# Define color scheme
regime_colors = {
    'accumulation': '#2ECC71',  # Green (theoretical accumulation phase)
    'distribution': '#E74C3C',  # Red (theoretical distribution phase)
    'markup': '#3498DB',        # Blue (theoretical markup phase)
    'markdown': '#E67E22'       # Orange (theoretical markdown phase)
}

# Candlestick colors
candle_colors = {
    'up': '#26A69A',    # Teal for positive close
    'down': '#EF5350'   # Red for negative close
}

# Create figure with subplots
fig = plt.figure(figsize=(24, 14), facecolor='#0B0E11')
gs = fig.add_gridspec(3, 2, height_ratios=[2.5, 2.5, 1], width_ratios=[4, 1],
                      hspace=0.12, wspace=0.08)

# Create axes
ax_main = fig.add_subplot(gs[:2, 0])     # Price chart
ax_vp = fig.add_subplot(gs[:2, 1])       # Volume profile
ax_regime = fig.add_subplot(gs[2, :])    # Regime timeline

# Set dark background for all axes
for ax in [ax_main, ax_vp, ax_regime]:
    ax.set_facecolor('#151A1E')

# %%
# PRICE CHART - HISTORICAL DATA VISUALIZATION
# Add regime background coloring for educational reference
for i in range(len(df)-1):
    regime = df['regime'].iloc[i]
    ax_main.axvspan(df.index[i], df.index[i+1],
                    alpha=0.08, color=regime_colors[regime],
                    zorder=0, linewidth=0)

# Plot candlesticks
for i in range(len(df)):
    row = df.iloc[i]
    is_up = row['close'] >= row['open']
    color = candle_colors['up'] if is_up else candle_colors['down']

    # Wicks (high-low lines)
    ax_main.plot([df.index[i], df.index[i]], [row['low'], row['high']],
                 color=color, alpha=0.7, linewidth=1.2, zorder=4)

    # Bodies (open-close bars)
    body_height = abs(row['close'] - row['open'])
    body_bottom = min(row['open'], row['close'])

    if body_height > 0:
        ax_main.bar(df.index[i], body_height, bottom=body_bottom,
                    width=0.15, color=color, alpha=0.8,
                    edgecolor=color, linewidth=0.8, zorder=5)

# Add price line
ax_main.plot(df.index, df['close'], color='#ECEFF1', linewidth=1.8,
             alpha=0.9, zorder=6, label='Close Price')

# POC line - educational reference only
ax_main.axhline(y=poc_price, color='#FFD700', linewidth=2.5, alpha=0.8,
                linestyle='--', label=f'Volume POC: ${poc_price:,.0f}', zorder=7)
ax_main.axhline(y=poc_price, color='#FFD700', linewidth=6, alpha=0.2, zorder=6)

# Moving averages - for educational reference
if 'sma_20' in df.columns:
    ax_main.plot(df.index, df['sma_20'], color='#9C27B0', linewidth=1.3,
                 alpha=0.7, label='20-Period MA', linestyle='-')
if 'sma_50' in df.columns:
    ax_main.plot(df.index, df['sma_50'], color='#00BCD4', linewidth=1.3,
                 alpha=0.7, label='50-Period MA', linestyle='-')

# Mark regime changes for educational observation
regime_changes = []
for i in range(1, len(df)):
    if df['regime'].iloc[i] != df['regime'].iloc[i-1]:
        regime_changes.append({
            'index': i,
            'from': df['regime'].iloc[i-1],
            'to': df['regime'].iloc[i],
            'price': df['close'].iloc[i]
        })
        # Add vertical line at transition
        ax_main.axvline(x=df.index[i], color='white',
                        linestyle=':', alpha=0.3, linewidth=1)

# Label recent regime changes - educational labels only
for change in regime_changes[-5:]:
    regime_labels = {
        'accumulation': 'ACC',
        'markup': 'MKP',
        'distribution': 'DST',
        'markdown': 'MKD'
    }
    ax_main.annotate(regime_labels[change['to']],
                     xy=(df.index[change['index']], change['price']),
                     xytext=(0, 15), textcoords='offset points',
                     fontsize=9, color=regime_colors[change['to']],
                     bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='#0B0E11',
                               edgecolor=regime_colors[change['to']],
                               alpha=0.8),
                     ha='center', va='bottom', zorder=8)

# Styling
ax_main.set_xlabel('Date', fontsize=13, color='#B0BEC5')
ax_main.set_ylabel('Price (USDT)', fontsize=13, color='#B0BEC5')
ax_main.set_title('Historical Price Data - Educational Visualization Only',
                  fontsize=16, color='#ECEFF1', pad=15)
ax_main.grid(True, alpha=0.15, color='#37474F', linestyle='-', linewidth=0.5)
ax_main.legend(loc='upper left', framealpha=0.9, facecolor='#1C2025',
               edgecolor='#37474F', fontsize=11)

# Format dates
ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax_main.xaxis.set_major_locator(mdates.DayLocator(interval=7))
plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45, ha='right')

# %%
# VOLUME PROFILE - HISTORICAL ANALYSIS
max_volume = volume_profile.max() if volume_profile.max() > 0 else 1

for i, (price, volume, regime) in enumerate(zip(price_levels, volume_profile, regime_profile)):
    if volume > 0:
        normalized_volume = volume / max_volume
        color = regime_colors[regime]

        # Highlight high volume areas
        if volume > high_volume_threshold:
            ax_vp.barh(price, 1,
                       height=(price_levels[1] - price_levels[0]) * 0.9,
                       color=color, alpha=0.15, edgecolor='none', zorder=1)

        # Main volume bar
        ax_vp.barh(price, normalized_volume,
                   height=(price_levels[1] - price_levels[0]) * 0.85,
                   color=color, alpha=0.7, edgecolor=color,
                   linewidth=0.8, zorder=2)

# POC line
ax_vp.axhline(y=poc_price, color='#FFD700', linewidth=3, alpha=0.9, zorder=10)
ax_vp.text(0.95, poc_price, 'POC', color='#FFD700', fontsize=11,
           fontweight='bold', ha='right', va='bottom',
           transform=ax_vp.get_yaxis_transform())

# Mark high volume areas - for educational reference
for i, (price, vol) in enumerate(high_volume_nodes[:3]):
    ax_vp.axhline(y=price, color='#B0BEC5', linestyle=':',
                  alpha=0.4, linewidth=1)
    ax_vp.text(0.02, price, f'HVN{i+1}', color='#B0BEC5', fontsize=9,
               ha='left', va='center', transform=ax_vp.get_yaxis_transform())

# Styling
ax_vp.set_xlabel('Relative Volume', fontsize=13, color='#B0BEC5')
ax_vp.set_title('Volume Distribution', fontsize=14, color='#ECEFF1')
ax_vp.set_ylim(ax_main.get_ylim())
ax_vp.grid(True, alpha=0.1, color='#37474F', axis='y')
ax_vp.set_xticks([0, 0.5, 1])
ax_vp.set_xticklabels(['0%', '50%', '100%'])
ax_vp.tick_params(colors='#B0BEC5')

# %%
# REGIME TIMELINE - THEORETICAL MARKET PHASES
# Create regime bands for educational visualization
for i in range(len(df)-1):
    regime = df['regime'].iloc[i]
    strength = df['regime_strength'].iloc[i] / 100

    ax_regime.fill_between([df.index[i], df.index[i+1]],
                           0, 1,
                           color=regime_colors[regime],
                           alpha=0.3 + strength * 0.3,
                           edgecolor='none')

# Add regime reference lines and educational labels
regime_y_positions = {
    'accumulation': 0.125,
    'markup': 0.375,
    'distribution': 0.625,
    'markdown': 0.875
}

# Create educational label area
label_x = df.index[0] - timedelta(days=3)
for regime, y_pos in regime_y_positions.items():
    # Reference line
    ax_regime.axhline(y=y_pos, color=regime_colors[regime],
                      linestyle='--', alpha=0.2, linewidth=1)

    # Educational labels only
    regime_educational_labels = {
        'accumulation': 'Accumulation\n(Theoretical Phase)',
        'markup': 'Markup\n(Theoretical Phase)',
        'distribution': 'Distribution\n(Theoretical Phase)',
        'markdown': 'Markdown\n(Theoretical Phase)'
    }

    ax_regime.text(label_x, y_pos, regime_educational_labels[regime],
                   color=regime_colors[regime], fontsize=10,
                   fontweight='bold', va='center', ha='right',
                   bbox=dict(boxstyle='round,pad=0.4',
                             facecolor='#0B0E11',
                             edgecolor=regime_colors[regime],
                             alpha=0.8, linewidth=1))

# Mark transitions
for i in range(1, len(df)):
    if df['regime'].iloc[i] != df['regime'].iloc[i-1]:
        ax_regime.axvline(x=df.index[i], color='#ECEFF1',
                          linestyle='-', alpha=0.3, linewidth=1.5)

# Current observation - no recommendations
current_regime = df['regime'].iloc[-1]
current_price = df['close'].iloc[-1]

ax_regime.text(df.index[-1], 0.5,
               f'Current Classification:\n{current_regime.upper()}\n(Educational Model)',
               color=regime_colors[current_regime], fontsize=12,
               fontweight='bold', ha='right', va='center',
               bbox=dict(boxstyle='round,pad=0.6',
                         facecolor='#0B0E11',
                         edgecolor=regime_colors[current_regime],
                         linewidth=2, alpha=0.9))

# Styling
ax_regime.set_ylabel('Theoretical Market Phase', fontsize=13, color='#B0BEC5')
ax_regime.set_xlabel('Date', fontsize=13, color='#B0BEC5')
ax_regime.set_title(
    'Market Regime Classification - Educational Model', fontsize=14, color='#ECEFF1')
ax_regime.set_ylim(0, 1)
ax_regime.set_yticks([])
ax_regime.grid(True, alpha=0.1, color='#37474F', axis='x')
ax_regime.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax_regime.xaxis.set_major_locator(mdates.DayLocator(interval=7))
plt.setp(ax_regime.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Main title with disclaimer
fig.suptitle('VOLUME PROFILE & MARKET REGIME ANALYSIS',
             fontsize=22, color='#ECEFF1', y=0.97, fontweight='bold')
fig.text(0.5, 0.94, f'{symbol} • {timeframe} • Educational Visualization Only',
         fontsize=14, color='#B0BEC5', alpha=0.8, ha='center')
fig.text(0.5, 0.91, 'NOT FINANCIAL ADVICE - For Educational Purposes Only',
         fontsize=12, color='#E74C3C', alpha=0.9, ha='center', style='italic')

# Adjust layout
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 📚 Educational Summary
#
# ### What This Visualization Shows:
#
# 1. **Price Action**: Historical price movements with candlestick charts
# 2. **Volume Profile**: Where the most trading activity occurred across different price levels
# 3. **Market Regimes**: Theoretical classification of market phases based on technical indicators
#
# ### Key Takeaways:
#
# - **Volume Profile** helps identify price levels where significant trading occurred
# - **Market Regimes** are theoretical constructs to understand market cycles
# - **Technical Analysis** uses historical data to identify patterns
#
# ### Important Reminders:
#
# - Past performance does not indicate future results
# - Market behavior is influenced by countless factors not captured in price charts
# - This is an educational script for learning about market structure concepts
# - Always do your own research and consult with qualified professionals
#
# ### Further Learning:
#
# - Study Wyckoff Method for market cycle theory
# - Learn about Market Profile and auction market theory
# - Understand the limitations of technical analysis
# - Practice risk management and position sizing
#
# **Remember: This visualization is for educational purposes only and should not be used as the basis for investment decisions.**

# %%
# Print educational disclaimer
print("\n" + "="*70)
print("⚠️  EDUCATIONAL DISCLAIMER")
print("="*70)
print("\nThis notebook demonstrates technical analysis concepts for educational purposes.")
print("It is not intended as financial advice or trading recommendations.")
print("\nKey Points:")
print("• Market regimes are theoretical classifications")
print("• Volume profile shows historical trading activity")
print("• Technical indicators are mathematical calculations based on past data")
print("• No analysis method can predict future market movements with certainty")
print("\nAlways conduct thorough research and consult qualified professionals")
print("before making any investment decisions.")
print("\n" + "="*70)
