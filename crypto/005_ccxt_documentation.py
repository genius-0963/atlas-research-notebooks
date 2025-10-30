# %% [markdown]
# # CCXT Usage
# 
# The ccxt library is a collection of available crypto exchanges or exchange classes. Each class implements the public and private API for a particular crypto exchange. All exchanges are derived from the base Exchange class and share a set of common methods. To access a particular exchange from ccxt library you need to create an instance of corresponding exchange class. Supported exchanges are updated frequently and new exchanges are added regularly.

# %%
!pip install mplfinance ccxt 

# %% [markdown]
# ## Imports

# %%
import ccxt
import mplfinance as mpl
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import datetime


# %% [markdown]
# # Instantiation
# 
# To connect to an exchange and start trading you need to instantiate an exchange class from ccxt library.

# %%
import ccxt
print (ccxt.exchanges)

# %% [markdown]
# # Order Book
# 
# Exchanges expose information on open orders with bid (buy) and ask (sell) prices, volumes and other data. Usually there is a separate endpoint for querying current state (stack frame) of the order book for a particular market. An order book is also often called market depth. The order book information is used in the trading decision making process.
# 
# To get data on order books, you can use
# 
# - fetchOrderBook () // for a single markets order books
# - fetchOrderBooks ( symbols ) // for multiple markets order books
# - fetchOrderBooks () // for the order books of all markets
# 
# Orderbook Response structure
# 
# ```json
# {
#     'bids': [
#         [ price, amount ], // [ float, float ]
#         [ price, amount ],
#         ...
#     ],
#     'asks': [
#         [ price, amount ],
#         [ price, amount ],
#         ...
#     ],
#     'symbol': 'ETH/BTC', // a unified market symbol
#     'timestamp': 1499280391811, // Unix Timestamp in milliseconds (seconds * 1000)
#     'datetime': '2017-07-05T18:47:14.692Z', // ISO8601 datetime string with milliseconds
#     'nonce': 1499280391811, // an increasing unique identifier of the orderbook snapshot
# }
# ```

# %%
# Set dark theme
plt.style.use('dark_background')

# Instantiate exchange
exchange = ccxt.binance()

# Fetch order book for BTC/USD pair
symbol = 'BTC/USD'
orderbook = exchange.fetch_order_book(symbol, limit=50)

# Extract more data points for smoother depth chart
bids = orderbook['bids'][:25]
asks = orderbook['asks'][:25]

# Convert to DataFrames and calculate cumulative volumes
bids_df = pd.DataFrame(bids, columns=['price', 'volume'])
asks_df = pd.DataFrame(asks, columns=['price', 'volume'])

# Sort and calculate cumulative volumes for depth
bids_df = bids_df.sort_values('price', ascending=False)
asks_df = asks_df.sort_values('price', ascending=True)

bids_df['cumulative_volume'] = bids_df['volume'].cumsum()
asks_df['cumulative_volume'] = asks_df['volume'].cumsum()

# Create the depth chart
fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor('#0e1621')
ax.set_facecolor('#0e1621')

# Plot bids (buy side) - green
ax.fill_between(bids_df['price'], 0, bids_df['cumulative_volume'], 
                step='post', alpha=0.6, color='#00d4aa', label='Bids')
ax.plot(bids_df['price'], bids_df['cumulative_volume'], 
        drawstyle='steps-post', color='#00d4aa', linewidth=2)

# Plot asks (sell side) - red
ax.fill_between(asks_df['price'], 0, asks_df['cumulative_volume'], 
                step='pre', alpha=0.6, color='#f84960', label='Asks')
ax.plot(asks_df['price'], asks_df['cumulative_volume'], 
        drawstyle='steps-pre', color='#f84960', linewidth=2)

# Customize the plot with Binance-like styling
ax.set_xlabel('Price (USD)', color='#d1d4dc', fontsize=12)
ax.set_ylabel('Cumulative Volume (BTC)', color='#d1d4dc', fontsize=12)
ax.set_title(f'{symbol} Order Book Depth Chart', color='#f0b90b', fontsize=16, fontweight='bold')

# Style the axes
ax.tick_params(colors='#d1d4dc')
ax.spines['bottom'].set_color('#2b3139')
ax.spines['top'].set_color('#2b3139')
ax.spines['left'].set_color('#2b3139')
ax.spines['right'].set_color('#2b3139')

# Add grid
ax.grid(True, alpha=0.2, color='#2b3139')

# Legend styling
legend = ax.legend(loc='upper right', facecolor='#1e2329', edgecolor='#2b3139')
legend.get_frame().set_alpha(0.9)
for text in legend.get_texts():
    text.set_color('#d1d4dc')

# Format axes
ax.ticklabel_format(style='plain', axis='x')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Display market summary
spread = asks[0][0] - bids[0][0]
spread_pct = (spread / bids[0][0]) * 100
total_bid_volume = sum([bid[1] for bid in bids])
total_ask_volume = sum([ask[1] for ask in asks])

print(f"📊 Market Summary for {symbol}")
print(f"Best Bid: ${bids[0][0]:,.2f}")
print(f"Best Ask: ${asks[0][0]:,.2f}")
print(f"Spread: ${spread:,.2f} ({spread_pct:.3f}%)")
print(f"Total Bid Volume: {total_bid_volume:.4f} BTC")
print(f"Total Ask Volume: {total_ask_volume:.4f} BTC")

# %% [markdown]
# # Price Tickers
# 
# A price ticker contains statistics for a particular market/symbol for some period of time in recent past, usually last 24 hours. The methods for fetching tickers are described below.
# 
# ```python
# # one ticker
# fetchTicker (symbol, params = {})
# 
# # example
# fetchTicker ('ETH/BTC')
# fetchTicker ('BTC/USDT')
# ```
# 
# ### Multiple Tickers for many symbols
# 
# ```python
# # multiple tickers
# fetchTickers (symbols = undefined, params = {})  // for all tickers at once
# 
# # for example
# fetchTickers () // all symbols
# fetchTickers ([ 'ETH/BTC', 'BTC/USDT' ]) // an array of specific symbols
# ```
# 
# ### Ticker Structure
# 
# ```javascript
# {
#     'symbol':        string symbol of the market ('BTC/USD', 'ETH/BTC', ...)
#     'info':        { the original non-modified unparsed reply from exchange API },
#     'timestamp':     int (64-bit Unix Timestamp in milliseconds since Epoch 1 Jan 1970)
#     'datetime':      ISO8601 datetime string with milliseconds
#     'high':          float, // highest price
#     'low':           float, // lowest price
#     'bid':           float, // current best bid (buy) price
#     'bidVolume':     float, // current best bid (buy) amount (may be missing or undefined)
#     'ask':           float, // current best ask (sell) price
#     'askVolume':     float, // current best ask (sell) amount (may be missing or undefined)
#     'vwap':          float, // volume weighed average price
#     'open':          float, // opening price
#     'close':         float, // price of last trade (closing price for current period)
#     'last':          float, // same as `close`, duplicated for convenience
#     'previousClose': float, // closing price for the previous period
#     'change':        float, // absolute change, `last - open`
#     'percentage':    float, // relative change, `(change/open) * 100`
#     'average':       float, // average price, `(last + open) / 2`
#     'baseVolume':    float, // volume of base currency traded for last 24 hours
#     'quoteVolume':   float, // volume of quote currency traded for last 24 hours
# }
# ```
# 
# ### Notes On Ticker Structure
# - All fields in the ticker represent the past 24 hours prior to timestamp.
# - The bidVolume is the volume (amount) of current best bid in the orderbook.
# - The askVolume is the volume (amount) of current best ask in the orderbook.
# - The baseVolume is the amount of base currency traded (bought or sold) in last 24 hours.
# - The quoteVolume is the amount of quote currency traded (bought or sold) in last 24 hours.
# 
# All prices in ticker structure are in quote currency. Some fields in a returned ticker structure may be undefined/None/null.
# 
# BTC (Base Currency) / USDT (Quote Currency)
# 
# Timestamp and datetime are both UTC in milliseconds

# %% [markdown]
# # Singluar Ticker

# %%
eth_usdt = exchange.fetchTicker("ETH/USDT")
print(eth_usdt)

# %%
# Fetching Multiple
multiple_tickers = exchange.fetch_tickers(['ETH/BTC', 'LTC/BTC'])
print(multiple_tickers)

# %% [markdown]
# # OHLCV Candle Stick Charts
# 
# Most exchanges have endpoints for fetching OHLCV data, but some of them don't. The exchange boolean (true/false) property named has['fetchOHLCV'] indicates whether the exchange supports candlestick data series or not.
# 
# To fetch OHLCV candles/bars from an exchange, ccxt has the fetchOHLCV method, which is declared in the following way:
# 
# ```javascript
# fetchOHLCV (symbol, timeframe = '1m', since = undefined, limit = undefined, params = {})
# ```
# 
# ### OHLCV Structure
# 
# The fetchOHLCV method shown above returns a list (a flat array) of OHLCV candles represented by the following structure:
# 
# ```javascript
# [
#     [
#         1504541580000, // UTC timestamp in milliseconds, integer
#         4235.4,        // (O)pen price, float
#         4240.6,        // (H)ighest price, float
#         4230.0,        // (L)owest price, float
#         4230.7,        // (C)losing price, float
#         37.72941911    // (V)olume float (usually in terms of the base currency, the exchanges docstring may list whether quote or base units are used)
#     ],
#     ...
# ]
# ```
# 
# The list of candles is returned sorted in ascending (historical/chronological) order, oldest candle first, most recent candle last.

# %%
response = exchange.fetch_ohlcv('ADA/USDT', '1h')
print(response)

# %%
import pandas as pd
import mplfinance as mpf

df = pd.DataFrame(response[-50:], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

mpf.plot(df, 
         type='candle', 
         style='nightclouds',
         title='ADA/USDT 1H Candlestick Chart', 
         volume=True, 
         figsize=(12, 8))

# %% [markdown]
# # Funding Rate
# 
# **contract only**
# 
# Data on the current, most recent, and next funding rates can be obtained using the methods
# 
# - fetchFundingRates () for all market symbols
# - fetchFundingRates ([ symbol1, symbol2, ... ]) for multiple market symbols
# - fetchFundingRate (symbol) for a single market symbol
# 
# ### Funding Rate Structure
# 
# ```javascript
# {
#     info: { ... },
#     symbol: 'BTC/USDT:USDT',
#     markPrice: 39294.43,
#     indexPrice: 39291.78,
#     interestRate: 0.0003,
#     estimatedSettlePrice: undefined,
#     timestamp: undefined,
#     datetime: undefined,
#     fundingRate: 0.000072,
#     fundingTimestamp: 1645833600000,
#     fundingDatetime: '2022-02-26T00:00:00.000Z',
#     nextFundingRate: -0.000018, // nextFundingRate is actually two funding rates from now
#     nextFundingTimestamp: undefined,
#     nextFundingDatetime: undefined,
#     previousFundingRate: undefined,
#     previousFundingTimestamp: undefined,
#     previousFundingDatetime: undefined,
#     interval: '8h',
# }
# ```
# 

# %%
futures_exchange = ccxt.binance({
    'sandbox': False,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future'
    }
})

symbol = 'ETH/USDT'

funding_rate = futures_exchange.fetch_funding_rate(symbol)
print(f"Current Funding Rate for {symbol}:")
print(f"Rate: {funding_rate['fundingRate']:.6f}")
print(f"Symbol: {funding_rate['symbol']}")
funding_time = pd.to_datetime(funding_rate['timestamp'], unit='ms')
print(f"Funding Time: {funding_time}")



# %%
funding_rate_history = futures_exchange.fetch_funding_rate_history(symbol, limit=20)
print(f"\nRecent Funding Rate History for {symbol}:")

history_df = pd.DataFrame(funding_rate_history)
history_df['datetime'] = pd.to_datetime(history_df['timestamp'], unit='ms')
history_df['fundingRate'] = history_df['fundingRate'].astype(float)
print(history_df[['datetime', 'fundingRate']].round(6).to_string(index=False))

import matplotlib.pyplot as plt

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(history_df['datetime'], history_df['fundingRate'], 
        color='cyan', linewidth=2, marker='o', markersize=4)

ax.fill_between(history_df['datetime'], history_df['fundingRate'], 0, 
                where=(history_df['fundingRate'] >= 0), color='green', alpha=0.3, interpolate=True)
ax.fill_between(history_df['datetime'], history_df['fundingRate'], 0, 
                where=(history_df['fundingRate'] < 0), color='red', alpha=0.3, interpolate=True)

ax.set_title(f'{symbol} Funding Rate History', fontsize=16, color='white')
ax.set_xlabel('Date', fontsize=12, color='white')
ax.set_ylabel('Funding Rate', fontsize=12, color='white')
ax.axhline(y=0, color='white', linestyle='-', alpha=0.5)
ax.grid(True, alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Open Interest
# 
# **contract only**
# 
# Use the fetchOpenInterest method to get the current open interest for a symbol from the exchange.
# 
# ```javascript
# fetchOpenInterest (symbol, params = {})
# ```
# 
# ### Open Interest Structure
# 
# ```javascript
# {
#     symbol: 'BTC/USDT',
#     baseVolume: 80872.801, // deprecated
#     quoteVolume: 3508262107.38, // deprecated
#     openInterestAmount: 80872.801,
#     openInterestValue: 3508262107.38,
#     timestamp: 1649379000000,
#     datetime: '2022-04-08T00:50:00.000Z',
#     info: {
#         symbol: 'BTCUSDT',
#         sumOpenInterest: '80872.80100000',
#         sumOpenInterestValue: '3508262107.38000000',
#         timestamp: '1649379000000'
#     }
# }
# ```
# 

# %%
open_interest = futures_exchange.fetch_open_interest(symbol)
print(open_interest)

# %% [markdown]
# # Open Interest History
# 
# **contract only**
# 
# Use the fetchOpenInterestHistory method to get a history of open interest for a symbol from the exchange.
# 
# ```javascript
# fetchOpenInterestHistory (symbol, timeframe = '5m', since = undefined, limit = undefined, params = {})
# ```
# 
# Parameters
# 
# - symbol (String) Unified CCXT market symbol (e.g. "BTC/USDT:USDT")
# - timeframe (String) Check exchange.timeframes for available values
# - since (Integer) Timestamp for the earliest open interest record (e.g. 1645807945000)
# - limit (Integer) The maximum number of open interest structures to retrieve (e.g. 10)
# - params (Dictionary) Extra parameters specific to the exchange API endpoint (e.g. {"endTime": 1645807945000})
# 
# Note for OKX users: instead of a unified symbol okx.fetchOpenInterestHistory expects a unified currency code in the symbol argument (e.g. 'BTC').
# 
# Returns an array of open interest structures

# %%
oi_history = futures_exchange.fetch_open_interest_history(symbol, timeframe='1h', limit=100)

# Convert to DataFrame with improved processing
oi_df = pd.DataFrame(oi_history)
oi_df['datetime'] = pd.to_datetime(oi_df['timestamp'], unit='ms')

# Convert both amount and value to float, handling potential None values
oi_df['openInterestAmount'] = pd.to_numeric(oi_df['openInterestAmount'], errors='coerce')
oi_df['openInterestValue'] = pd.to_numeric(oi_df['openInterestValue'], errors='coerce')

# Format the value column for better readability (millions/billions)
oi_df['openInterestValue_M'] = oi_df['openInterestValue'] / 1_000_000

print(f"Open Interest History for {symbol}:")
print("Recent data (showing USDT value which matches Binance dashboard):")
display_df = oi_df[['datetime', 'openInterestValue_M', 'openInterestAmount']].tail(10).copy()
display_df.columns = ['DateTime', 'OI Value (M USDT)', 'OI Amount (Contracts)']
print(display_df.to_string(index=False, float_format='%.2f'))

# %%
# Set dark theme
plt.style.use('dark_background')

# Create figure with dual y-axes
fig, ax1 = plt.subplots(figsize=(14, 8))

# Plot Open Interest Amount (Contracts) on left y-axis
color1 = 'cyan'
ax1.set_xlabel('Time', fontsize=12, color='white')
ax1.set_ylabel('Open Interest (Contracts)', fontsize=12, color=color1)
line1 = ax1.plot(oi_df['datetime'], oi_df['openInterestAmount'], linewidth=2, color=color1, label='OI Amount (Contracts)')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.tick_params(axis='x', rotation=45)

# Format left y-axis to show values in millions if large
max_oi_amount = oi_df['openInterestAmount'].max()
if max_oi_amount > 1000000:
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))

# Create second y-axis for USDT value
ax2 = ax1.twinx()
color2 = 'orange'
ax2.set_ylabel('Open Interest Value (M USDT)', fontsize=12, color=color2)
line2 = ax2.plot(oi_df['datetime'], oi_df['openInterestValue_M'], linewidth=2, color=color2, label='OI Value (M USDT)')
ax2.tick_params(axis='y', labelcolor=color2)

# Add title and grid
plt.title(f'{symbol} - Open Interest History (Amount & Value)', fontsize=16, fontweight='bold', color='white', pad=20)
ax1.grid(True, alpha=0.3, color='gray')

# Add legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.8)

plt.tight_layout()
plt.show()

# Enhanced summary statistics for both metrics
print(f"\nSummary Statistics:")
print(f"{'='*50}")
print(f"Open Interest Amount (Contracts):")
print(f"  Current: {oi_df['openInterestAmount'].iloc[-1]:,.0f}")
print(f"  Average: {oi_df['openInterestAmount'].mean():,.0f}")
print(f"  Max: {oi_df['openInterestAmount'].max():,.0f}")
print(f"  Min: {oi_df['openInterestAmount'].min():,.0f}")
print(f"\nOpen Interest Value (USDT):")
print(f"  Current: ${oi_df['openInterestValue_M'].iloc[-1]:,.1f}M")
print(f"  Average: ${oi_df['openInterestValue_M'].mean():,.1f}M")
print(f"  Max: ${oi_df['openInterestValue_M'].max():,.1f}M")
print(f"  Min: ${oi_df['openInterestValue_M'].min():,.1f}M")

# %% [markdown]
# # Historical Volatility
# 
# **option only**
# 
# Use the fetchVolatilityHistory method to get the volatility history for the code of an options underlying asset from the exchange.
# 
# ```javascript
# fetchVolatilityHistory (code, params = {})
# ```
# 
# Parameters
# 
# - code (String) required Unified CCXT currency code (e.g. "BTC")
# - params (Dictionary) Extra parameters specific to the exchange API endpoint (e.g. {"endTime": 1645807945000})
# 
# Returns
# 
# - An array of volatility history structures
# 
# ### Volatility Structure
# 
# ```javascript
# {
#     info: {
#         "period": 7,
#         "value": "0.23854072",
#         "time": "1690574400000"
#     }
#     timestamp: 1649379000000,
#     datetime: '2023-07-28T00:50:00.000Z',
#     volatility: 0.23854072,
# }
# ```
# 

# %% [markdown]
# # Settlement History
# 
# **contract only**
# 
# Use the fetchSettlementHistory method to get the public settlement history for a contract market from the exchange.
# 
# ```javascript
# fetchSettlementHistory (symbol = undefined, since = undefined, limit = undefined, params = {})
# ```
# 
# Parameters
# 
# - symbol (String) Unified CCXT symbol (e.g. "BTC/USDT:USDT-230728-25500-P")
# - since (Integer) Timestamp for the earliest settlement (e.g. 1694073600000)
# - limit (Integer) The maximum number of settlements to retrieve (e.g. 10)
# - params (Dictionary) Extra parameters specific to the exchange API endpoint (e.g. {"endTime": 1645807945000})
# 
# Returns
# 
# - An array of settlement history structures
# 
# ```javascript
# {
#     info: { ... },
#     symbol: 'BTC/USDT:USDT-230728-25500-P',
#     price: 25761.35807869,
#     timestamp: 1694073600000,
#     datetime: '2023-09-07T08:00:00.000Z',
# }
# 
# ```

# %% [markdown]
# # Liquidations
# 
# Use the fetchLiquidations method to get the public liquidations of a trading pair from the exchange.
# 
# ```javascript
# fetchLiquidations (symbol, since = undefined, limit = undefined, params = {})
# ```
# 
# Parameters
# 
# - symbol (String) Unified CCXT symbol (e.g. "BTC/USDT:USDT-231006-25000-P")
# - since (Integer) Timestamp for the earliest liquidation (e.g. 1694073600000)
# - limit (Integer) The maximum number of liquidations to retrieve (e.g. 10)
# - params (Dictionary) Extra parameters specific to the exchange API endpoint (e.g. {"until": 1645807945000})
# 
# Returns 
# 
# - An array of liquidation structures
# 
# ### Liquidation Structures
# 
# ```javascript
# [
#     {
#         'info':          { ... },                        // the original decoded JSON as is
#         'symbol':        'BTC/USDT:USDT-231006-25000-P', // unified CCXT market symbol
#         'contracts':     2,                              // the number of derivative contracts
#         'contractSize':  0.001,                          // the contract size for the trading pair
#         'price':         27038.64,                       // the average liquidation price in the quote currency
#         'baseValue':     0.002,                          // value in the base currency (contracts * contractSize)
#         'quoteValue':    54.07728,                       // value in the quote currency ((contracts * contractSize) * price)
#         'timestamp':     1696996782210,                  // Unix timestamp in milliseconds
#         'datetime':      '2023-10-11 03:59:42.000',      // ISO8601 datetime with milliseconds
#     },
#     ...
# ]
# ```
# 
# 
# 

# %% [markdown]
# # Greeks
# 
# **option only**
# 
# Use the fetchGreeks method to get the public greeks and implied volatility of an options trading pair from the exchange. Use fetchAllGreeks to get the greeks for all symbols or multiple symbols. The greeks measure how factors like the underlying assets price, time to expiration, volatility, and interest rates, affect the price of an options contract.
# 
# ```javascript
# fetchGreeks (symbol, params = {})
# ```
# 
# Parameters
# 
# - symbol (String) Unified CCXT symbol (e.g. "BTC/USD:BTC-240927-40000-C")
# - params (Dictionary) Extra parameters specific to the exchange API endpoint (e.g. {"category": "options"})
# 
# Returns
# 
# - A Greeks structure
# 
# ```javascript
# {
#     'symbol': 'BTC/USD:BTC-240927-40000-C',     // unified CCXT market symbol
#     'timestamp': 1699593511632,                 // unix timestamp in milliseconds
#     'datetime': '2023-11-10T05:18:31.632Z',     // ISO8601 datetime with milliseconds
#     'delta': 0.59833,                           // measures the rate of change in the options price per $1 change in the underlying assets price
#     'gamma': 0.00002,                           // measures the rate of change in the delta per $1 change in the underlying assets price
#     'theta': -13.4441,                          // measures the dollar amount that an options price will decline per day
#     'vega': 142.30124,                          // measures the dollar amount that an options price changes with a 1% change in the implied volatility
#     'rho': 131.82621,                           // measures the dollar amount that an options price changes with a 1% change in interest rates
#     'vanna': 0.06671,                           // measures the amount that an options delta changes with a 1% change in implied volatility
#     'volga': 925.95015,                         // measures the amount that an options vega changes with a 1% change in implied volatility
#     'charm': 0.18433,                           // measures the amount that an options delta changes each day until expiration
#     'bidSize': 2.2,                             // the options bid amount
#     'askSize': 9,                               // the options ask amount
#     'bidImpliedVolatility': 60.06,              // the expected percentage price change of the underlying asset, over the remaining life of the option, calculated using the bid price
#     'askImpliedVolatility': 61.85,              // the expected percentage price change of the underlying asset, over the remaining life of the option, calculated using the ask price
#     'markImpliedVolatility': 60.86,             // the expected percentage price change of the underlying asset, over the remaining life of the option, calculated using the mark price
#     'bidPrice': 0.214,                          // the bid price of the option
#     'askPrice': 0.2205,                         // the ask price of the option
#     'markPrice': 0.2169,                        // the mark price of the option
#     'lastPrice': 0.215,                         // the last price of the option
#     'underlyingPrice': 39165.86,                // the current market price of the underlying asset
#     'info': { ... },                            // the original decoded JSON as is
# }
# 
# ```
# 

# %% [markdown]
# # Options Chain
# 
# **option only**
# 
# ## fetchOption Method
# 
# Use the `fetchOption` method to get the public details of a single option contract from the exchange.
# 
# ```javascript
# fetchOption (symbol, params = {})
# ```
# 
# ### Parameters
# 
# - **symbol** (String) - Unified CCXT market symbol (e.g. "BTC/USD:BTC-240927-40000-C")
# - **params** (Dictionary) - Extra parameters specific to the exchange API endpoint (e.g. {"category": "options"})
# 
# ### Returns
# 
# - An option chain structure
# 
# ---
# 
# ## fetchOptionChain Method
# 
# Use the `fetchOptionChain` method to get the public option chain data of an underlying currency from the exchange.
# 
# ```javascript
# fetchOptionChain (code, params = {})
# ```
# 
# ### Parameters
# 
# - **code** (String) - Unified CCXT currency code (e.g. "BTC")
# - **params** (Dictionary) - Extra parameters specific to the exchange API endpoint (e.g. {"category": "options"})
# 
# ### Returns
# 
# - A list of option chain structures
# 
# ---
# 
# ## Option Chain Structure
# 
# ```javascript
# {
#     'info': { ... },                            // the original decoded JSON as is
#     'currency': 'BTC',                          // unified CCXT currency code
#     'symbol': 'BTC/USD:BTC-240927-40000-C',     // unified CCXT market symbol
#     'timestamp': 1699593511632,                 // unix timestamp in milliseconds
#     'datetime': '2023-11-10T05:18:31.632Z',     // ISO8601 datetime with milliseconds
#     'impliedVolatility': 60.06,                 // the expected percentage price change of the underlying asset, over the remaining life of the option
#     'openInterest': 10,                         // the number of open options contracts that have not been settled
#     'bidPrice': 0.214,                          // the bid price of the option
#     'askPrice': 0.2205,                         // the ask price of the option
#     'midPrice': 0.2205,                         // the price in between the bid and the ask
#     'markPrice': 0.2169,                        // the mark price of the option
#     'lastPrice': 0.215,                         // the last price of the option
#     'underlyingPrice': 39165.86,                // the current market price of the underlying asset
#     'change': 15.43,                            // the 24 hour price change in a dollar amount
#     'percentage': 11.86,                        // the 24 hour price change as a percentage
#     'baseVolume': 100.86,                       // the volume in units of the base currency
#     'quoteVolume': 23772.86,                    // the volume in units of the quote currency
# }
# ```

# %% [markdown]
# # Long Short Ratio
# 
# **contract only**
# 
# Use the fetchLongShortRatio method to fetch the current long short ratio of a symbol and use the fetchLongShortRatioHistory to fetch the history of long short ratios for a symbol.
# 
# ## fetchLongShortRatio Method
# 
# Use the `fetchLongShortRatio` method for the current ratio of a single market symbol.
# 
# ```javascript
# fetchLongShortRatio (symbol, period = undefined, params = {})
# ```
# 
# ### Parameters
# 
# - **symbol** (String) - Required. Unified CCXT symbol (e.g. "BTC/USDT:USDT")
# - **period** (String) - The period to calculate the ratio from (e.g. "24h")
# - **params** (Dictionary) - Parameters specific to the exchange API endpoint (e.g. {"endTime": 1645807945000})
# 
# ### Returns
# 
# - A long short ratio structure
# 
# ---
# 
# ## fetchLongShortRatioHistory Method
# 
# Use the `fetchLongShortRatioHistory` method for the history of ratios of a single market symbol.
# 
# ```javascript
# fetchLongShortRatioHistory (symbol = undefined, period = undefined, since = undefined, limit = undefined, params = {})
# ```
# 
# ### Parameters
# 
# - **symbol** (String) - Unified CCXT symbol (e.g. "BTC/USDT:USDT")
# - **period** (String) - The period to calculate the ratio from (e.g. "24h")
# - **since** (Integer) - Timestamp for the earliest ratio data (e.g. 1694073600000)
# - **limit** (Integer) - The maximum number of ratio records to retrieve (e.g. 100)
# - **params** (Dictionary) - Parameters specific to the exchange API endpoint (e.g. {"endTime": 1645807945000})
# 
# ### Returns
# 
# - An array of long short ratio structures

# %%


