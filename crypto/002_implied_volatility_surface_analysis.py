# %% [markdown]
# # Implied Volatility Surface Analysis for Crypto Options
#
# This notebook demonstrates how to fetch real cryptocurrency options data using the CCXT library and construct an implied volatility surface. We'll analyze Bitcoin options from Deribit exchange and visualize the volatility surface
#
# The implied volatility surface shows how implied volatility varies across different strike prices and expiration dates, providing crucial insights for options trading and risk management.


# %% [markdown]
# ## Install Required Packages

# %%
# !pip install ccxt numpy pandas matplotlib seaborn scipy plotly

# %% [markdown]
# ## Import Libraries

# %%
import ccxt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import plotly.graph_objects as go
import warnings


# %% [markdown]
# ## Import Libraries and Configure Environment
#
# We import essential libraries for data fetching (ccxt), numerical computation (numpy, scipy), data manipulation (pandas), and visualization (matplotlib, plotly). We also set up the plotting environment with appropriate configurations for our analysis.

# %%
warnings.filterwarnings('ignore')

plt.style.use('dark_background')
np.random.seed(42)

# %% [markdown]
# ## Initialize Exchange Connection and Fetch Market Data
#
# We connect to the Deribit exchange, which is one of the largest crypto derivatives platforms. We'll fetch the current Bitcoin spot price and available options instruments. Deribit provides comprehensive options data including strikes, expirations, and real-time pricing.

# %%
exchange = ccxt.deribit({
    'sandbox': False,
    'enableRateLimit': True,
})

try:
    btc_ticker = exchange.fetch_ticker('BTC-PERPETUAL')
    spot_price = btc_ticker['last']
    print(f"Current BTC Price: ${spot_price:,.2f}")

    markets = exchange.load_markets()
    option_symbols = [symbol for symbol in markets.keys(
    ) if 'BTC' in symbol and ('-C' in symbol or '-P' in symbol)]
    print(f"Found {len(option_symbols)} BTC options")

except Exception as e:
    print(f"Error fetching live data: {e}")
    print("Using simulated data instead...")
    spot_price = 45000
    option_symbols = []

# %% [markdown]
# ## Generate Comprehensive Options Dataset
#
# Since live options data can be limited or unavailable, we create a realistic synthetic dataset that mirrors actual market conditions. This includes multiple expiration dates, various strike prices around the current spot price, and realistic bid-ask spreads with implied volatilities that follow typical market patterns.

# %%


def fetch_or_generate_options_data(spot_price, exchange):
    try:
        option_chain = exchange.fetchOptionChain('BTC')

        options_data = []
        for option in option_chain:
            if isinstance(option, dict) and 'symbol' in option:
                symbol_parts = option['symbol'].split('-')
                if len(symbol_parts) >= 4:
                    exp_str = symbol_parts[1]
                    strike = float(symbol_parts[2])
                    option_type = 'call' if symbol_parts[3].endswith(
                        'C') else 'put'

                    exp_date = pd.to_datetime(exp_str, format='%d%b%y')
                    days_to_exp = (exp_date - pd.Timestamp.now()).days
                    time_to_exp = max(1/365, days_to_exp / 365.0)

                    options_data.append({
                        'symbol': option['symbol'],
                        'type': option_type,
                        'strike': strike,
                        'expiration': exp_date,
                        'days_to_exp': days_to_exp,
                        'time_to_exp': time_to_exp,
                        'bid': option.get('bidPrice', 0),
                        'ask': option.get('askPrice', 0),
                        'mid_price': option.get('midPrice', (option.get('bidPrice', 0) + option.get('askPrice', 0)) / 2),
                        'implied_volatility': option.get('impliedVolatility', 0) / 100 if option.get('impliedVolatility') else None,
                        'moneyness': strike / spot_price,
                        'open_interest': option.get('openInterest', 0),
                        'mark_price': option.get('markPrice', 0),
                        'last_price': option.get('lastPrice', 0)
                    })

        if options_data:
            df = pd.DataFrame(options_data)
            df = df[df['days_to_exp'] > 0]
            print(f"Fetched {len(df)} live options contracts")
            return df

    except Exception as e:
        print(f"Error fetching live options data: {e}")
        print("Falling back to synthetic data...")

    options_data = []
    base_date = pd.Timestamp.now()
    expirations = [7, 14, 30, 60]
    strikes_per_exp = 15

    for days_to_exp in expirations:
        exp_date = base_date + pd.Timedelta(days=days_to_exp)
        time_to_exp = days_to_exp / 365.0

        strike_range = np.linspace(
            spot_price * 0.8, spot_price * 1.2, strikes_per_exp)

        for strike in strike_range:
            moneyness = strike / spot_price
            base_iv = 0.8 + 0.3 * abs(moneyness - 1) + \
                0.1 * np.sqrt(time_to_exp)
            implied_vol = max(0.2, base_iv + np.random.normal(0, 0.05))

            for option_type in ['call', 'put']:
                d1 = (np.log(spot_price/strike) + (0.05 + 0.5*implied_vol**2)
                      * time_to_exp) / (implied_vol*np.sqrt(time_to_exp))
                d2 = d1 - implied_vol*np.sqrt(time_to_exp)

                if option_type == 'call':
                    price = spot_price * \
                        norm.cdf(d1) - strike * \
                        np.exp(-0.05*time_to_exp)*norm.cdf(d2)
                else:
                    price = strike*np.exp(-0.05*time_to_exp) * \
                        norm.cdf(-d2) - spot_price*norm.cdf(-d1)

                spread = price * 0.02

                options_data.append({
                    'symbol': f'BTC-{exp_date.strftime("%d%b%y").upper()}-{int(strike)}-{option_type[0].upper()}',
                    'type': option_type,
                    'strike': strike,
                    'expiration': exp_date,
                    'days_to_exp': days_to_exp,
                    'time_to_exp': time_to_exp,
                    'bid': max(0.01, price - spread/2),
                    'ask': price + spread/2,
                    'mid_price': price,
                    'implied_volatility': implied_vol,
                    'moneyness': moneyness,
                    'open_interest': np.random.randint(1, 100),
                    'mark_price': price,
                    'last_price': price * (1 + np.random.normal(0, 0.01))
                })

    print(f"Generated {len(options_data)} synthetic options contracts")
    return pd.DataFrame(options_data)


options_df = fetch_or_generate_options_data(spot_price, exchange)
print(f"Total options contracts: {len(options_df)}")
print(f"Expiration dates: {sorted(options_df['days_to_exp'].unique())} days")
print(
    f"Strike range: ${options_df['strike'].min():,.0f} - ${options_df['strike'].max():,.0f}")

# %% [markdown]
# ## Process and Structure Options Data for Surface Construction
#
# We clean and organize the options data, focusing on liquid contracts and filtering out extreme outliers. We calculate key metrics like moneyness (strike/spot ratio) and ensure we have sufficient data points across different strikes and expirations to create a smooth volatility surface.

# %%
calls_df = options_df[options_df['type'] == 'call'].copy()
puts_df = options_df[options_df['type'] == 'put'].copy()

calls_df = calls_df[(calls_df['moneyness'] >= 0.85) &
                    (calls_df['moneyness'] <= 1.15)]
calls_df = calls_df[(calls_df['implied_volatility'] >= 0.1)
                    & (calls_df['implied_volatility'] <= 2.0)]

print("Options Data Summary:")
print(f"Total Call Options: {len(calls_df)}")
print(
    f"Expiration Range: {calls_df['days_to_exp'].min()} - {calls_df['days_to_exp'].max()} days")
print(
    f"Strike Range: ${calls_df['strike'].min():,.0f} - ${calls_df['strike'].max():,.0f}")
print(
    f"IV Range: {calls_df['implied_volatility'].min():.1%} - {calls_df['implied_volatility'].max():.1%}")

pivot_data = calls_df.pivot_table(
    values='implied_volatility',
    index='strike',
    columns='days_to_exp',
    aggfunc='mean'
).fillna(method='ffill').fillna(method='bfill')

print(f"\nVolatility Surface Dimensions: {pivot_data.shape}")
print("Days to Expiration:", list(pivot_data.columns))

# %% [markdown]
# ## Create Interactive 3D Volatility Surface Visualization
#
# We construct a three-dimensional visualization of the implied volatility surface using Plotly. The surface shows how implied volatility varies across strike prices (x-axis) and time to expiration (y-axis), with volatility levels represented by both height (z-axis) and color intensity using a cool color scheme.

# %%
strikes = pivot_data.index.values
expirations = pivot_data.columns.values
iv_surface = pivot_data.values

X, Y = np.meshgrid(expirations, strikes)
Z = iv_surface

fig = go.Figure(data=[go.Surface(
    x=X, y=Y, z=Z,
    colorscale='Viridis',
    colorbar=dict(
        title="Implied Volatility",
        titleside="right",
        tickmode="linear",
        tick0=Z.min(),
        dtick=(Z.max() - Z.min()) / 10
    ),
    hovertemplate='<b>Days to Exp:</b> %{x}<br><b>Strike:</b> $%{y:,.0f}<br><b>IV:</b> %{z:.1%}<extra></extra>'
)])

fig.update_layout(
    title={
        'text': 'Bitcoin Options Implied Volatility Surface',
        'x': 0.5,
        'font': {'size': 20, 'color': 'white'}
    },
    scene=dict(
        xaxis_title='Days to Expiration',
        yaxis_title='Strike Price ($)',
        zaxis_title='Implied Volatility',
        bgcolor='rgb(10, 10, 10)',
        xaxis=dict(gridcolor='rgb(50, 50, 50)', color='white'),
        yaxis=dict(gridcolor='rgb(50, 50, 50)', color='white'),
        zaxis=dict(gridcolor='rgb(50, 50, 50)', color='white')
    ),
    paper_bgcolor='rgb(10, 10, 10)',
    plot_bgcolor='rgb(10, 10, 10)',
    font=dict(color='white'),
    width=900,
    height=700
)

fig.show()

# %% [markdown]
# ## Generate 2D Volatility Smile Analysis
#
# We create traditional volatility smile plots for different expiration periods. These 2D cross-sections of the volatility surface show how implied volatility varies with moneyness (strike relative to spot price) for each expiration, revealing the characteristic smile or skew patterns in crypto options markets.

# %%
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Bitcoin Options Volatility Smiles by Expiration',
             fontsize=16, color='white', y=0.95)

colors = ['#00FFFF', '#0080FF', '#8000FF', '#FF0080']
expirations_to_plot = sorted(calls_df['days_to_exp'].unique())

for idx, days in enumerate(expirations_to_plot):
    row, col = idx // 2, idx % 2
    ax = axes[row, col]

    exp_data = calls_df[calls_df['days_to_exp']
                        == days].sort_values('moneyness')

    ax.plot(exp_data['moneyness'], exp_data['implied_volatility'],
            'o-', color=colors[idx], linewidth=2, markersize=6, alpha=0.8)

    ax.axvline(x=1.0, color='white', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Moneyness (Strike/Spot)', color='white', fontsize=11)
    ax.set_ylabel('Implied Volatility', color='white', fontsize=11)
    ax.set_title(f'{days} Days to Expiration', color='white', fontsize=12)
    ax.grid(True, alpha=0.3, color='gray')
    ax.tick_params(colors='white')

    for spine in ax.spines.values():
        spine.set_color('white')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Statistical Analysis and Surface Metrics
#
# We calculate key statistical measures of the volatility surface including term structure analysis, skew measurements, and surface curvature. These metrics help quantify the shape and characteristics of the volatility surface, providing insights into market sentiment and risk pricing across different strikes and expirations.

# %%
print("=== VOLATILITY SURFACE ANALYSIS ===\n")

atm_data = calls_df[abs(calls_df['moneyness'] - 1.0) <
                    0.02].groupby('days_to_exp')['implied_volatility'].mean()
print("At-The-Money Term Structure:")
for days, iv in atm_data.items():
    print(f"  {days:2d} days: {iv:.1%}")

print(
    f"\nTerm Structure Slope: {(atm_data.iloc[-1] - atm_data.iloc[0]) / (atm_data.index[-1] - atm_data.index[0]) * 365:.1%} per year")

skew_analysis = []
for days in calls_df['days_to_exp'].unique():
    day_data = calls_df[calls_df['days_to_exp'] == days]
    otm_put = day_data[day_data['moneyness']
                       < 0.95]['implied_volatility'].mean()
    atm = day_data[abs(day_data['moneyness'] - 1.0) <
                   0.02]['implied_volatility'].mean()
    otm_call = day_data[day_data['moneyness']
                        > 1.05]['implied_volatility'].mean()

    put_skew = otm_put - atm if not np.isnan(otm_put) else 0
    call_skew = otm_call - atm if not np.isnan(otm_call) else 0

    skew_analysis.append({
        'days': days,
        'put_skew': put_skew,
        'call_skew': call_skew,
        'total_skew': put_skew - call_skew
    })

skew_df = pd.DataFrame(skew_analysis)
print(f"\nVolatility Skew Analysis:")
print(f"Average Put Skew: {skew_df['put_skew'].mean():+.1%}")
print(f"Average Call Skew: {skew_df['call_skew'].mean():+.1%}")
print(f"Average Total Skew: {skew_df['total_skew'].mean():+.1%}")

surface_stats = {
    'min_iv': calls_df['implied_volatility'].min(),
    'max_iv': calls_df['implied_volatility'].max(),
    'mean_iv': calls_df['implied_volatility'].mean(),
    'iv_std': calls_df['implied_volatility'].std()
}

print(f"\nSurface Statistics:")
print(
    f"IV Range: {surface_stats['min_iv']:.1%} - {surface_stats['max_iv']:.1%}")
print(
    f"Mean IV: {surface_stats['mean_iv']:.1%} ± {surface_stats['iv_std']:.1%}")

# %% [markdown]
# ## Key Findings and Market Insights
#
# The analysis reveals several important characteristics of the Bitcoin options volatility surface:
#
# **Term Structure**: The volatility term structure shows how implied volatility changes with time to expiration, indicating market expectations of future volatility and mean reversion effects.
#
# **Volatility Skew**: The pronounced skew pattern reflects the market's pricing of tail risks, with out-of-the-money puts typically showing higher implied volatilities due to demand for downside protection.
#
# **Surface Shape**: The 3D surface visualization demonstrates the complex relationship between strike price, time to expiration, and implied volatility, which is crucial for options pricing, hedging strategies, and risk management in cryptocurrency markets.
#
# This volatility surface analysis provides traders and risk managers with essential insights for making informed decisions in the crypto options market, highlighting areas of relative value and potential trading opportunities.
