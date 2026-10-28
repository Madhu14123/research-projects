import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('BNBUSD_1m_Binance.csv')

# Parse 'Open time' as datetime
df['Open time'] = pd.to_datetime(df['Open time'])

# Set datetime index for resampling
df.set_index('Open time', inplace=True)

# Resample to hourly level
hourly = df.resample('h').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum',
    'Quote asset volume': 'sum',
    'Number of trades': 'sum',
    'Taker buy base asset volume': 'sum',
    'Taker buy quote asset volume': 'sum'
})

# Reset index to bring 'Open time' back as a column
hourly = hourly.reset_index()

# Rename for consistency (optional)
hourly.rename(columns={
    'Quote asset volume': 'QuoteVolume',
    'Number of trades': 'Trades',
    'Taker buy base asset volume': 'Taker buy base asset volume_sum',
    'Taker buy quote asset volume': 'Taker buy quote asset volume_sum'
}, inplace=True)

# Feature Engineering
clean_data = hourly.copy()

# Extract time-based features
clean_data['hour'] = clean_data['Open time'].dt.hour
clean_data['dayofweek'] = clean_data['Open time'].dt.dayofweek
clean_data['month'] = clean_data['Open time'].dt.month
clean_data['is_weekend'] = clean_data['dayofweek'] >= 5

# Cyclical encoding
clean_data['hour_sin'] = np.sin(2 * np.pi * clean_data['hour'] / 24)
clean_data['hour_cos'] = np.cos(2 * np.pi * clean_data['hour'] / 24)
clean_data['dayofweek_sin'] = np.sin(2 * np.pi * clean_data['dayofweek'] / 7)
clean_data['dayofweek_cos'] = np.cos(2 * np.pi * clean_data['dayofweek'] / 7)
clean_data['month_sin'] = np.sin(2 * np.pi * clean_data['month'] / 12)
clean_data['month_cos'] = np.cos(2 * np.pi * clean_data['month'] / 12)

# Price Change
clean_data['price_change'] = clean_data['Close'] - clean_data['Open']
clean_data['price_change_pct'] = (clean_data['Close'] - clean_data['Open']) / clean_data['Open']

# Moving Averages
clean_data['MA_5'] = clean_data['Close'].rolling(window=5).mean()
clean_data['MA_15'] = clean_data['Close'].rolling(window=15).mean()
clean_data['EMA_10'] = clean_data['Close'].ewm(span=10, adjust=False).mean()

# RSI
delta = clean_data['Close'].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(window=14).mean()
avg_loss = pd.Series(loss).rolling(window=14).mean()
rs = avg_gain / avg_loss
clean_data['RSI_14'] = 100 - (100 / (1 + rs))

# MACD
ema12 = clean_data['Close'].ewm(span=12, adjust=False).mean()
ema26 = clean_data['Close'].ewm(span=26, adjust=False).mean()
clean_data['MACD'] = ema12 - ema26

# Volatility
clean_data['Volatility_10'] = clean_data['Close'].rolling(window=10).std()

# Bollinger Bands Width
rolling_mean = clean_data['Close'].rolling(window=20).mean()
rolling_std = clean_data['Close'].rolling(window=20).std()
clean_data['BB_width'] = 2 * rolling_std

# Lag features
clean_data['Close_lag1'] = clean_data['Close'].shift(1)
clean_data['Close_lag2'] = clean_data['Close'].shift(2)
clean_data['Close_lag3'] = clean_data['Close'].shift(3)

# Momentum
clean_data['Momentum_5'] = clean_data['Close'] - clean_data['Close'].shift(5)

# Candle Direction
clean_data['Candle'] = np.where(clean_data['Close'] > clean_data['Open'], 1,
                         np.where(clean_data['Close'] < clean_data['Open'], -1, 0))

# Open/High/Low min-max in 5-hour rolling window
for col in ['Open', 'High', 'Low']:
    clean_data[f'{col}_min'] = clean_data[col].rolling(window=5).min()
    clean_data[f'{col}_max'] = clean_data[col].rolling(window=5).max()

# Save final DataFrame to CSV
clean_data.to_csv('engineered_output.csv', index=False)
