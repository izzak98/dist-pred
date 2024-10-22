"""Module for generating technical analysis features."""
import json
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy.stats import skew, kurtosis


def calculate_return(df, period) -> Series:
    """Calculate the simple return over a given period."""
    if period == 2:
        period = 1
    return df['Adj Close'].pct_change(period)


def calculate_log_return(df, period) -> Series:
    """Calculate the log return over a given period."""
    if period == 2:
        period = 1
    return np.log(df['Adj Close'] / df['Adj Close'].shift(period))


def calculate_cumulative_return(df, period) -> Series:
    """Calculate the cumulative return over a given period."""
    if period == 1:
        period = 2
    return (1 + df['Adj Close'].pct_change()).rolling(window=period).apply(np.prod, raw=True) - 1


def calculate_volatility(df, period) -> Series:
    """Calculate the rolling standard deviation of returns over a given period."""
    if period == 1:
        period = 2
    return df['Adj Close'].pct_change().rolling(window=period).std()


def calculate_skewness(df, period) -> Series:
    """Calculate the skewness of returns over a given period."""
    if period == 1:
        period = 2
    return df['Adj Close'].pct_change().rolling(window=period).apply(lambda x: skew(x), raw=True)


def calculate_kurtosis(df, period) -> Series:
    """Calculate the kurtosis of returns over a given period."""
    if period == 1:
        period = 2
    return df['Adj Close'].pct_change().rolling(window=period).apply(lambda x: kurtosis(x), raw=True)


def calculate_sma(df, period) -> Series:
    """Calculate the Simple Moving Average (SMA) over a given period."""
    return df['Adj Close'].rolling(window=period).mean()


def calculate_ema(df, period) -> Series:
    """Calculate the Exponential Moving Average (EMA) over a given period."""
    return df['Adj Close'].ewm(span=period, adjust=False).mean()


def calculate_rsi(df, period) -> Series:
    """
    Calculate the Relative Strength Index (RSI) over a given period.

    RSI = 100 - (100 / (1 + RS))

    RS = Average Gain / Average Loss
    """
    delta = df['Adj Close'].diff()  # Daily price changes
    gain = np.where(delta > 0, delta, 0)  # Separate gains
    loss = np.where(delta < 0, -delta, 0)  # Separate losses

    # Calculate the exponential moving averages for gains and losses
    avg_gain = Series(gain).ewm(
        span=period, adjust=False).mean().dropna() + 1e-10
    avg_loss = Series(loss).ewm(
        span=period, adjust=False).mean().dropna() + 1e-10

    # Calculate the Relative Strength (RS)
    rs = avg_gain / avg_loss

    # Calculate the RSI
    rsi = 100 - (100 / (1 + rs))
    rsi.index = df.index[-len(rsi):]  # Align the index

    return rsi


def calculate_macd(df) -> Series:
    """Calculate the Moving Average Convergence Divergence (MACD) over a given period."""
    short_ema = calculate_ema(df, 12)  # Standard MACD settings (12-day)
    long_ema = calculate_ema(df, 26)   # Standard MACD settings (26-day)
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(
        span=9, adjust=False).mean()  # Signal line (9-day)
    return macd_line - signal_line


def calculate_bollinger_bands(df, period) -> tuple[Series, Series]:
    """Calculate the Bollinger Bands over a given period."""
    if period == 1:
        period = 2
    sma = calculate_sma(df, period)
    std = df['Adj Close'].rolling(window=period).std()
    upper_band = sma + (2 * std)
    lower_band = sma - (2 * std)
    return upper_band, lower_band


def calculate_stochastic_oscillator(df, period) -> Series:
    """Calculate the Stochastic Oscillator over a given period."""
    if period == 1:
        period = 2
    lowest_low = df['Low'].rolling(window=period).min()
    highest_high = df['High'].rolling(window=period).max()
    return 100 * ((df['Adj Close'] - lowest_low) / (highest_high - lowest_low))


def calculate_vwap(df, period) -> Series:
    """
    Calculate the Volume-Weighted Average Price (VWAP) over a given period.

    VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)

    Typical Price = (High + Low + Close) / 3
    """
    if period == 1:
        period = 2
    typical_price = (df['High'] + df['Low'] + df['Adj Close']) / 3
    vwap = (typical_price * df['Volume']).rolling(window=period).sum() / \
        df['Volume'].rolling(window=period).sum()
    if any(vwap != vwap):
        vwap = vwap.fillna(0)
    return vwap


def calculate_sharpe_ratio(df, period) -> Series:
    """Calculate the Sharpe Ratio over a given period."""
    if period == 1:
        period = 2
    returns = df['Adj Close'].pct_change()
    mean_return = returns.rolling(window=period).mean()
    volatility = returns.rolling(window=period).std()
    # Simplified Sharpe Ratio (risk-free rate = 0)
    return mean_return / volatility


def generate_technical_features(df) -> DataFrame:
    """Generate technical features for a given DataFrame."""
    with open('config.json', encoding="utf-8") as f:
        config = json.load(f)
    periods = config['general']['periods']
    features = pd.DataFrame(index=df.index)

    for period in periods:
        features[f'return_{period}d'] = calculate_log_return(df, period)
        features[f'simple_return_{period}d'] = calculate_return(df, period)
        features[f'cumulative_return_{period}d'] = calculate_cumulative_return(
            df, period)
        features[f'volatility_{period}d'] = calculate_volatility(df, period)
        features[f'skewness_{period}d'] = calculate_skewness(df, period)
        features[f'kurtosis_{period}d'] = calculate_kurtosis(df, period)
        features[f'sma_{period}d'] = calculate_sma(df, period)
        features[f'ema_{period}d'] = calculate_ema(df, period)
        features[f'rsi_{period}d'] = calculate_rsi(df, period)
        features[f'upper_band_{period}d'], features[f'lower_band_{period}d'] = calculate_bollinger_bands(
            df, period)
        features[f'stoch_osc_{period}d'] = calculate_stochastic_oscillator(
            df, period)
        features[f'vwap_{period}d'] = calculate_vwap(df, period)
        features[f'sharpe_ratio_{period}d'] = calculate_sharpe_ratio(
            df, period)
    features['macd'] = calculate_macd(df)
    features = features.sort_index()
    return features.dropna()
