"""Module to generate synthetic data with a target correlation to market data."""
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from scipy.stats import pearsonr, gamma, lognorm, uniform

from utils.ta_utils import generate_technical_features


def generate_synthetic_data_with_target_correlation(
    market_data: ndarray,
    target_correlation: float,
    mu: float,
    sigma: float,
    distribution: str = 'normal'
) -> DataFrame:
    """
    Generate a synthetic dataset with returns that have a target correlation to the market_data.

    Parameters:
    market_data (numpy.ndarray): Log returns from the market data.
    target_correlation (float): The target correlation for synthetic data's
    log returns to the market data.
    mu (float): Mean of the synthetic dataset.
    sigma (float): Standard deviation of the synthetic dataset.
    distribution (str): The distribution to sample synthetic data from
    ('normal', 'gamma', 'lognormal', 'uniform').

    Returns:
        pd.DataFrame: DataFrame with synthetic prices.
    """

    num_periods = len(market_data)

    # Step 1: Generate independent noise based on the specified distribution
    if distribution == 'normal':
        noise = np.random.normal(loc=0, scale=1, size=num_periods)  # Standard normal noise
    elif distribution == 'gamma':
        shape = 2  # Shape parameter for the gamma distribution
        noise = gamma.rvs(a=shape, scale=1, size=num_periods)
        noise = (noise - np.mean(noise)) / np.std(noise)  # Standardize noise
    elif distribution == 'lognormal':
        noise = lognorm.rvs(s=1, scale=np.exp(0), size=num_periods)
        noise = (noise - np.mean(noise)) / np.std(noise)  # Standardize noise
    elif distribution == 'uniform':
        noise = uniform.rvs(loc=-1, scale=2, size=num_periods)  # Uniform distribution [-1, 1]
    else:
        raise ValueError(
            "Unsupported distribution. Choose from 'normal', 'gamma', 'lognormal', or 'uniform'.")

    # Step 2: Standardize market data
    market_std = (market_data - np.mean(market_data)) / np.std(market_data)

    # Step 3: Create a linear combination of market data and noise for correlated returns
    synthetic_log_returns = target_correlation * \
        market_std + np.sqrt(1 - target_correlation**2) * noise

    # Step 4: Adjust the synthetic returns to have the desired mean and standard deviation
    synthetic_log_returns = synthetic_log_returns * sigma + mu

    # Step 5: Convert synthetic log returns to price data
    initial_price = np.random.uniform(5, 1000)  # Random initial price
    synthetic_prices = initial_price * np.exp(np.cumsum(synthetic_log_returns))

    # Step 6: Create a DataFrame with synthetic price data
    df = pd.DataFrame(synthetic_prices, columns=['Adj Close'])

    # Add High, Low, Volume columns to allow all functions to work
    df['High'] = df['Adj Close'] * (1 + np.random.normal(0, 0.01, size=len(df)))
    df['Low'] = df['Adj Close'] * (1 - np.random.normal(0, 0.01, size=len(df)))
    df['Volume'] = 0

    df = generate_technical_features(df)

    return df


def generate_synthetic_data(
    num_periods: int,
    target_correlation: float,
    n_assets: int = 10,
    n_market_assets: int = 15,
    distribution: str = 'normal'
) -> tuple[list[DataFrame], DataFrame]:
    """Generate synthetic data for multiple assets with a target correlation to market data.

    Parameters:
        num_periods (int): Number of periods for the synthetic data.
        target_correlation (float): The target correlation for synthetic data's 
        log returns to the market data.
        n_assets (int): Number of synthetic assets to generate.
        n_market_assets (int): Number of market assets to generate.
        distribution (str): The distribution to sample synthetic data from
        ('normal', 'gamma', 'lognormal', 'uniform').

        Returns:
            tuple: A tuple containing:
                - list[DataFrame]: A list of DataFrames with synthetic asset prices.
                - DataFrame: DataFrame with market data prices."""
    # Sample log returns from market data
    market_mu = np.random.uniform(-0.001, 0.001)
    market_sigma = np.random.uniform(0.01, 0.03)

    # Example market data (log returns)
    market_data = np.random.normal(market_mu, market_sigma, num_periods)

    # Parameters for the synthetic data
    dates = pd.date_range(start='2020-01-01', periods=num_periods, freq='D', tz='UTC')

    synthetic_dfs = []
    for i in range(n_assets):
        mu = np.random.uniform(-0.001, 0.001)
        sigma = np.random.uniform(0.01, 0.03)
        print(f"Generated mu: {mu:.6f}, sigma: {sigma:.6f}")

        synthetic_df = generate_synthetic_data_with_target_correlation(
            market_data, target_correlation, mu, sigma, distribution)
        synthetic_df.index = dates[-len(synthetic_df):]
        synthetic_dfs.append(synthetic_df)

    market_datas = pd.DataFrame(market_data, columns=['1'], index=dates)
    market_datas = market_datas.loc[synthetic_df.index]

    for i, synthetic_df in enumerate(synthetic_dfs):
        correlation, _ = pearsonr(synthetic_df['return_2d'], market_datas["1"])
        print(f"Asset {i+1} correlation with market data: {correlation:.4f}")

    for _ in range(n_market_assets - 1):
        sub_market_mu = np.random.uniform(-0.001, 0.001)
        sub_market_sigma = np.random.uniform(0.01, 0.03)
        sub_market_data = np.random.normal(sub_market_mu, sub_market_sigma, len(market_datas))
        market_datas[str(len(market_datas.columns) + 1)] = sub_market_data

    return synthetic_dfs, market_datas


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    synth_df, market = generate_synthetic_data(
        1000, 0.5, distribution='uniform', n_assets=1, n_market_assets=1)
    plt.hist(synth_df[0]['return_2d'], bins=50, alpha=0.5, label='Synthetic')
    plt.title('Synthetic Data Distribution')
    plt.legend()
    plt.show()
