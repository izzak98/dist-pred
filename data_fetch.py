"""Module to fetch the data from Yahoo Finance and save it in the data folder."""

import os
import json
import pandas as pd
from pandas import DataFrame
import yfinance as yf
from tqdm import tqdm

from utils.ta_utils import generate_technical_features


def fetch_assets(tickers: list[str], start_date: str, end_date: str) -> DataFrame:
    """Fetches the data for the given tickers from Yahoo Finance."""
    if not tickers:
        raise ValueError("The tickers list cannot be empty.")
    data = yf.download(tickers, start=start_date, end=end_date, interval='1d')
    return data


def init_folders() -> None:
    """Initializes the folders where the data will be stored."""
    with open('config.json', encoding="utf8") as f:
        config = json.load(f)
    if not os.path.exists('data'):
        os.makedirs('data')
    groupings = config['tickers']['assets'].keys()
    for grouping in groupings:
        if not os.path.exists(os.path.join('data', grouping)):
            os.makedirs(os.path.join('data', grouping))
    if not os.path.exists(os.path.join('data', 'market_data')):
        os.makedirs(os.path.join('data', 'market_data'))


def main() -> None:
    """
    Main function to fetch the data from Yahoo Finance.

    The data is fetched for the assets and market data specified in the config.json file.
    After fetching the data, the technical features are generated and saved in the data folder.

    Raises:
        ValueError: If no data is found for the tickers.
        ValueError: If the index of the data is not a DatetimeIndex.
    """
    init_folders()
    with open('config.json', encoding="utf8") as f:
        config = json.load(f)
    start_date = "2000-01-01"
    end_date = config['general']['dates']['test_period']['end_date']
    cols = ["Adj Close", "High", "Low", "Open", "Volume"]
    for grouping in config['tickers']['assets'].keys():
        tickers = config['tickers']['assets'][grouping]
        data = fetch_assets(tickers, start_date, end_date)
        if data is None:
            raise ValueError(f"No data found for the tickers: {tickers}")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("The index of the data should be a DatetimeIndex.")
        if data.index.tz is not None:
            data.index = data.index.tz_convert("UTC")
        save_path = os.path.join('data', grouping)
        for ticker in tqdm(tickers, desc=f"Processing {grouping}"):
            df = pd.DataFrame()
            for col in cols:
                df[col] = data[col][ticker]
            df = df.dropna()
            df = generate_technical_features(df)
            df.to_csv(os.path.join(save_path, f"{ticker}.csv"))

    market_data = config['tickers']['market_data']
    data = fetch_assets(market_data, start_date, end_date)
    data = data["Adj Close"].dropna()
    save_path = os.path.join('data', 'market_data')
    data.to_csv(os.path.join(save_path, "market_data.csv"))


if __name__ == "__main__":
    main()
