import os
import json
import pandas as pd
from pandas import DataFrame
import yfinance as yf
from tqdm import tqdm

from ta_utils import generate_technical_features


def fetch_assets(tickers, start_date, end_date) -> DataFrame:
    data = yf.download(tickers, start=start_date, end=end_date, interval='1d')
    return data


def init_folders() -> None:
    with open('config.json') as f:
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
    init_folders()
    with open('config.json') as f:
        config = json.load(f)
    start_date = "2000-01-01"
    end_date = config['general']['dates']['test_period']['end_date']
    cols = ["Adj Close", "High", "Low", "Open", "Volume"]
    for grouping in config['tickers']['assets'].keys():
        tickers = config['tickers']['assets'][grouping]
        data = fetch_assets(tickers, start_date, end_date)
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
