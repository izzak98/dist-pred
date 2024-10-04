import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from pandas import DataFrame
from collections import defaultdict

from utils.synthetic_utils import generate_synthetic_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.random.manual_seed(271198)
np.random.seed(271198)


def calculate_volatility_ewma(returns, decay_factor=0.94):
    squared_returns = returns ** 2
    # Use the ewm() method in pandas to calculate the exponentially weighted moving average
    ewma_volatility = squared_returns.ewm(
        span=(2/(1-decay_factor) - 1), adjust=False).mean()
    return ewma_volatility


def cross_sectional_volatility(groupings: list[dict[str, DataFrame]], decay_factor=0.94):
    df_returns = pd.DataFrame()
    for grouping in groupings:
        df_returns[grouping["asset"]] = grouping["data"]["return_2d"]
    stock_volatilities = df_returns.apply(
        lambda returns: calculate_volatility_ewma(returns, decay_factor))

    # Compute the cross-sectional average volatility (mean volatility across all stocks at each period)
    cross_sectional_avg_volatility = stock_volatilities.mean(axis=1)

    cross_sectional_avg_volatility.index = df_returns.index

    return cross_sectional_avg_volatility * np.sqrt(252)


class Dist_Dataset(Dataset):
    def __init__(self,
                 datas: dict[str, list[dict[str, DataFrame]]],
                 market_data: DataFrame,
                 normalization_lookback: int,
                 start_date: str,
                 end_date: str,
                 lookahead: int = None
                 ):
        self.datas = datas
        self.market_data = market_data
        self.start_date = pd.to_datetime(
            start_date, format="%Y-%m-%d", utc=True)
        self.end_date = pd.to_datetime(end_date, format="%Y-%m-%d", utc=True)
        self.normalization_lookback = normalization_lookback
        self.x = []
        self.s = []
        self.z = []
        self.y = []
        self.cat = []
        self.asset_name = []

        self.lookahead = lookahead

        self.gen_data()

    def gen_data(self):
        for grouping in self.datas.keys():
            cross_vol = cross_sectional_volatility(self.datas[grouping])
            for data in self.datas[grouping]:
                asset_name = data["asset"]
                df = data["data"]

                shared_index = list(set(df.index) & set(cross_vol.index)
                                    & set(self.market_data.index))
                df = df.loc[shared_index]
                df = df.sort_index()
                returns = df["return_2d"]
                rolling_mean = df.rolling(
                    window=self.normalization_lookback).mean()
                rolling_std = df.rolling(
                    window=self.normalization_lookback).std()
                normalized_df = (df - rolling_mean) / rolling_std
                normalized_df = normalized_df.iloc[self.normalization_lookback:]
                normalized_df = normalized_df.fillna(0)

                start_date = max(normalized_df.index[0], min(
                    normalized_df.index, key=lambda x: abs(x - self.start_date)))
                end_date = min(
                    normalized_df.index[-1], min(normalized_df.index, key=lambda x: abs(x - self.end_date)))

                normalized_df = normalized_df.loc[start_date:end_date]
                while True:
                    if self.lookahead is None:
                        lookforward = np.random.randint(1, 30)
                    else:
                        lookforward = self.lookahead
                    if len(normalized_df) < 5:
                        break
                    if len(normalized_df) < lookforward*2:
                        lookforward = len(normalized_df)//2
                    y_index = normalized_df.iloc[lookforward:lookforward*2].index
                    x = normalized_df.iloc[0:lookforward]
                    s = cross_vol.loc[x.index].mean()
                    z = x.index
                    y = returns.loc[y_index]

                    cat = [0] * len(self.datas.keys())
                    cat[list(self.datas.keys()).index(grouping)] = 1

                    self.cat.append(cat)

                    self.x.append(x)
                    self.s.append(s)
                    self.z.append(z)
                    self.y.append(y)
                    self.asset_name.append(asset_name)

                    assert len(x) == len(y) == len(z)
                    assert len(s.shape) == 0
                    normalized_df = normalized_df.iloc[lookforward:]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx].values, dtype=torch.float32).to(DEVICE)
        s = torch.tensor(self.s[idx], dtype=torch.float32).to(
            DEVICE).view(-1, 1)
        sub_z = self.market_data.loc[self.z[idx]]
        z = torch.tensor(sub_z.values, dtype=torch.float32).to(DEVICE)
        y = torch.tensor(self.y[idx].values, dtype=torch.float32).to(
            DEVICE).view(-1, 1) * 100
        sy = (y/s)/100
        if not isinstance(self.cat[idx], torch.Tensor):
            self.cat[idx] = torch.tensor(
                self.cat[idx], dtype=torch.float32).to(DEVICE)
        x = torch.cat((x, self.cat[idx].repeat(x.size(0), 1)), dim=1)
        z = torch.cat((z, self.cat[idx].repeat(z.size(0), 1)), dim=1)

        return x, s, z, y, sy


class StaticDistDataset(Dataset):
    def __init__(self, datas: dict[str, list[dict[str, DataFrame]]], market_data: DataFrame, normalization_lookback: int, start_date: str, end_date: str):
        self.datas = datas
        self.market_data = market_data
        self.normalization_lookback = normalization_lookback
        self.start_date = pd.to_datetime(
            start_date, format="%Y-%m-%d", utc=True)
        self.end_date = pd.to_datetime(end_date, format="%Y-%m-%d", utc=True)
        self.x = []
        self.s = []
        self.z = []
        self.y = []
        self.cat = []
        self.asset_name = []

        self.gen_data()

    def gen_data(self):
        for grouping in self.datas.keys():
            cross_vol = cross_sectional_volatility(self.datas[grouping])
            for data in self.datas[grouping]:
                asset_name = data["asset"]
                df = data["data"]

                shared_index = list(set(df.index) & set(cross_vol.index)
                                    & set(self.market_data.index))
                df = df.loc[shared_index]
                df = df.sort_index()
                returns = df["return_2d"]
                rolling_mean = df.rolling(
                    window=self.normalization_lookback).mean()
                rolling_std = df.rolling(
                    window=self.normalization_lookback).std()
                normalized_df = (df - rolling_mean) / rolling_std
                normalized_df = normalized_df.iloc[self.normalization_lookback:]
                normalized_df = normalized_df.fillna(0)

                start_date = max(normalized_df.index[0], min(
                    normalized_df.index, key=lambda x: abs(x - self.start_date)))
                end_date = min(
                    normalized_df.index[-1], min(normalized_df.index, key=lambda x: abs(x - self.end_date)))

                normalized_df = normalized_df.loc[start_date:end_date]
                for i in range(0, len(normalized_df)-22, 22):
                    date = normalized_df.index[i]
                    x = normalized_df.loc[date]
                    s = cross_vol.loc[date]
                    z = date
                    y = returns.loc[normalized_df.iloc[i+22].name]

                    cat = [0] * len(self.datas.keys())
                    cat[list(self.datas.keys()).index(grouping)] = 1

                    self.cat.append(cat)

                    self.x.append(x)
                    self.s.append(s)
                    self.z.append(z)
                    self.y.append(y)
                    self.asset_name.append(asset_name)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx].values, dtype=torch.float32).to(DEVICE)
        s = torch.tensor(self.s[idx], dtype=torch.float32).to(DEVICE)
        sub_z = self.market_data.loc[self.z[idx]]
        z = torch.tensor(sub_z.values, dtype=torch.float32).to(DEVICE)
        y = torch.tensor(self.y[idx], dtype=torch.float32).to(DEVICE) * 100
        sy = (y/s)/100
        if not isinstance(self.cat[idx], torch.Tensor):
            self.cat[idx] = torch.tensor(
                self.cat[idx], dtype=torch.float32).to(DEVICE)
        x = torch.cat((x, self.cat[idx]))
        z = torch.cat((z, self.cat[idx]))
        return x, s.view(1), z, y.view(1), sy.view(1)


def get_grouping(outer_dict, inner_str):
    # Iterate over the outer dict
    for outer_key, list_of_dicts in outer_dict.items():
        # Iterate over the list of dictionaries
        for inner_dict in list_of_dicts:
            # Check if the inner string is in this inner dictionary
            if inner_str in inner_dict["asset"]:
                return outer_key
    # Return None or handle the case if not found
    return None


class TestDataset(Dataset):
    def __init__(self, datas: dict[str, list[dict[str, DataFrame]]], market_data: DataFrame, normalization_lookback: int, start_date: str, end_date: str, lookforward: int):
        super().__init__()
        self.datas = datas
        self.market_data = market_data
        self.normalization_lookback = normalization_lookback
        self.start_date = pd.to_datetime(
            start_date, format="%Y-%m-%d", utc=True)
        self.end_date = pd.to_datetime(end_date, format="%Y-%m-%d", utc=True)
        self.assets = [asset["asset"]
                       for group in datas.values() for asset in group]
        self.main_asset = self.assets[0]

        self.x = []
        self.s = []
        self.z = []
        self.y = []
        self.cat = []

        self.lookforward = lookforward

        self.gen_data()

    def set_main_asset(self, asset):
        self.main_asset = asset
        self.gen_data()

    def gen_data(self):
        self.x = []
        self.s = []
        self.z = []
        self.y = []
        self.cat = []

        grouping = get_grouping(self.datas, self.main_asset)
        cross_vol = cross_sectional_volatility(self.datas[grouping])

        data = [asset for asset in self.datas[grouping]
                if asset["asset"] == self.main_asset][0]["data"]

        shared_index = list(set(data.index) & set(cross_vol.index)
                            & set(self.market_data.index))

        df = data.loc[shared_index]
        df = df.sort_index()
        returns = df["return_2d"]
        rolling_mean = df.rolling(window=self.normalization_lookback).mean()
        rolling_std = df.rolling(window=self.normalization_lookback).std()
        normalized_df = (df - rolling_mean) / rolling_std
        normalized_df = normalized_df.iloc[self.normalization_lookback:]
        normalized_df = normalized_df.fillna(0)

        start_date = max(normalized_df.index[0], min(
            normalized_df.index, key=lambda x: abs(x - self.start_date)))
        end_date = min(
            normalized_df.index[-1], min(normalized_df.index, key=lambda x: abs(x - self.end_date)))

        normalized_df = normalized_df.loc[start_date:end_date]

        for i in range(0, len(normalized_df)-self.lookforward):
            x = normalized_df.iloc[i:i+self.lookforward]
            s = cross_vol.loc[x.index]
            z = x.index
            y = returns.loc[x.index]

            cat = [0] * len(self.datas.keys())
            cat[list(self.datas.keys()).index(grouping)] = 1

            self.cat.append(cat)

            self.x.append(x)
            self.s.append(s)
            self.z.append(z)
            self.y.append(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx].values, dtype=torch.float32).to(DEVICE)
        s = torch.tensor(self.s[idx], dtype=torch.float32).to(
            DEVICE).view(-1, 1)
        sub_z = self.market_data.loc[self.z[idx]]
        z = torch.tensor(sub_z.values, dtype=torch.float32).to(DEVICE)
        y = torch.tensor(self.y[idx].values, dtype=torch.float32).to(
            DEVICE).view(-1, 1) * 100
        sy = (y/s)/100
        if not isinstance(self.cat[idx], torch.Tensor):
            self.cat[idx] = torch.tensor(
                self.cat[idx], dtype=torch.float32).to(DEVICE)
        x = torch.cat((x, self.cat[idx].repeat(x.size(0), 1)), dim=1)
        z = torch.cat((z, self.cat[idx].repeat(z.size(0), 1)), dim=1)

        return x, s, z, y, sy


class DynamicBatchSampler:
    def __init__(self, dataset, batch_size, num_buckets=10):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_buckets = num_buckets
        self.buckets = defaultdict(list)
        self._create_buckets()

    def _create_buckets(self):
        lengths = [len(self.dataset[i][0]) for i in range(len(self.dataset))]
        min_len, max_len = min(lengths), max(lengths)
        bucket_width = (max_len - min_len) / self.num_buckets

        for idx, length in enumerate(lengths):
            bucket = min(int((length - min_len) / bucket_width),
                         self.num_buckets - 1)
            self.buckets[bucket].append(idx)

    def __iter__(self):
        for bucket in self.buckets.values():
            np.random.shuffle(bucket)
            for i in range(0, len(bucket), self.batch_size):
                yield bucket[i:i + self.batch_size]

    def __len__(self):
        return sum(len(bucket) // self.batch_size + (1 if len(bucket) % self.batch_size else 0)
                   for bucket in self.buckets.values())


def collate_fn(batch):
    x, s, z, y, sy = zip(*batch)

    # Find max length in the batch
    max_len = max(seq.size(0) for seq in x)

    # Pad sequences to max_len
    x_padded = torch.stack([torch.nn.functional.pad(
        seq, (0, 0, 0, max_len - seq.size(0))) for seq in x])
    z_padded = torch.stack([torch.nn.functional.pad(
        seq, (0, 0, 0, max_len - seq.size(0))) for seq in z])
    y_padded = torch.stack([torch.nn.functional.pad(
        seq, (0, 0, 0, max_len - seq.size(0))) for seq in y])
    sy_padded = torch.stack([torch.nn.functional.pad(
        seq, (0, 0, 0, max_len - seq.size(0))) for seq in sy])
    s_padded = torch.stack(s).view(-1, 1)

    return x_padded, s_padded, z_padded, y_padded, sy_padded


def get_dataset(nomalization_lookback: int, start_date: str, end_date: str, lookahead: int = None):
    folders = os.listdir('data')
    folders.pop(folders.index('market_data'))
    datas = {}
    for folder in folders:
        datas[folder] = []
        for file in os.listdir(os.path.join('data', folder)):
            data = pd.read_csv(os.path.join(
                'data', folder, file), index_col=0, parse_dates=True)
            datas[folder].append({"asset": file.split('.')[0], "data": data})
    market_data = pd.read_csv(os.path.join('data', 'market_data',
                              'market_data.csv'), index_col=0, parse_dates=True)
    dataset = Dist_Dataset(
        datas, market_data, nomalization_lookback, start_date, end_date, lookahead)
    return dataset


def get_static_dataset(nomalization_lookback: int, start_date: str, end_date: str):
    folders = os.listdir('data')
    folders.pop(folders.index('market_data'))
    datas = {}
    for folder in folders:
        datas[folder] = []
        for file in os.listdir(os.path.join('data', folder)):
            data = pd.read_csv(os.path.join(
                'data', folder, file), index_col=0, parse_dates=True)
            datas[folder].append({"asset": file.split('.')[0], "data": data})
    market_data = pd.read_csv(os.path.join('data', 'market_data',
                              'market_data.csv'), index_col=0, parse_dates=True)
    dataset = StaticDistDataset(
        datas, market_data, nomalization_lookback, start_date, end_date)
    return dataset


def get_test_dataset(nomalization_lookback: int, start_date: str, end_date: str, lookforward: int = 22):
    folders = os.listdir('data')
    folders.pop(folders.index('market_data'))
    datas = {}
    for folder in folders:
        datas[folder] = []
        for file in os.listdir(os.path.join('data', folder)):
            data = pd.read_csv(os.path.join(
                'data', folder, file), index_col=0, parse_dates=True)
            datas[folder].append({"asset": file.split('.')[0], "data": data})
    market_data = pd.read_csv(os.path.join('data', 'market_data',
                              'market_data.csv'), index_col=0, parse_dates=True)
    dataset = TestDataset(
        datas, market_data, nomalization_lookback, start_date, end_date, lookforward)
    return dataset


def get_test_synthetic_dataset(dense_normalization_lookback: int,
                               lstm_normalization_lookback: int,
                               num_periods: int,
                               target_correlation: float,
                               n_assets: int = 10,
                               n_market_assets: int = 15,
                               ):
    dense_num_periods = num_periods + dense_normalization_lookback + 22
    lstm_num_periods = num_periods + lstm_normalization_lookback + 22

    num_periods = max(dense_num_periods, lstm_num_periods)

    synthetic_dfs, market_datas = generate_synthetic_data(
        num_periods, target_correlation, n_assets, n_market_assets)
    
    datas = {"synthetic": []}
    folders = os.listdir('data')
    folders.pop(folders.index('market_data'))
    cats = len(folders)
    for i in range(cats-1):
        datas[f"cat_{i}"] = []
    for i, df in enumerate(synthetic_dfs):
        datas["synthetic"].append({"asset": f"synthetic_{i}", "data": df})
    market_data = market_datas
    dense_dataset = TestDataset(
        datas, market_data, dense_normalization_lookback, "2020-01-01", "2021-01-01", 22)
    lstm_dataset = TestDataset(
        datas, market_data, lstm_normalization_lookback, "2020-01-01", "2021-01-01", 22)
    
    return dense_dataset, lstm_dataset


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    # dataset = get_static_dataset(100, "2000-01-01", "2021-01-01")
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # dataset = get_dataset(100, "2000-01-01", "2021-01-01", 22)
    # batch_sampler = DynamicBatchSampler(dataset, batch_size=32)
    # dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)

    # dataset = get_test_dataset(100, "2000-01-01", "2021-01-01", 22)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    dataset = get_test_synthetic_dataset(100, 1000, 0.5)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    for x, s, z, y, sy in dataloader:
        print(x.shape, s.shape, z.shape, y.shape, sy.shape)
        # Use the data...
        # break
