import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from pandas import DataFrame
from collections import defaultdict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.random.manual_seed(271198)
np.random.seed(271198)


def calculate_volatility_ewma(returns, decay_factor=0.94):
    squared_returns = returns ** 2
    # Use the ewm() method in pandas to calculate the exponentially weighted moving average
    ewma_volatility = squared_returns.ewm(span=(2/(1-decay_factor) - 1), adjust=False).mean()
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
                 end_date: str
                 ):
        self.datas = datas
        self.market_data = market_data
        self.start_date = pd.to_datetime(start_date, format="%Y-%m-%d", utc=True)
        self.end_date = pd.to_datetime(end_date, format="%Y-%m-%d", utc=True)
        self.normalization_lookback = normalization_lookback
        self.x = []
        self.s = []
        self.z = []
        self.y = []
        self.cat = []

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
                while True:
                    lookforward = np.random.randint(1, 30)
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

                    assert len(x) == len(y) == len(z)
                    assert len(s.shape) == 0
                    normalized_df = normalized_df.iloc[lookforward:]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx].values, dtype=torch.float32).to(DEVICE)
        s = torch.tensor(self.s[idx], dtype=torch.float32).to(DEVICE).view(-1, 1)
        sub_z = self.market_data.loc[self.z[idx]]
        z = torch.tensor(sub_z.values, dtype=torch.float32).to(DEVICE)
        y = torch.tensor(self.y[idx].values, dtype=torch.float32).to(DEVICE).view(-1, 1) * 100
        sy = (y/s)/100
        if not isinstance(self.cat[idx], torch.Tensor):
            self.cat[idx] = torch.tensor(self.cat[idx], dtype=torch.float32).to(DEVICE)
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
            bucket = min(int((length - min_len) / bucket_width), self.num_buckets - 1)
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


def get_dataset(nomalization_lookback: int, start_date: str, end_date: str):
    folders = os.listdir('data')
    folders.pop(folders.index('market_data'))
    datas = {}
    for folder in folders:
        datas[folder] = []
        for file in os.listdir(os.path.join('data', folder)):
            data = pd.read_csv(os.path.join('data', folder, file), index_col=0, parse_dates=True)
            datas[folder].append({"asset": file.split('.')[0], "data": data})
    market_data = pd.read_csv(os.path.join('data', 'market_data',
                              'market_data.csv'), index_col=0, parse_dates=True)
    dataset = Dist_Dataset(datas, market_data, nomalization_lookback, start_date, end_date)
    return dataset


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = get_dataset(100, "2000-01-01", "2021-01-01")
    batch_sampler = DynamicBatchSampler(dataset, batch_size=32)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)

    for x, s, z, y, sy in dataloader:
        print(x.shape, s.shape, z.shape, y.shape, sy.shape)
        # Use the data...
        # break
