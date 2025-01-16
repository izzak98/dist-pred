"""Module for loading and processing financial data."""
import json
import os
from typing import Optional, Union, Generator
from collections import defaultdict
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import pandas as pd
from pandas import DataFrame, Series

from utils.synthetic_utils import generate_synthetic_data

with open('config.json', encoding="utf8") as f:
    CONFIG = json.load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.random.manual_seed(CONFIG["general"]["seed"])
np.random.seed(CONFIG["general"]["seed"])


def calculate_volatility_ewma(returns, decay_factor=0.94) -> Series:
    """
    Calculate the exponentially weighted moving average (EWMA) volatility of a series of returns.

    Parameters:
        returns (pd.Series): A pandas Series of returns.
        decay_factor (float): The decay factor for the EWMA calculation. Default is 0.94.

    Returns:
        pd.Series: A pandas Series representing the EWMA volatility.
    """
    squared_returns = returns ** 2
    # Use the ewm() method in pandas to calculate the exponentially weighted moving average
    ewma_volatility = squared_returns.ewm(
        span=(2/(1-decay_factor) - 1), adjust=False).mean()
    return ewma_volatility


def cross_sectional_volatility(groupings: list[dict[str, DataFrame]], decay_factor=0.94):
    """
    Calculate the cross-sectional volatility for a list of groupings.

    Parameters:
        groupings (list[dict[str, DataFrame]]): A list of dictionaries, 
        each containing an asset name and its corresponding DataFrame.
        decay_factor (float): The decay factor for the EWMA calculation. Default is 0.94.

    Returns:
        pd.Series: A pandas Series representing the cross-sectional average volatility.
    """
    df_returns = pd.DataFrame()
    for grouping in groupings:
        df_returns[grouping["asset"]] = grouping["data"]["return_2d"]
    if not isinstance(df_returns.index, pd.DatetimeIndex):
        raise ValueError("The index of the data should be a DatetimeIndex.")
    stock_volatilities = df_returns.apply(
        lambda returns: calculate_volatility_ewma(returns, decay_factor))

    # Compute the cross-sectional average volatility
    # (mean volatility across all stocks at each period)
    cross_sectional_avg_volatility = stock_volatilities.mean(axis=1)

    cross_sectional_avg_volatility.index = df_returns.index
    if cross_sectional_avg_volatility.index.tz is None:  # type: ignore
        cross_sectional_avg_volatility.index = cross_sectional_avg_volatility.index.tz_localize(  # type: ignore
            "UTC")

    return cross_sectional_avg_volatility * np.sqrt(252)


class DistDataset(Dataset):
    """
    A PyTorch Dataset class for loading and processing financial data.

    Attributes:
        datas (dict[str, list[dict[str, DataFrame]]]): A dictionary containing asset data.
        market_data (DataFrame): A DataFrame containing market data.
        normalization_lookback (int): The lookback period for normalization.
        start_date (str): The start date for the dataset.
        end_date (str): The end date for the dataset.
        lookahead (int, optional): The lookahead period for generating data. Default is None.
        x (list): A list to store the input features.
        s (list): A list to store the cross-sectional volatility.
        z (list): A list to store the market data.
        y (list): A list to store the target values.
        cat (list): A list to store the categorical data.
        asset_name (list): A list to store the asset names.
    """

    def __init__(self,
                 datas: dict[str, list[dict[str, DataFrame]]],
                 market_data: DataFrame,
                 normalization_lookback: int,
                 start_date: str,
                 end_date: str,
                 lookahead: Optional[int] = None,
                 flow: Optional[bool] = False
                 ) -> None:
        self.datas = datas
        self.market_data = market_data
        if not isinstance(self.market_data.index, pd.DatetimeIndex):
            raise ValueError("The index of the market data should be a DatetimeIndex.")
        if self.market_data.index.tz is None:
            self.market_data.index = self.market_data.index.tz_localize("UTC")
        self.start_date = pd.to_datetime(
            start_date, format="%Y-%m-%d", utc=True)
        self.end_date = pd.to_datetime(end_date, format="%Y-%m-%d", utc=True)
        self.normalization_lookback = normalization_lookback
        self.x = []
        self.s = []
        self.z = []
        self.y = []
        self.observed_returns = []
        self.cat = []
        self.asset_name = []
        self.flow = flow

        self.lookahead = lookahead

        self.gen_data()

    def gen_data(self) -> None:
        """
        Generate the data for the dataset.

        This method generates the input features, cross-sectional volatility,
        market data, target values, and categorical data. 
        It does so by iterating over the asset data and calculating the normalized returns,
        rolling mean, and rolling standard deviation.
        The data is then normalized and filtered based on the start and end dates.
        The method then generates the input features, cross-sectional volatility, market data,
        target values, and categorical data for each asset
        based on the lookback period and lookahead period.

        raises:
            ValueError: If the index of the data is not a DatetimeIndex.
        """
        for grouping in self.datas.keys():
            cross_vol = cross_sectional_volatility(self.datas[grouping])
            for data in self.datas[grouping]:
                asset_name = data["asset"]
                df = data["data"]
                if not isinstance(df.index, pd.DatetimeIndex):
                    raise ValueError("The index of the data should be a DatetimeIndex.")
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")

                shared_index = list(set(df.index) & set(cross_vol.index)
                                    & set(self.market_data.index))
                df = df.loc[shared_index]
                df = df.sort_index()
                self.df = df
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
                    normalized_df.index[-1],
                    min(normalized_df.index, key=lambda x: abs(x - self.end_date))
                )  # pylint: disable=W3301

                normalized_df = normalized_df.loc[start_date:end_date]
                while True:
                    if self.lookahead is None:
                        lookforward = np.random.randint(15, 30)
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
                    if self.flow:
                        observed_returns = self.df["return_2d"].loc[y_index]
                        self.observed_returns.append(observed_returns)

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

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
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
        if self.flow:
            observed_returns = torch.tensor(
                self.observed_returns[idx].values, dtype=torch.float32).to(DEVICE).view(-1, 1)
            return x, s, z, y, observed_returns
        else:
            return x, s, z, y, sy


class StaticDistDataset(Dataset):
    """
    A PyTorch Dataset class for loading and processing static financial data. 

    Attributes:
        datas (dict[str, list[dict[str, DataFrame]]]): A dictionary containing asset data.
        market_data (DataFrame): A DataFrame containing market data.
        normalization_lookback (int): The lookback period for normalization.
        start_date (str): The start date for the dataset.
        end_date (str): The end date for the dataset.
        x (list): A list to store the input features.
        s (list): A list to store the cross-sectional volatility.
        z (list): A list to store the market data.
        y (list): A list to store the target values.
        cat (list): A list to store the categorical data.
        asset_name (list): A list to store the asset names.
    """

    def __init__(self,
                 datas: dict[str, list[dict[str, DataFrame]]],
                 market_data: DataFrame,
                 normalization_lookback: int,
                 start_date: str,
                 end_date: str
                 ) -> None:
        self.datas = datas
        self.market_data = market_data
        if not isinstance(self.market_data.index, pd.DatetimeIndex):
            raise ValueError("The index of the market data should be a DatetimeIndex.")
        if self.market_data.index.tz is None:
            self.market_data.index = self.market_data.index.tz_localize("UTC")
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

    def gen_data(self) -> None:
        """
        Generate the data for the dataset.

        This method generates the input features, cross-sectional volatility,
        market data, target values, and categorical data.
        It does so by iterating over the asset data and calculating the normalized returns,
        rolling mean, and rolling standard deviation.
        The data is then normalized and filtered based on the start and end dates.
        The method then generates the input features, cross-sectional volatility, market data,
        target values, and categorical data for each asset.

        raises:
            ValueError: If the index of the data is not a DatetimeIndex.
        """
        for grouping in self.datas.keys():
            cross_vol = cross_sectional_volatility(self.datas[grouping])
            for data in self.datas[grouping]:
                asset_name = data["asset"]
                df = data["data"]
                if not isinstance(df.index, pd.DatetimeIndex):
                    raise ValueError("The index of the data should be a DatetimeIndex.")
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
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
                    normalized_df.index[-1],
                    min(normalized_df.index, key=lambda x: abs(x - self.end_date))
                )  # pylint: disable=W3301

                normalized_df = normalized_df.loc[start_date:end_date]
                for i in range(0, len(normalized_df)-22, 22):
                    date = normalized_df.index[i]
                    x = normalized_df.loc[date]
                    s = cross_vol.loc[date]
                    z = date
                    y = returns.loc[normalized_df.iloc[i+22].name]  # type: ignore

                    cat = [0] * len(self.datas.keys())
                    cat[list(self.datas.keys()).index(grouping)] = 1

                    self.cat.append(cat)

                    self.x.append(x)
                    self.s.append(s)
                    self.z.append(z)
                    self.y.append(y)
                    self.asset_name.append(asset_name)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
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


def get_grouping(outer_dict, inner_str) -> Union[str, None]:
    """Get the key of the outer dictionary that contains the inner string."""
    # Iterate over the outer dict
    for outer_key, list_of_dicts in outer_dict.items():
        # Iterate over the list of dictionaries
        for inner_dict in list_of_dicts:
            # Check if the inner string is in this inner dictionary
            if inner_str == inner_dict["asset"]:
                return outer_key
    # Return None or handle the case if not found
    return None


class TestDataset(Dataset):
    """
    A PyTorch Dataset class for loading and processing financial data for testing.

    Attributes:
        datas (dict[str, list[dict[str, DataFrame]]): A dictionary containing asset data.
        market_data (DataFrame): A DataFrame containing market data.
        normalization_lookback (int): The lookback period for normalization.
        start_date (str): The start date for the dataset.
        end_date (str): The end date for the dataset.
        lookforward (int): The lookforward period for generating data.
        x (list): A list to store the input features.
        s (list): A list to store the cross-sectional volatility.
        z (list): A list to store the market data.
        y (list): A list to store the target values.
        cat (list): A list to store the categorical data.
        main_asset (str): The main asset for the dataset.
    """

    def __init__(self,
                 datas: dict[str, list[dict[str, DataFrame]]],
                 market_data: DataFrame,
                 normalization_lookback: int,
                 start_date: str,
                 end_date: str,
                 lookforward: int,
                 test: bool = False
                 ) -> None:
        super().__init__()
        self.datas = datas
        self.market_data = market_data
        if not isinstance(self.market_data.index, pd.DatetimeIndex):
            raise ValueError("The index of the market data should be a DatetimeIndex.")
        if self.market_data.index.tz is None:
            self.market_data.index = self.market_data.index.tz_localize("UTC")
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

        self.test = test
        self.df = None

        self.gen_data()

    def set_main_asset(self, asset) -> None:
        """Set the main asset for the dataset."""
        self.main_asset = asset
        self.gen_data()

    def gen_data(self) -> None:
        """
        Generate the data for the dataset.

        This method generates the input features, cross-sectional volatility, 
        market data, target values, and categorical data. 
        It does so by iterating over the asset data and calculating the normalized returns,
        rolling mean, and rolling standard deviation. The data is then normalized and filtered
        based on the start and end dates. The method then generates the input features,
        cross-sectional volatility, market data, target values, and categorical data for each asset
        based on the lookforward period.

        raises:
            ValueError: If the index of the data is not a DatetimeIndex.
            ValueError: If the main asset is not found in the dataset.
        """
        self.x = []
        self.s = []
        self.z = []
        self.y = []
        self.cat = []
        self.denorm = []

        grouping = get_grouping(self.datas, self.main_asset)
        if grouping is None:
            raise ValueError("The main asset is not found in the dataset.")
        cross_vol = cross_sectional_volatility(self.datas[grouping])

        data = [asset for asset in self.datas[grouping]
                if asset["asset"] == self.main_asset][0]["data"]
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("The index of the data should be a DatetimeIndex.")
        if data.index.tz is None:
            data.index = data.index.tz_localize("UTC")

        shared_index = list(set(data.index) & set(cross_vol.index)
                            & set(self.market_data.index))

        df = data.loc[shared_index]
        df = df.sort_index()
        self.df = df
        returns = df["return_2d"].shift(-self.lookforward)
        rolling_mean = df.rolling(window=self.normalization_lookback).mean()
        rolling_std = df.rolling(window=self.normalization_lookback).std()
        normalized_df = (df - rolling_mean) / rolling_std
        normalized_df = normalized_df.iloc[self.normalization_lookback:]
        normalized_df = normalized_df.fillna(0)

        start_date = max(normalized_df.index[0], min(
            normalized_df.index, key=lambda x: abs(x - self.start_date)))
        end_date = min(
            normalized_df.index[-1],
            min(normalized_df.index, key=lambda x: abs(x - self.end_date))
        )  # pylint: disable=W3301

        normalized_df = normalized_df.loc[start_date:end_date]

        for i in range(0, len(normalized_df)-self.lookforward*2):
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

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
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

        if not self.test:
            return x, s, z, y, sy
        else:
            assert self.df is not None
            sub_observed_returns = self.df["return_2d"].loc[self.z[idx]]
            observed_returns = torch.tensor(
                sub_observed_returns.values, dtype=torch.float32).to(DEVICE).view(-1, 1)
            return x, s, z, y, observed_returns


class DynamicBatchSampler:
    """
    A custom batch sampler for dynamic batching.

    Attributes:
        dataset (Dataset): The PyTorch Dataset object.
        batch_size (int): The batch size.
        num_buckets (int): The number of buckets for dynamic batching.
        buckets (defaultdict): A defaultdict to store the buckets.
    """

    def __init__(self,
                 dataset: Dataset,
                 batch_size: int,
                 num_buckets: int = 10
                 ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_buckets = num_buckets
        self.buckets = defaultdict(list)
        self._create_buckets()

    def _create_buckets(self) -> None:
        """Create the buckets for dynamic batching."""
        lengths = [len(self.dataset[i][0]) for i in range(len(self.dataset))]  # type: ignore
        min_len, max_len = min(lengths), max(lengths)
        bucket_width = (max_len - min_len) / self.num_buckets

        for idx, length in enumerate(lengths):
            bucket = min(int((length - min_len) / bucket_width),
                         self.num_buckets - 1)
            self.buckets[bucket].append(idx)

    def __iter__(self) -> Generator[list[int], None, None]:
        for bucket in self.buckets.values():
            np.random.shuffle(bucket)
            for i in range(0, len(bucket), self.batch_size):
                yield bucket[i:i + self.batch_size]

    def __len__(self) -> int:
        return sum(len(bucket) // self.batch_size + (1 if len(bucket) % self.batch_size else 0)
                   for bucket in self.buckets.values())


def collate_fn(batch) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Collate function for the DataLoader.

    This function pads the sequences in the batch to the same length. To allow for dynamic batching,
    the sequences are sorted by length before padding.

    Parameters:
        batch (list): A list of tuples containing the data for each sample in the batch.

    Returns:
        tuple: A tuple containing the padded input features, cross-sectional volatility, 
        market data, target values, and scaled target values.
    """
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


def get_dataset(
    normalization_lookback: int,
    start_date: str,
    end_date: str,
    lookahead: Optional[int] = None,
    flow: Optional[bool] = False
) -> DistDataset:
    """Get the dataset for training and validation."""

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
    dataset = DistDataset(
        datas, market_data, normalization_lookback, start_date, end_date, lookahead, flow)
    return dataset


def get_static_dataset(
    normalization_lookback: int,
    start_date: str,
    end_date: str
) -> StaticDistDataset:
    """Get the static dataset for training and validation."""
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
        datas, market_data, normalization_lookback, start_date, end_date)
    return dataset


def get_test_dataset(
    normalization_lookback: int,
    start_date: str,
    end_date: str,
    lookforward: int = 22,
    test: bool = False
) -> TestDataset:
    """Get the test dataset for testing."""
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
        datas, market_data, normalization_lookback, start_date, end_date, lookforward, test)
    return dataset


def get_test_synthetic_dataset(dense_normalization_lookback: int,
                               lstm_normalization_lookback: int,
                               num_periods: int,
                               target_correlation: float,
                               n_assets: int = 10,
                               n_market_assets: int = 15,
                               distribution: str = 'normal'
                               ) -> tuple[TestDataset, TestDataset]:
    """
    Get the synthetic test dataset for testing.

    The function generates synthetic data with the desired target correlation and
    creates a test dataset for the dense model and the LSTM model. The dataset is created
    with the specified normalization lookback periods and number of periods.

    Parameters:
        dense_normalization_lookback (int): The lookback period for normalization
        for the dense model.
        lstm_normalization_lookback (int): The lookback period for normalization 
        for the LSTM model.
        num_periods (int): The number of periods for the synthetic data.
        target_correlation (float): The desired target correlation for the synthetic data.
        n_assets (int): The number of synthetic assets. Default is 10.

    Returns:
        tuple: A tuple containing the dense test dataset and the LSTM test dataset."""
    dense_num_periods = num_periods + dense_normalization_lookback + 22 + 22
    lstm_num_periods = num_periods + lstm_normalization_lookback + 22 + 22

    num_periods = max(dense_num_periods, lstm_num_periods)

    synthetic_dfs, market_datas = generate_synthetic_data(
        num_periods, target_correlation, n_assets, n_market_assets, distribution)

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
        datas, market_data, dense_normalization_lookback, "2020-01-01", "2100-01-01", 22)
    lstm_dataset = TestDataset(
        datas, market_data, lstm_normalization_lookback, "2020-01-01", "2100-01-01", 22)
    return dense_dataset, lstm_dataset
