"""LSTM model for quantile regression."""
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import Tensor

from utils.train_utils import TwoStageQuantileLoss as FullQuantileLoss
from utils.train_utils import train
from utils.data_utils import collate_fn, DynamicBatchSampler, get_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("config.json", "r", encoding="utf-8") as file:
    CONFIG = json.load(file)


class LSTM_Model(nn.Module):
    """
    LSTM model for quantile regression.

    Parameters:
        lstm_layers (int): Number of LSTM layers for the normalization module.
        lstm_h (int): Number of hidden units in the LSTM layers.
        hidden_layers (list[int]): Number of units in each hidden layer for the
        normalization module.
        hidden_activation (str): Activation function for hidden layers in the normalization module.
        market_lstm_layers (int): Number of LSTM layers for the market module.
        market_lstm_h (int): Number of hidden units in the LSTM layers for the market module.
        market_hidden_layers (list[int]): Number of units in each hidden layer
        for the market module.
        market_hidden_activation (str): Activation function for hidden layers in the market module.
        dropout (float): Dropout rate.
        layer_norm (bool): Whether to use layer normalization.
        input_size (int): Number of features in the raw data.
        market_data_size (int): Number of features in the market data.

    Inputs:
        x (torch.Tensor): Input tensor of raw data.
        s (torch.Tensor): Input tensor of standardized data.
        z (torch.Tensor): Input tensor of market data.

    Returns:
        normalized_output (torch.Tensor): Normalized output tensor.
        raw_output (torch.Tensor): Raw output tensor.
    """

    def __init__(self,
                 lstm_layers: int,
                 lstm_h: int,
                 hidden_layers: list[int],
                 hidden_activation: str,
                 market_lstm_layers: int,
                 market_lstm_h: int,
                 market_hidden_layers: list[int],
                 market_hidden_activation: str,
                 dropout: float,
                 layer_norm: bool = False,
                 input_size: int = 49,
                 market_data_size: int = 21,
                 output_size: int = 37
                 ) -> None:
        super().__init__()
        self.layer_norm = layer_norm
        self.dropout = dropout

        # Normalize module LSTM and Linear layers
        self.normalize_lstm = nn.LSTM(
            input_size, lstm_h, lstm_layers, dropout=dropout, batch_first=True)
        self.normalize_module = self._build_module(
            lstm_h, hidden_layers, output_size, hidden_activation)

        # Market module LSTM and Linear layers
        self.market_lstm = nn.LSTM(
            market_data_size, market_lstm_h, market_lstm_layers, dropout=dropout, batch_first=True)
        self.market_module = self._build_module(
            market_lstm_h, market_hidden_layers, 1, market_hidden_activation)

    def _build_module(self, input_size, hidden_layers, output_size, activation) -> nn.Sequential:
        layers = []
        for i, neurons in enumerate(hidden_layers):
            layers.append(nn.Linear(input_size if i ==
                          0 else hidden_layers[i-1], neurons))
            if self.layer_norm:
                layers.append(nn.LayerNorm(neurons))
            if activation:
                layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(hidden_layers[-1], output_size))
        return nn.Sequential(*layers)

    def _get_activation(self, activation: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU()
        }
        if activation not in activations:
            raise ValueError(f"Activation {activation} not supported")
        return activations[activation]

    def forward(self, x, s, z) -> tuple[Tensor, Tensor]:
        """Forward pass."""
        # x: (batch_size, seq_len, input_size)
        # s: (batch_size, seq_len, 1)
        # z: (batch_size, seq_len, market_data_size)

        # Normalize LSTM stage
        normalized_lstm_output, _ = self.normalize_lstm(x)
        normalized_lstm_output = normalized_lstm_output[:, -1, :]
        # (batch_size,  output_size)
        normalized_output = self.normalize_module(normalized_lstm_output)
        normalized_output, _ = torch.sort(normalized_output, dim=1)

        # Apply scaling factor from 's'
        new_output = normalized_output  # * s

        # Market LSTM stage
        market_lstm_output, _ = self.market_lstm(z)
        market_lstm_output = market_lstm_output[:, -1, :]
        # (batch_size, seq_len, 1)
        estimated_sigma = self.market_module(market_lstm_output)

        # Raw output scaling with estimated sigma
        raw_output = new_output * estimated_sigma  # Element-wise multiplication

        return normalized_output, raw_output


def objective(trial) -> float:
    """objective function for Optuna optimization."""
    batch_size = trial.suggest_categorical(
        "batch_size", [32, 64, 128, 256, 512, 1024, 2048])
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
    normalization_window = trial.suggest_int("normalazation_window", 5, 250)
    raw_lstm_layers = trial.suggest_int("raw_lstm_layers", 1, 5)
    raw_lstm_h = trial.suggest_categorical(
        "raw_lstm_h", [16, 32, 64, 128, 256])
    raw_hidden_layers = []
    n_hidden_layers = trial.suggest_int("raw_hidden_layers", 1, 5)
    for i in range(n_hidden_layers):
        raw_hidden_layers.append(trial.suggest_categorical(
            f"raw_hidden_layer_{i}", [16, 32, 64, 128, 256]))

    market_lstm_layers = trial.suggest_int("market_lstm_layers", 1, 5)
    market_lstm_h = trial.suggest_categorical(
        "market_lstm_h", [16, 32, 64, 128, 256])
    market_hidden_layers = []
    n_hidden_layers = trial.suggest_int("market_hidden_layers", 1, 5)
    for i in range(n_hidden_layers):
        market_hidden_layers.append(trial.suggest_categorical(
            f"market_hidden_layer_{i}", [16, 32, 64, 128, 256]))

    dropout = trial.suggest_float("dropout", 0.0, 0.9)

    market_activation = trial.suggest_categorical(
        "market_activation", ["relu", "tanh", "sigmoid", "leaky_relu", "elu", ""])
    hidden_activation = trial.suggest_categorical(
        "hidden_activation", ["relu", "tanh", "sigmoid", "leaky_relu", "elu", ""])

    use_layer_norm = trial.suggest_categorical("use_layer_norm", [1, 0])
    use_layer_norm = bool(use_layer_norm)

    l1_reg = trial.suggest_float("l1_reg", 0.0, 1e-3)
    l2_reg = trial.suggest_float("l2_reg", 0.0, 1e-3)

    model = LSTM_Model(
        lstm_layers=raw_lstm_layers,
        lstm_h=raw_lstm_h,
        hidden_layers=raw_hidden_layers,
        hidden_activation=hidden_activation,
        market_lstm_layers=market_lstm_layers,
        market_lstm_h=market_lstm_h,
        market_hidden_layers=market_hidden_layers,
        market_hidden_activation=market_activation,
        dropout=dropout,
        layer_norm=use_layer_norm
    )
    model.to(DEVICE)
    model.compile()

    validation_start_date = CONFIG["general"]["dates"]["validation_period"]["start_date"]
    validation_end_date = CONFIG["general"]["dates"]["validation_period"]["end_date"]
    train_dataset = get_dataset(
        normalization_window, "1998-01-01", validation_start_date)
    val_dataset = get_dataset(normalization_window, validation_start_date, validation_end_date)
    train_batch_sampler = DynamicBatchSampler(
        train_dataset, batch_size=batch_size)
    val_batch_sampler = DynamicBatchSampler(val_dataset, batch_size=batch_size)

    train_loader = DataLoader(
        train_dataset, batch_sampler=train_batch_sampler, collate_fn=collate_fn)
    val_loader = DataLoader(
        val_dataset, batch_sampler=val_batch_sampler, collate_fn=collate_fn)

    quantiles = CONFIG["general"]["quantiles"]

    loss_fn = FullQuantileLoss(quantiles)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    best_loss, _ = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=loss_fn,
        optimizer=optimizer,
        num_epochs=100,
        patience=10,
        l1_reg=l1_reg,
        lstm=False,
        verbose=False
    )
    return best_loss


def test_functionality() -> None:
    """Test the functionality of the model."""
    model_test_params = {
        "lstm_layers": 1,
        "lstm_h": 16,
        "hidden_layers": [16],
        "hidden_activation": "relu",
        "market_lstm_layers": 1,
        "market_lstm_h": 16,
        "market_hidden_layers": [16],
        "market_hidden_activation": "relu",
        "dropout": 0.1,
        "layer_norm": True
    }
    normalization_window = 250
    batch_size = 256
    learning_rate = 1e-3
    l1_reg = 0
    l2_reg = 0.02

    model = LSTM_Model(**model_test_params)
    model.to(DEVICE)

    validation_start_date = CONFIG["general"]["dates"]["validation_period"]["start_date"]
    validation_end_date = CONFIG["general"]["dates"]["validation_period"]["end_date"]
    train_dataset = get_dataset(
        normalization_window, "1998-01-01", validation_start_date)
    val_dataset = get_dataset(normalization_window, validation_start_date, validation_end_date)
    train_batch_sampler = DynamicBatchSampler(
        train_dataset, batch_size=batch_size)
    val_batch_sampler = DynamicBatchSampler(val_dataset, batch_size=batch_size)

    train_loader = DataLoader(
        train_dataset, batch_sampler=train_batch_sampler, collate_fn=collate_fn)
    val_loader = DataLoader(
        val_dataset, batch_sampler=val_batch_sampler, collate_fn=collate_fn)

    quantiles = CONFIG["general"]["quantiles"]

    loss_fn = FullQuantileLoss(quantiles)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    _, model = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=loss_fn,
        optimizer=optimizer,
        num_epochs=100,
        patience=10,
        l1_reg=l1_reg,
        lstm=True,
        verbose=True
    )
    # torch.save(model.state_dict(), 'lstm_model.pth')


if __name__ == "__main__":
    test_functionality()
