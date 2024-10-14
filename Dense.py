"""Module for the Dense model."""
import json
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from utils.train_utils import TwoStageQuantileLoss as FullQuantileLoss
from utils.train_utils import train
from utils.data_utils import get_static_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("config.json", "r", encoding="utf-8") as file:
    CONFIG = json.load(file)


class QuantileDense(nn.Module):
    """
    Dense model for quantile regression.

    Parameters:
        raw_hidden_layers (list[int]): Number of units in each hidden layer for raw data.
        hidden_activation (str): Activation function for hidden layers.
        market_hidden_layers (list[int]): Number of units in each hidden layer for market data.
        market_activation (str): Activation function for market data hidden layers.
        dropout (float): Dropout rate.
        layer_norm (bool): Whether to use layer normalization.
        input_size (int): Number of features in the raw data.
        market_data_size (int): Number of features in the market data.
        output_size (int): Number of output units.

    Inputs:
        x (torch.Tensor): Input tensor of raw data.
        s (torch.Tensor): Input tensor of standardized data.
        z (torch.Tensor): Input tensor of market data.

    Returns:
        normalized_output (torch.Tensor): Normalized output tensor.
        raw_output (torch.Tensor): Raw output tensor.
    """

    def __init__(self,
                 raw_hidden_layers: list[int],
                 hidden_activation: str,
                 market_hidden_layers: list[int],
                 market_activation: str,
                 dropout: float,
                 layer_norm: bool = False,
                 input_size: int = 49,
                 market_data_size: int = 21,
                 output_size: int = 37
                 ) -> None:
        super(QuantileDense, self).__init__()
        self.layer_norm = layer_norm
        self.raw_hidden_layers = raw_hidden_layers
        self.market_hidden_layers = market_hidden_layers
        self.raw_layers = nn.ModuleList()
        for i, p in enumerate(raw_hidden_layers):
            if i == 0:
                self.raw_layers.append(nn.Linear(input_size, p))
            else:
                self.raw_layers.append(nn.Linear(raw_hidden_layers[i - 1], p))
            if layer_norm:
                self.raw_layers.append(nn.LayerNorm(p))
            self.raw_layers.append(nn.Dropout(dropout))
            if hidden_activation:
                self.raw_layers.append(self._get_activation(hidden_activation))
        self.raw_layers.append(nn.Linear(raw_hidden_layers[-1], output_size))
        self.raw_model = nn.Sequential(*self.raw_layers)

        self.market_layers = nn.ModuleList()
        for i, p in enumerate(market_hidden_layers):
            if i == 0:
                self.market_layers.append(nn.Linear(market_data_size, p))
            else:
                self.market_layers.append(nn.Linear(market_hidden_layers[i - 1], p))
            if layer_norm:
                self.market_layers.append(nn.LayerNorm(p))
            self.market_layers.append(nn.Dropout(dropout))
            if market_activation:
                self.market_layers.append(self._get_activation(market_activation))
        self.market_layers.append(nn.Linear(market_hidden_layers[-1], 1))
        self.market_model = nn.Sequential(*self.market_layers)

    def _get_activation(self, activation: str) -> nn.Module:
        """Return the activation function."""
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
        normalized_output = self.raw_model(x)
        normalized_output, _ = torch.sort(normalized_output, dim=1)
        market_output = self.market_model(z)

        raw_output = normalized_output * market_output * s
        return normalized_output, raw_output


def objective(trial) -> float:
    """Objective function for Optuna optimization."""
    batch_size = trial.suggest_categorical(
        "batch_size", [32, 64, 128, 256, 512, 1024, 2048])
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
    normalization_window = trial.suggest_int("normalazation_window", 5, 250)
    raw_hidden_layers = []
    n_raw_hidden_layers = trial.suggest_int("n_raw_hidden_layers", 1, 5)
    for i in range(n_raw_hidden_layers):
        raw_hidden_layers.append(trial.suggest_categorical(
            f"raw_hidden_layer_{i}", [16, 32, 64, 128, 256, 512]))
    hidden_activation = trial.suggest_categorical(
        "hidden_activation", ["relu", "tanh", "sigmoid", "leaky_relu", "elu", ""])

    market_hidden_layers = []
    n_market_hidden_layers = trial.suggest_int("n_market_hidden_layers", 1, 5)
    for i in range(n_market_hidden_layers):
        market_hidden_layers.append(trial.suggest_categorical(
            f"market_hidden_layer_{i}", [16, 32, 64, 128, 256, 512]))
    market_activation = trial.suggest_categorical(
        "market_activation", ["relu", "tanh", "sigmoid", "leaky_relu", "elu", ""])

    dropout = trial.suggest_float("dropout", 0.0, 0.9)
    layer_norm = trial.suggest_categorical("layer_norm", [0, 1])
    layer_norm = bool(layer_norm)

    l1_reg = trial.suggest_float("l1_reg", 0.0, 1e-3)
    l2_reg = trial.suggest_float("l2_reg", 0.0, 1e-3)

    model = QuantileDense(
        raw_hidden_layers=raw_hidden_layers,
        hidden_activation=hidden_activation,
        market_hidden_layers=market_hidden_layers,
        market_activation=market_activation,
        dropout=dropout,
        layer_norm=layer_norm
    ).to(DEVICE)

    validation_period_start = CONFIG["general"]["dates"]["validation_period"]["start_date"]
    validation_period_end = CONFIG["general"]["dates"]["validation_period"]["end_date"]
    train_dataset = get_static_dataset(
        normalization_window, "1998-01-01", validation_period_start)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = get_static_dataset(
        normalization_window, validation_period_start, validation_period_end)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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
        verbose=False
    )
    return best_loss


def test_functionality() -> None:
    """Test the functionality of the model."""
    model_test_params = {
        "raw_hidden_layers": [128, 64],
        "hidden_activation": "",
        "market_hidden_layers": [32, 16],
        "market_activation": "elu",
        "dropout": 0.2,
        "layer_norm": True
    }
    normalization_window = 100
    batch_size = 1024
    learning_rate = 1e-3
    l1_reg = 0.2
    l2_reg = 0.2

    model = QuantileDense(**model_test_params).to(DEVICE)
    print(model)

    validation_period_start = CONFIG["general"]["dates"]["validation_period"]["start_date"]
    validation_period_end = CONFIG["general"]["dates"]["validation_period"]["end_date"]

    train_dataset = get_static_dataset(
        normalization_window, "1998-01-01", validation_period_start)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = get_static_dataset(
        normalization_window, validation_period_start, validation_period_end)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    quantiles = CONFIG["general"]["quantiles"]

    loss_fn = FullQuantileLoss(quantiles)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    _, _ = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=loss_fn,
        optimizer=optimizer,
        num_epochs=100,
        patience=10,
        l1_reg=l1_reg,
        verbose=True
    )


if __name__ == "__main__":
    test_functionality()
