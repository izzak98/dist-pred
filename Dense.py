import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.train_utils import TwoStageQuantileLoss as FullQuantileLoss
from utils.data_utils import get_static_dataset
from LSTM import train

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QuantileDense(nn.Module):
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
                 ):
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


    def _get_activation(self, activation: str):
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
    
    def forward(self, x, s, z):
        normalized_output = self.raw_model(x)
        normalized_output, _ = torch.sort(normalized_output, dim=1) 
        market_output = self.market_model(z)

        raw_output = normalized_output * market_output * s
        return normalized_output, raw_output
    
def objective(trial):
    batch_size = trial.suggest_categorical(
        "batch_size", [32, 64, 128, 256, 512, 1024, 2048])
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
    normalazation_window = trial.suggest_int("normalazation_window", 5, 250)
    raw_hidden_layers = []
    n_raw_hidden_layers = trial.suggest_int("n_raw_hidden_layers", 1, 5)
    for i in range(n_raw_hidden_layers):
        raw_hidden_layers.append(trial.suggest_categorical(f"raw_hidden_layer_{i}", [16, 32, 64, 128, 256, 512]))
    hidden_activation = trial.suggest_categorical("hidden_activation", ["relu", "tanh", "sigmoid", "leaky_relu", "elu", ""])

    market_hidden_layers = []
    n_market_hidden_layers = trial.suggest_int("n_market_hidden_layers", 1, 5)
    for i in range(n_market_hidden_layers):
        market_hidden_layers.append(trial.suggest_categorical(f"market_hidden_layer_{i}", [16, 32, 64, 128, 256, 512]))
    market_activation = trial.suggest_categorical("market_activation", ["relu", "tanh", "sigmoid", "leaky_relu", "elu", ""])

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

    print(
        f"Model at trial has {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")
    train_dataset = get_static_dataset(
        normalazation_window, "1998-01-01", "2018-01-01")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = get_static_dataset(
        normalazation_window, "2018-01-01", "2020-01-01")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    
    quantiles = [
        0.00005, 0.00025, 0.00075, 0.00125, 0.00175, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.03,
        0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7,
        0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 0.9975, 0.99925, 0.99975, 0.99995
    ]

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





        
if __name__ == "__main__":
    model_test_params = {
        "raw_hidden_layers": [128, 64],
        "hidden_activation": "",
        "market_hidden_layers": [32, 16],
        "market_activation": "elu",
        "dropout": 0.2,
        "layer_norm": True
    }
    normalazation_window = 100
    batch_size = 1024
    learning_rate = 1e-3
    l1_reg = 0.2
    l2_reg = 0.2

    model = QuantileDense(**model_test_params).to(DEVICE)
    print(model)

    train_dataset = get_static_dataset(
        normalazation_window, "1998-01-01", "2018-01-01")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = get_static_dataset(
        normalazation_window, "2018-01-01", "2020-01-01")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    quantiles = [
        0.00005, 0.00025, 0.00075, 0.00125, 0.00175, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.03,
        0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7,
        0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 0.9975, 0.99925, 0.99975, 0.99995
    ]

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
        verbose=True
    )

