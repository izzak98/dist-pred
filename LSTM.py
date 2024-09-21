import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.train_utils import TwoStageQuantileLoss as FullQuantileLoss
from utils.data_utils import collate_fn, DynamicBatchSampler, get_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTM_Model(nn.Module):
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
                 batch_norm: bool = False,
                 input_size: int = 40,
                 market_data_size: int = 15,
                 output_size: int = 37
                 ) -> None:
        super().__init__()
        self.lstm_layers = lstm_layers
        self.lstm_h = lstm_h
        self.hidden_layers = hidden_layers
        self.market_lstm_layers = market_lstm_layers
        self.market_lstm_h = market_lstm_h
        self.market_hidden_layers = market_hidden_layers
        self.dropout = dropout
        self.input_size = input_size
        self.output_size = output_size

        # Market module LSTM and Linear layers
        self.market_lstm = nn.LSTM(
            market_data_size, market_lstm_h, market_lstm_layers, dropout=dropout, batch_first=True)

        market_layers = []
        for i, p in enumerate(market_hidden_layers):
            if i == 0:
                market_layers.append(nn.Linear(market_lstm_h, p))
            else:
                market_layers.append(nn.Linear(market_hidden_layers[i-1], p))
            if batch_norm:
                market_layers.append(nn.BatchNorm1d(p))
            if market_hidden_activation != "":
                market_layers.append(self.get_activation(market_hidden_activation))
            
        market_layers.append(nn.Linear(market_hidden_layers[-1], 1))
        self.market_module = nn.Sequential(*market_layers)

        # Normalize module LSTM and Linear layers
        self.normalize_lstm = nn.LSTM(
            input_size, lstm_h, lstm_layers, dropout=dropout, batch_first=True)

        normalize_layers = []
        for i, p in enumerate(hidden_layers):
            if i == 0:
                normalize_layers.append(nn.Linear(lstm_h, p))
            else:
                normalize_layers.append(nn.Linear(hidden_layers[i-1], p))
            if batch_norm:
                normalize_layers.append(nn.BatchNorm1d(p))
            if hidden_activation != "":
                normalize_layers.append(self.get_activation(hidden_activation))
        normalize_layers.append(nn.Linear(hidden_layers[-1], output_size))
        self.normalize_module = nn.Sequential(*normalize_layers)

    def get_activation(self, activation: str):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "leaky_relu":
            return nn.LeakyReLU()
        elif activation == "elu":
            return nn.ELU()
        else:
            raise ValueError(f"Activation {activation} not supported")

    def forward(self, x, s, z):
        # x: (batch_size, seq_len, input_size)
        # s: (batch_size, seq_len, 1)
        # z: (batch_size, seq_len, market_data_size)

        # Normalize LSTM stage
        normalized_lstm_output, _ = self.normalize_lstm(
            x)  # LSTM returns (output, (h_n, c_n))
        # Pass the LSTM output through the linear layers
        normalized_output = self.normalize_module(
            normalized_lstm_output)  # (batch_size, seq_len, output_size)
        # order the output by the quantiles
        normalized_output, _ = torch.sort(
            normalized_output, dim=2, descending=False)

        # Apply scaling factor from 's'
        new_output = s * normalized_output

        # Market LSTM stage
        market_lstm_output, _ = self.market_lstm(
            z)  # LSTM returns (output, (h_n, c_n))
        # Pass the LSTM output through the linear layers of the market module
        estimated_sigma = self.market_module(
            market_lstm_output)  # (batch_size, seq_len, 1)

        # Raw output scaling with estimated sigma
        raw_output = new_output * estimated_sigma  # Element-wise multiplication

        return normalized_output, raw_output


def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        running_len = 0
        for x, s, z, y, sy in val_loader:
            normalized_output, raw_output = model(x, s, z)
            loss = criterion(raw_output, y, normalized_output, sy)
            total_loss += loss.item()
            running_len += len(x)
    return total_loss / running_len


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, l1_reg, verbose=True):
    best_loss = float('inf')
    n_no_improve = 0
    best_weights = None
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        if verbose:
            p_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        else:
            p_bar = train_loader
        running_len = 0
        for x, s, z, y, sy in p_bar:
            optimizer.zero_grad()
            normalized_output, raw_output = model(x, s, z)
            loss = criterion(raw_output, y, normalized_output, sy)
            
            # Add L1 regularization
            l1_loss = 0
            for param in model.parameters():
                l1_loss += torch.sum(torch.abs(param))
            loss += l1_reg * l1_loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            running_len += len(x)
            if verbose:
                p_bar.set_postfix({'loss': total_loss / running_len})

        val_loss = validate(model, val_loader, criterion)

        out = (
            f"Epoch {epoch+1}, "
            f"Train Loss: {total_loss / running_len:.6f}, "
            f"Val Loss: {val_loss:.6f}"
        )
        if val_loss < best_loss:
            best_loss = val_loss
            n_no_improve = 0
            best_weights = model.state_dict()
        else:
            n_no_improve += 1
            if n_no_improve >= patience:
                out += f"\nEarly stopping at epoch {epoch+1}"
                if verbose:
                    print(out)
                break
        if verbose:
            print(out)

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return best_loss, model


def objective(trial):
    batch_size = trial.suggest_categorical(
        "batch_size", [32, 64, 128, 256, 512, 1024, 2048, 4096])
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
    normalazation_window = trial.suggest_int("normalazation_window", 5, 250)
    raw_lstm_layers = trial.suggest_int("raw_lstm_layers", 1, 5)
    raw_lstm_h = trial.suggest_categorical(
        "raw_lstm_h", [16, 32, 64, 128, 256, 512, 1024])
    raw_hidden_layers = []
    n_hidden_layers = trial.suggest_int("raw_hidden_layers", 1, 5)
    for i in range(n_hidden_layers):
        raw_hidden_layers.append(trial.suggest_categorical(
            f"raw_hidden_layer_{i}", [16, 32, 64, 128, 256, 512, 1024]))

    market_lstm_layers = trial.suggest_int("market_lstm_layers", 1, 5)
    market_lstm_h = trial.suggest_categorical(
        "market_lstm_h", [16, 32, 64, 128, 256, 512, 1024])
    market_hidden_layers = []
    n_hidden_layers = trial.suggest_int("market_hidden_layers", 1, 5)
    for i in range(n_hidden_layers):
        market_hidden_layers.append(trial.suggest_categorical(
            f"market_hidden_layer_{i}", [16, 32, 64, 128, 256, 512, 1024]))

    dropout = trial.suggest_float("dropout", 0.0, 0.9)

    market_activation = trial.suggest_categorical(
        "market_activation", ["relu", "tanh", "sigmoid", "leaky_relu", "elu"])
    hidden_activation = trial.suggest_categorical(
        "hidden_activation", ["relu", "tanh", "sigmoid", "leaky_relu", "elu"])
    
    use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])

    l1_reg = trial.suggest_float("l1_reg", 0.0, 1e-3)
    l2_reg = trial.suggest_float("l2_reg", 0.0, 1e-3)

    model = LSTM_Model(
        lstm_layers = raw_lstm_layers,
        lstm_h = raw_lstm_h,
        hidden_layers = raw_hidden_layers,
        hidden_activation = hidden_activation,
        market_lstm_layers = market_lstm_layers,
        market_lstm_h = market_lstm_h,
        market_hidden_layers = market_hidden_layers,
        market_hidden_activation = market_activation,
        dropout = dropout,
        batch_norm = use_batch_norm
    )
    model.to(DEVICE)
    print(f"Model at trial has {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")
    train_dataset = get_dataset(
        normalazation_window, "1998-01-01", "2018-01-01")
    val_dataset = get_dataset(normalazation_window, "2018-01-01", "2019-01-01")
    train_batch_sampler = DynamicBatchSampler(
        train_dataset, batch_size=batch_size)
    val_batch_sampler = DynamicBatchSampler(val_dataset, batch_size=batch_size)

    train_loader = DataLoader(
        train_dataset, batch_sampler=train_batch_sampler, collate_fn=collate_fn)
    val_loader = DataLoader(
        val_dataset, batch_sampler=val_batch_sampler, collate_fn=collate_fn)

    quantiles = [
        0.00005, 0.00025, 0.00075, 0.00125, 0.00175, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.03,
        0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7,
        0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 0.9975, 0.99925, 0.99975, 0.99995
    ]

    loss_fn = FullQuantileLoss(quantiles)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
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
    import optuna

    n_trials = 100
    study = optuna.create_study(
        direction="minimize", study_name="LSTM", storage="sqlite:///LSTM.db", load_if_exists=True)
    n_trials = n_trials - len(study.trials)
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)
