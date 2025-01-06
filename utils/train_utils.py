"""Module for training utilities."""
from tqdm import tqdm
from torch import Tensor, nn
from torch.utils.data import DataLoader
import torch
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cheat_quantile_loss(y_true, y_pred, taus) -> Tensor:
    """
    Calculate the quantile loss for multiple quantiles.

    Args:
    y_pred (torch.Tensor): Predicted values (batch_size, num_quantiles)
    y_true (torch.Tensor): True values (batch_size, seq_len, 1)
    """
    y_true = y_true.squeeze(-1)
    realized_quantiles = np.quantile(y_true.cpu().numpy(), np.array(taus))
    realized_quantiles = torch.tensor(realized_quantiles).to(DEVICE)
    diff = realized_quantiles - y_pred
    return torch.mean(diff**2)/10


def quantile_loss(y_true, y_pred, tau) -> Tensor:
    """
    Calculate the quantile loss for a single quantile.

    Args:
    y_true (torch.Tensor): True values (batch_size, seq_len, 1)
    y_pred (torch.Tensor): Predicted values (batch_size, 1)
    tau (float): Quantile value

    Returns:
    torch.Tensor: Quantile loss
    """
    diff = y_true - y_pred.view(-1, 1).unsqueeze(1)  # Expand y_pred to (batch_size, seq_len, 1)
    return torch.mean(torch.max(tau * diff, (tau - 1) * diff))


def aggregate_quantile_loss(y_true, y_pred, taus) -> Tensor:
    """
    Calculate the aggregated quantile loss for multiple quantiles.

    Args:
    y_true (torch.Tensor): True values (batch_size, seq_len, 1)
    y_pred (torch.Tensor): Predicted values (batch_size, num_quantiles)
    taus (list or torch.Tensor): List of quantile values

    Returns:
    torch.Tensor: Aggregated quantile loss
    """
    losses = [quantile_loss(y_true, y_pred[:, i], tau) for i, tau in enumerate(taus)]
    return torch.mean(torch.stack(losses))


class TwoStageQuantileLoss(torch.nn.Module):
    """Quantile loss for two-stage training."""

    def __init__(self, taus) -> None:
        super().__init__()
        self.taus = taus

    def forward(self, y_pred_raw, y_true_raw, y_pred_std, y_true_std) -> Tensor:
        """
        Calculate the two-stage quantile loss.

        Args:
        y_pred_raw (torch.Tensor): Predicted raw returns (batch_size, num_quantiles)
        y_true_raw (torch.Tensor): True raw returns (batch_size, seq_len, 1)
        y_pred_std (torch.Tensor): Predicted standardized returns (batch_size, num_quantiles)
        y_true_std (torch.Tensor): True standardized returns (batch_size, seq_len, 1)

        Returns:
        torch.Tensor: Two-stage quantile loss
        """
        raw_loss = aggregate_quantile_loss(y_true_raw, y_pred_raw, self.taus)
        std_loss = aggregate_quantile_loss(y_true_std, y_pred_std, self.taus)

        return (raw_loss + std_loss)


class ComparisonQuantileLoss(torch.nn.Module):
    """Quantile loss for comparison."""

    def __init__(self, taus) -> None:
        super().__init__()
        self.register_buffer('taus', torch.tensor(taus).float().to(DEVICE))

    def forward(self, y_pred_raw, y_true_raw) -> Tensor:
        """
        Calculate the comparison quantile loss.

        Args:
            y_pred_raw (torch.Tensor): Predicted raw returns (batch_size, num_quantiles)
            y_true_raw (torch.Tensor): True raw returns (batch_size, 1)

        Returns:
            torch.Tensor: Comparison quantile loss (batch_size, 1)
        """
        # Ensure y_true_raw has the right shape
        y_true_raw = y_true_raw.expand(-1, y_pred_raw.size(1))

        diff = y_true_raw - y_pred_raw
        loss = torch.max(self.taus * diff, (self.taus - 1) * diff)
        return loss.mean(dim=1, keepdim=True)  # Average over quantiles


def validate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, lstm: bool = False) -> float:
    """Validate the model on the validation set."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        running_len = 0
        for x, s, z, y, sy in val_loader:
            normalized_output, raw_output = model(x, s, z)
            loss = criterion(raw_output, y, normalized_output, sy)
            if lstm:
                loss += cheat_quantile_loss(y, raw_output, criterion.taus)
            total_loss += loss.item()
            running_len += 1
    return total_loss / running_len


def train(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        patience: int,
        l1_reg: float = 0.0,
        lstm: bool = False,
        verbose: bool = True
) -> tuple[float, nn.Module]:
    """
    Train the model and return the best validation loss.

    Parameters:
        model (nn.Module): The model to train
        train_loader (DataLoader): The training data loader
        val_loader (DataLoader): The validation data loader
        criterion (nn.Module): The loss function
        optimizer (torch.optim.Optimizer): The optimizer
        num_epochs (int): The number of epochs to train
        patience (int): The number of epochs to wait for improvement before early stopping
        l1_reg (float): L1 regularization coefficient
        verbose (bool): Whether to print training progress
    """
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
            if lstm:
                loss += cheat_quantile_loss(y, raw_output, criterion.taus)
            # Add L1 regularization
            l1_loss = 0
            for param in model.parameters():
                l1_loss += torch.sum(torch.abs(param))
            loss += l1_reg * l1_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            running_len += 1
            if verbose and isinstance(p_bar, tqdm):
                p_bar.set_postfix({'loss': total_loss / running_len})

        val_loss = validate(model, val_loader, criterion, lstm)

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
