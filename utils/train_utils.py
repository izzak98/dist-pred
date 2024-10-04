import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def quantile_loss(y_true, y_pred, tau):
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

def aggregate_quantile_loss(y_true, y_pred, taus):
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
    def __init__(self, taus):
        super().__init__()
        self.taus = taus

    def forward(self, y_pred_raw, y_true_raw, y_pred_std, y_true_std):
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

        return 0.5 * (raw_loss + std_loss)
    
class ComparisonQuantileLoss(torch.nn.Module):
    def __init__(self, taus):
        super().__init__()
        self.register_buffer('taus', torch.tensor(taus).float().to(DEVICE))

    def forward(self, y_pred_raw, y_true_raw):
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

    
