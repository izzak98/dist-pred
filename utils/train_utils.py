import torch


def quantile_loss(y_true, y_pred, tau):
    """
    Calculate the quantile loss for a single quantile.

    Args:
    y_true (torch.Tensor): True values (batch_size, seq_len, 1)
    y_pred (torch.Tensor): Predicted values (batch_size, seq_len, 1)
    tau (float): Quantile value

    Returns:
    torch.Tensor: Quantile loss
    """
    diff = y_true - y_pred
    return torch.mean(torch.max(tau * diff, (tau - 1) * diff))


def aggregate_quantile_loss(y_true, y_pred, taus):
    """
    Calculate the aggregated quantile loss for multiple quantiles.

    Args:
    y_true (torch.Tensor): True values (batch_size, seq_len, 1)
    y_pred (torch.Tensor): Predicted values (batch_size, seq_len, num_quantiles)
    taus (list or torch.Tensor): List of quantile values

    Returns:
    torch.Tensor: Aggregated quantile loss
    """
    losses = [quantile_loss(y_true, y_pred[:, :, i:i+1], tau) for i, tau in enumerate(taus)]
    return torch.mean(torch.stack(losses))


class TwoStageQuantileLoss(torch.nn.Module):
    def __init__(self, taus):
        super().__init__()
        self.taus = taus

    def forward(self,  y_pred_raw, y_true_raw, y_pred_std, y_true_std):
        """
        Calculate the two-stage quantile loss.

        Args:
        y_true_raw (torch.Tensor): True raw returns (batch_size, seq_len, 1)
        y_pred_raw (torch.Tensor): Predicted raw returns (batch_size, seq_len, num_quantiles)
        y_true_std (torch.Tensor): True standardized returns (batch_size, seq_len, 1)
        y_pred_std (torch.Tensor): Predicted standardized returns (batch_size, seq_len, num_quantiles)

        Returns:
        torch.Tensor: Two-stage quantile loss
        """
        raw_loss = aggregate_quantile_loss(y_true_raw, y_pred_raw, self.taus)
        std_loss = aggregate_quantile_loss(y_true_std, y_pred_std, self.taus)

        return 0.5 * (raw_loss + std_loss)
