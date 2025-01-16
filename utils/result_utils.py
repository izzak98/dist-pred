"""Utility functions for reporting results and plotting PDFs."""
import os
import warnings
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch import Tensor
from scipy.interpolate import PchipInterpolator
from scipy.stats import wasserstein_distance, norm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inference(model, dataloader, test_loss_fn, is_dense=True, is_linear=False, hybrid=False, quant_probs=None):
    losses = []
    targets = []
    model.eval()
    for x, s, z, y, obs in dataloader:
        x, s, z, y = x.to(DEVICE), s.to(DEVICE), z.to(DEVICE), y.to(DEVICE)
        if is_dense or is_linear:
            x = x[:, 0, :]
            z = z[:, 0, :]
            s = s[:, 0]
        elif not is_dense and not is_linear:
            s = s.mean(dim=1)
        if is_linear:
            xx = torch.cat([x, z], dim=1)
            x = xx
        with torch.no_grad():
            if not is_linear:
                _, y_pred = model(x, s, z)
                if hybrid and not is_dense:
                    hybrid_quantiles = []
                    for obs_ in obs:
                        mean = obs_[:, -1].mean().cpu()
                        std = obs_[:, -1].std().cpu()
                        baseline_quantiles = norm.ppf(quant_probs, loc=mean, scale=std)
                        hybrid_quantiles.append(baseline_quantiles)
                    hybrid_quantiles = torch.tensor(hybrid_quantiles).to(DEVICE)
                    y_pred = (y_pred + hybrid_quantiles) / 2
            else:
                y_pred = model(x.cpu().numpy())
                y_pred = torch.tensor(y_pred).to(DEVICE)
            loss = test_loss_fn(y_pred, y[:, -1])
            losses.extend(loss.cpu().numpy())
            targets.extend(y[:, -1].cpu().numpy())
    losses = torch.tensor(losses)
    targets = torch.tensor(targets)

    return losses


def report_results(results):
    # Create DataFrame
    df = pd.DataFrame(results).T

    # Calculate total mean
    dense_total_mean = df['dense'].mean()
    lstm_total_mean = df['lstm'].mean()
    hybrid_total_mean = df['hybrid'].mean()
    linear_total_mean = df['linear'].mean()
    new_row = pd.DataFrame({'linear': linear_total_mean, 'dense': dense_total_mean,
                           'lstm': lstm_total_mean, "hybrid": hybrid_total_mean}, index=['Total Mean'])
    df = pd.concat([df, new_row])

    # Calculate difference and percentage difference for Dense - LSTM
    df['Difference (Dense - LSTM)'] = df['dense'] - df['lstm']
    df['Percentage Difference (Dense - LSTM)'] = (df['dense'] - df['lstm']
                                                  ) / ((df['dense'] + df['lstm']) / 2) * 100
    df['Difference (Dense - LSTM)'] = df.apply(
        lambda row: f"{row['Difference (Dense - LSTM)']:.4f} ({row['Percentage Difference (Dense - LSTM)']:.2f}%)", axis=1)
    df.drop(columns=['Percentage Difference (Dense - LSTM)'], inplace=True)

    # Calculate difference and percentage difference for Linear - LSTM
    df['Difference (Linear - LSTM)'] = df['linear'] - df['lstm']
    df['Percentage Difference (Linear - LSTM)'] = (df['linear'] - df['lstm']
                                                   ) / ((df['linear'] + df['lstm']) / 2) * 100
    df['Difference (Linear - LSTM)'] = df.apply(
        lambda row: f"{row['Difference (Linear - LSTM)']:.4f} ({row['Percentage Difference (Linear - LSTM)']:.2f}%)", axis=1)
    df.drop(columns=['Percentage Difference (Linear - LSTM)'], inplace=True)

    # Calculate difference and percentage difference for Linear - Dense
    df['Difference (Linear - Dense)'] = df['linear'] - df['dense']
    df['Percentage Difference (Linear - Dense)'] = (df['linear'] -
                                                    df['dense']) / ((df['linear'] + df['dense']) / 2) * 100
    df['Difference (Linear - Dense)'] = df.apply(
        lambda row: f"{row['Difference (Linear - Dense)']:.4f} ({row['Percentage Difference (Linear - Dense)']:.2f}%)", axis=1)
    df.drop(columns=['Percentage Difference (Linear - Dense)'], inplace=True)

    # Format the table
    formatted_df = df.apply(lambda x: round(x, 4))

    print(formatted_df.to_string())


def generate_smooth_pdf(quantiles, taus, min_density=1e-3, eps=1e-6, window=61):
    """
    Generate a smoothed PDF from quantiles with additional controls to prevent spikes
    and ensure the CDF is between [0, 1].
    """
    # Constants
    GRID_POINTS = 1000
    og_quants = quantiles.copy()
    og_taus = taus.copy()

    unique_mask = np.concatenate(([True], np.diff(quantiles) > eps))
    quantiles = quantiles[unique_mask]
    taus = taus[unique_mask]

    # Create denser grid
    grid_x = np.linspace(quantiles[0], quantiles[-1], GRID_POINTS)

    # Monotonic spline for the CDF
    try:
        cdf_monotonic = PchipInterpolator(quantiles, taus, extrapolate=False)
        cdf = cdf_monotonic(grid_x)
    except Exception as e:
        # print(f"Quantiles: {og_quants}")
        # print(f"Taus: {og_taus}")
        # print("Falling back to linear interpolation:", e)
        cdf = np.interp(grid_x, quantiles, taus)

    # Clamp CDF to [0,1], then ensure it's monotonically non-decreasing
    cdf = np.clip(cdf, 0, 1)
    cdf = np.maximum.accumulate(cdf)
    # Rescale so that it starts exactly at 0 and ends exactly at 1
    cdf -= cdf[0]
    if cdf[-1] > 0:
        cdf /= cdf[-1]

    # Approximate PDF from finite differences (or use derivative if PCHIP)
    density = np.gradient(cdf, grid_x)

    smoothed_density = np.convolve(density, np.ones(window)/window, mode='same')

    # Ensure non-negative and non-zero density
    smoothed_density = np.maximum(smoothed_density, min_density)

    # Normalize PDF to integrate to 1
    area = np.trapz(smoothed_density, grid_x)
    smoothed_density = smoothed_density / area

    # regenerate CDF
    cdf = np.cumsum(smoothed_density) * (grid_x[1] - grid_x[0])

    return grid_x, smoothed_density, cdf


def calculate_wasserstein(cdf, predicted_grid, realized_returns):
    """
    Calculate Wasserstein distance between predicted PDF and realized returns.

    Args:
    pdf: array of predicted probability density values
    predicted_grid: array of x values where density is evaluated
    realized_returns: array of actual realized returns

    Returns:
    float: Wasserstein distance
    """

    # Sample predicted returns from the PDF
    random_uniform_samples = np.random.uniform(0, 1, len(realized_returns))
    predicted_samples = np.interp(random_uniform_samples, cdf, predicted_grid)

    # Calculate Wasserstein distance
    return wasserstein_distance(predicted_samples, realized_returns)


def calculate_crps(predicted_cdf, observed_values):
    """
    Calculate the Continuous Ranked Probability Score (CRPS) for predicted CDFs and observed values.

    Parameters:
    - predicted_cdf: array-like, shape (n_points)
        Predicted cumulative distribution functions for each observation.
    - observed_values: array-like, shape (n_samples,)
        Observed values to compare against the predicted CDFs.

    Returns:
    - crps_scores: array-like, shape (n_samples,)
        The CRPS for each observation.
    - mean_crps: float
        The mean CRPS across all observations.
    """
    n_samples = len(observed_values)
    crps_scores = np.zeros(n_samples)

    for i in range(n_samples):
        # The predicted CDF values for the i-th observation
        # Observed value for the i-th observation
        observed = observed_values[i]

        # Define the step function for the observed value
        step_function = np.array(
            [1 if x >= observed else 0 for x in np.linspace(0, 1, len(predicted_cdf))])

        # Compute the CRPS for the i-th observation
        crps_scores[i] = np.mean((predicted_cdf - step_function) ** 2)

    # Calculate the mean CRPS
    mean_crps = np.mean(crps_scores)

    return mean_crps


def calculate_var_metric(predicted_quantiles, hybrid_quantiles, observed_returns, baseline_quantiles, quantile_level, quant_probs):
    """
    Calculate the deviation from the desired quantile level for predicted quantiles, observed returns, 
    and a Gaussian baseline.

    Parameters:
    - predicted_quantiles: array-like, shape (n_quantiles,)
        Predicted quantiles from the model.
    - observed_returns: array-like, shape (n_observations,)
        Observed future returns.
    - baseline_quantiles: array-like, shape (n_quantiles,)
        Quantiles from the Gaussian baseline.
    - quantile_level: float
        The desired quantile level (e.g., 0.05 for 5%).
    - quant_probs: array-like, shape (n_quantiles,)
        List of quantile levels corresponding to the predicted and baseline quantiles.

    Returns:
    - var_distance_results: dict
        Dictionary containing the deviation from the desired quantile level for the predicted VaR and baseline VaR.
    """
    # Find the index of the desired quantile level
    quantile_idx = quant_probs.index(quantile_level)

    # Predicted VaR
    var_predicted = predicted_quantiles[quantile_idx]

    # Baseline (Gaussian) VaR
    var_baseline = baseline_quantiles[quantile_idx]

    var_hybrid = hybrid_quantiles[quantile_idx]

    # Count violations: observed returns less than the predicted/baseline VaR
    predicted_violations = np.sum(observed_returns < var_predicted)
    baseline_violations = np.sum(observed_returns < var_baseline)
    hybrid_violations = np.sum(observed_returns < var_hybrid)

    # Total number of observations
    total_observations = len(observed_returns)

    # Calculate violation rates
    violation_rate_predicted = predicted_violations / total_observations
    violation_rate_baseline = baseline_violations / total_observations
    violation_rate_hybrid = hybrid_violations / total_observations

    # Calculate the distance from the desired quantile level
    distance_predicted = abs(violation_rate_predicted - quantile_level)
    distance_baseline = abs(violation_rate_baseline - quantile_level)
    distance_hybrid = abs(violation_rate_hybrid - quantile_level)

    # Compile results
    var_distance_results = {
        "var_error_predicted": distance_predicted,
        "var_error_baseline": distance_baseline,
        "var_error_hybrid": distance_hybrid
    }

    return var_distance_results


def quantiles_to_pdf_kde(quantiles, probs=None):
    """Convert quantiles to PDF using KDE."""
    if probs is None:
        probs = np.array([0.00005, 0.00025, 0.00075, 0.00125, 0.00175, 0.0025, 0.005,
                         0.01, 0.015, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                         0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
                         0.9, 0.95, 0.98, 0.99, 0.995, 0.9975, 0.99925, 0.99975, 0.99995])

    x, pdf, cdf = generate_smooth_pdf(quantiles, probs)
    return pdf, cdf, x


def plot_pdf(asset, dense_model, lstm_model, quant_reg, dense_test_dataset, lstm_test_dataset, is_linear=False):
    np.random.seed(42)
    dense_test_dataset.set_main_asset(asset)
    lstm_test_dataset.set_main_asset(asset)
    idx = np.random.randint(0, len(lstm_test_dataset))

    # Dense or Linear model input
    dense_input = dense_test_dataset[idx]
    x, s, z, _, _ = dense_input
    x = x.to(DEVICE)[0].view(1, -1)
    s = s.to(DEVICE)[0].view(1, -1)
    z = z.to(DEVICE)[0].view(1, -1)

    if is_linear:
        xx = torch.cat([x, z], dim=1)
        x = xx

    dense_model.eval()
    with torch.no_grad():
        if not is_linear:
            _, dense_quantiles = dense_model(x, s, z)
        else:
            dense_quantiles = quant_reg(x.cpu().numpy())
            dense_quantiles = torch.tensor(dense_quantiles).to(DEVICE)
    dense_quantiles = dense_quantiles.detach().cpu().numpy().flatten()

    # LSTM model input
    lstm_input = lstm_test_dataset[idx]
    lstm_model.eval()
    x, s, z, _, _ = lstm_input
    x = x.to(DEVICE).view(1, x.shape[0], -1)
    s = s.to(DEVICE).mean().view(1, -1)
    z = z.to(DEVICE).view(1, z.shape[0], -1)

    with torch.no_grad():
        _, lstm_quantiles = lstm_model(x, s, z)
    lstm_quantiles = lstm_quantiles.detach().cpu().numpy().flatten()

    # Convert quantiles to PDFs using new implementation
    dense_pdf, dense_cdf, dense_grid = quantiles_to_pdf_kde(dense_quantiles/100)
    lstm_pdf, lstm_cdf, lstm_grid = quantiles_to_pdf_kde(lstm_quantiles/100)

    plt.figure(figsize=(12, 12))
    plt.suptitle(f'PDF and CDF of {asset}\'s Log Return', fontsize=16)

    # Dense or Linear PDF
    plt.subplot(2, 2, 1)
    plt.plot(dense_grid, dense_pdf,
             label=f"{'Dense' if not is_linear else 'Linear'} PDF", color="blue", linestyle='-')
    plt.fill_between(dense_grid.flatten(), dense_pdf, color="lightblue", alpha=0.5)
    plt.title(f"{'Dense' if not is_linear else 'Linear'} PDF", fontsize=14)
    plt.xlabel("Log Return", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(loc="upper right", fontsize=10)

    # Dense or Linear CDF
    plt.subplot(2, 2, 3)
    plt.plot(dense_grid, dense_cdf,
             label=f"{'Dense' if not is_linear else 'Linear'} CDF", color="blue", linestyle='-')
    plt.title(f"{'Dense' if not is_linear else 'Linear'} CDF", fontsize=14)
    plt.xlabel("Log Return", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    plt.legend(loc="lower right", fontsize=10)

    # LSTM PDF
    plt.subplot(2, 2, 2)
    plt.plot(lstm_grid, lstm_pdf, label="LSTM PDF", color="red", linestyle='-')
    plt.fill_between(lstm_grid.flatten(), lstm_pdf, color="pink", alpha=0.5)
    plt.title("LSTM PDF", fontsize=14)
    plt.xlabel("Log Return", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(loc="upper right", fontsize=10)

    # LSTM CDF
    plt.subplot(2, 2, 4)
    plt.plot(lstm_grid, lstm_cdf, label="LSTM CDF", color="red", linestyle='-')
    plt.title("LSTM CDF", fontsize=14)
    plt.xlabel("Log Return", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    plt.legend(loc="lower right", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join("plots", f'{asset}_pdf_cdf.eps'), format='eps')
    plt.show()


def plot_3d_combined_pdfs(asset, dense_model, lstm_model, quant_reg, dense_test_dataset, lstm_test_dataset, is_linear=False):
    """Plot the combined PDFs of the Dense and LSTM models."""
    dense_test_dataset.set_main_asset(asset)
    lstm_test_dataset.set_main_asset(asset)
    dense_model.eval()
    lstm_model.eval()

    fig = plt.figure(figsize=(18, 8))

    # Plot for Dense or Linear Model
    ax1 = fig.add_subplot(121, projection='3d')
    X, Y, Z = [], [], []

    iterations = min(len(dense_test_dataset), 100)
    for idx in range(iterations):
        x, s, z, _, _ = dense_test_dataset[idx]
        x = x[0].to(DEVICE).view(1, -1)
        s = s[0].to(DEVICE).view(1, -1)
        z = z[0].to(DEVICE).view(1, -1)

        if is_linear:
            xx = torch.cat([x, z], dim=1)
            x = xx

        with torch.no_grad():
            if not is_linear:
                _, dense_quantiles = dense_model(x, s, z)
            else:
                dense_quantiles = quant_reg(x.cpu().numpy())
                dense_quantiles = torch.tensor(dense_quantiles).to(DEVICE)

        dense_quantiles = dense_quantiles.detach().cpu().numpy().flatten()
        dense_pdf, _, dense_grid = quantiles_to_pdf_kde(dense_quantiles/100)

        X.append(dense_grid)
        Z.append(dense_pdf)
        Y.append(np.full(dense_grid.shape, idx))

    X, Y, Z = np.array(X), np.array(Y), np.array(Z)
    X, Y = np.meshgrid(X[0], np.arange(len(X)))

    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, rstride=1,
                     cstride=1, linewidth=0, antialiased=True)

    ax1.view_init(elev=30, azim=-60)
    ax1.set_xlabel('Log Return')
    ax1.set_ylabel('Time Step')
    ax1.set_zlabel('Density')
    ax1.set_title(f"{'Dense' if not is_linear else 'Linear'} Model PDF")

    # Plot for LSTM Model (similar structure)
    ax2 = fig.add_subplot(122, projection='3d')
    X, Y, Z = [], [], []

    iterations = min(len(lstm_test_dataset), 100)
    for idx in range(iterations):
        x, s, z, _, _ = lstm_test_dataset[idx]
        x = x.to(DEVICE).view(1, x.shape[0], -1)
        s = s.to(DEVICE).mean().view(1, -1)
        z = z.to(DEVICE).view(1, z.shape[0], -1)

        with torch.no_grad():
            _, lstm_quantiles = lstm_model(x, s, z)

        lstm_quantiles = lstm_quantiles.detach().cpu().numpy().flatten()
        lstm_pdf, _, lstm_grid = quantiles_to_pdf_kde(lstm_quantiles/100)

        X.append(lstm_grid)
        Z.append(lstm_pdf)
        Y.append(np.full(lstm_grid.shape, idx))

    X, Y, Z = np.array(X), np.array(Y), np.array(Z)
    X, Y = np.meshgrid(X[0], np.arange(len(X)))

    ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, rstride=1,
                     cstride=1, linewidth=0, antialiased=True)

    ax2.view_init(elev=30, azim=-60)
    ax2.set_xlabel('Log Return')
    ax2.set_ylabel('Time Step')
    ax2.set_zlabel('Density')
    ax2.set_title('LSTM Model PDF')

    plt.suptitle(f'PDFs of {asset}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(os.path.join("plots", f'{asset}_combined_pdf.eps'),
                format='eps', dpi=300, bbox_inches='tight')
    plt.show()
