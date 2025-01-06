"""Utility functions for reporting results and plotting PDFs."""
import os
import warnings
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch import Tensor
from scipy.interpolate import PchipInterpolator


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inference(model, dataloader, test_loss_fn, is_dense=True, is_linear=False) -> Tensor:
    losses = []
    targets = []
    model.eval()
    for x, s, z, y, _ in dataloader:
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
    linear_total_mean = df['linear'].mean()
    new_row = pd.DataFrame({'linear': linear_total_mean, 'dense': dense_total_mean,
                           'lstm': lstm_total_mean}, index=['Total Mean'])
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


def generate_smooth_pdf(quantiles, taus, plot=False):
    """
    Generate a smoothed PDF from quantiles with additional controls to prevent spikes
    and ensure the CDF is between [0, 1].
    """
    # Constants
    GRID_POINTS = 1000
    MIN_DENSITY = 1e-5
    eps = 1e-4
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
        print("Falling back to linear interpolation:", e)
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

    # Smooth the PDF if desired
    window = 51  # Must be odd
    smoothed_density = np.convolve(density, np.ones(window)/window, mode='same')

    # Ensure non-negative and non-zero density
    smoothed_density = np.maximum(smoothed_density, MIN_DENSITY)

    # Normalize PDF to integrate to 1
    area = np.trapz(smoothed_density, grid_x)
    smoothed_density = smoothed_density / area

    if plot:
        plt.figure(figsize=(12, 5))
        plt.plot(grid_x, smoothed_density, label='Smoothed PDF')
        plt.title('Smoothed PDF')
        plt.xlabel('Returns')
        plt.ylabel('Density')
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.figure(figsize=(12, 5))
        plt.scatter(og_quants, og_taus, color='red', label='Original Quantiles')
        plt.plot(grid_x, cdf, 'b-', alpha=0.7, label='CDF')
        plt.title('CDF with Monotonic Spline + Clamping')
        plt.xlabel('Returns')
        plt.ylabel('Probability')
        plt.legend()
        plt.grid(True)
        plt.show()

    return grid_x, smoothed_density, cdf


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
