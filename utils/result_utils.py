"""Utility functions for reporting results and plotting PDFs."""
import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch import Tensor

from utils.dist_utils import quantiles_to_pdf_kde

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def plot_3d_combined_pdfs(asset,
                          dense_model,
                          lstm_model,
                          quant_reg,
                          dense_test_dataset,
                          lstm_test_dataset,
                          is_linear=False) -> None:
    """Plot the combined PDFs of the Dense and LSTM models."""
    dense_test_dataset.set_main_asset(asset)
    lstm_test_dataset.set_main_asset(asset)
    dense_model.eval()
    lstm_model.eval()

    fig = plt.figure(figsize=(18, 8))  # Reduced width to minimize white space

    # Plot for Dense or Linear Model
    ax1 = fig.add_subplot(121, projection='3d')
    X = []  # to store the grid (log returns)  # pylint: disable=C0103
    Y = []  # to store time steps # pylint: disable=C0103
    Z = []  # to store PDF values (density) # pylint: disable=C0103

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
                # Assuming `quant_reg` is the function for linear models
                dense_quantiles = quant_reg(x.cpu().numpy())
                dense_quantiles = torch.tensor(dense_quantiles).to(DEVICE)

        dense_quantiles = dense_quantiles.detach().cpu().numpy().flatten()
        dense_pdf, _, dense_grid = quantiles_to_pdf_kde(dense_quantiles / 100, bandwidth=0.01)

        X.append(dense_grid)  # X-axis is the grid (log returns)
        Z.append(dense_pdf)   # Z-axis is the PDF (density values)
        Y.append(np.full(dense_grid.shape, idx))  # Y-axis is the time step (idx)

    X = np.array(X)  # pylint: disable=C0103
    Y = np.array(Y)  # pylint: disable=C0103
    Z = np.array(Z)  # pylint: disable=C0103

    # Create meshgrid for X (log returns) and Y (time steps)
    X, Y = np.meshgrid(X[0], np.arange(len(X)))

    surface = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8,  # type: ignore  # pylint: disable=W0612
                               rstride=1, cstride=1, linewidth=0, antialiased=True)

    for i in range(len(X)):   # pylint: disable=C0200
        ax1.plot(X[i], Y[i], Z[i], color=plt.cm.viridis(i / len(X)),  # pylint: disable=E1101 # type: ignore
                 alpha=(i + 1) / len(X) * 0.6)

    ax1.view_init(elev=30, azim=-60)  # type: ignore
    ax1.set_xlabel('Log Return')
    ax1.set_ylabel('Time Step')
    ax1.set_zlabel('Density')  # type: ignore
    ax1.set_title(f"{'Dense' if not is_linear else 'Linear'} Model PDF")

    # Plot for LSTM Model
    ax2 = fig.add_subplot(122, projection='3d')
    X = []  # Reset for LSTM
    Y = []  # Reset for LSTM
    Z = []  # Reset for LSTM

    iterations = min(len(lstm_test_dataset), 100)
    for idx in range(iterations):
        x, s, z, _, _ = lstm_test_dataset[idx]
        x = x.to(DEVICE).view(1, x.shape[0], -1)
        s = s.to(DEVICE).mean().view(1, -1)
        z = z.to(DEVICE).view(1, z.shape[0], -1)

        with torch.no_grad():
            _, lstm_quantiles = lstm_model(x, s, z)

        lstm_quantiles = lstm_quantiles.detach().cpu().numpy().flatten()
        lstm_pdf, _, lstm_grid = quantiles_to_pdf_kde(lstm_quantiles / 100, bandwidth=0.01)

        X.append(lstm_grid)  # X-axis is the grid (log returns)
        Z.append(lstm_pdf)   # Z-axis is the PDF (density values)
        Y.append(np.full(lstm_grid.shape, idx))  # Y-axis is the time step (idx)

    X = np.array(X)  # pylint: disable=C0103
    Y = np.array(Y)  # pylint: disable=C0103
    Z = np.array(Z)  # pylint: disable=C0103

    # Create meshgrid for X (log returns) and Y (time steps)
    X, Y = np.meshgrid(X[0], np.arange(len(X)))  # pylint: disable=C0103

    surface = ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8,  # type: ignore
                               rstride=1, cstride=1, linewidth=0, antialiased=True)

    for i in range(len(X)):  # pylint: disable=C0200
        ax2.plot(X[i], Y[i], Z[i], color=plt.cm.viridis(i / len(X)),  # pylint: disable=E1101 # type: ignore
                 alpha=(i + 1) / len(X) * 0.6)

    ax2.view_init(elev=30, azim=-60)  # type: ignore
    ax2.set_xlabel('Log Return')
    ax2.set_ylabel('Time Step')
    ax2.set_zlabel('Density')  # type: ignore
    ax2.set_title('LSTM Model PDF')

    # Adding the title for the entire plot using the asset name
    plt.suptitle(f'PDFs of {asset}', fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Adjust layout to make space for the title
    plt.savefig(os.path.join("plots", f'{asset}_combined_pdf.eps'),
                format='eps', dpi=300, bbox_inches='tight')
    plt.show()


def plot_pdf(asset,
             dense_model,
             lstm_model,
             quant_reg,
             dense_test_dataset,
             lstm_test_dataset,
             is_linear=False) -> None:
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

    # Convert quantiles to PDFs
    dense_pdf, _, dense_grid = quantiles_to_pdf_kde(dense_quantiles/100)
    lstm_pdf, _, lstm_grid = quantiles_to_pdf_kde(lstm_quantiles/100)

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

    # LSTM PDF
    plt.subplot(2, 2, 2)
    plt.plot(lstm_grid, lstm_pdf, label="LSTM PDF", color="red", linestyle='-')
    plt.fill_between(lstm_grid.flatten(), lstm_pdf, color="pink", alpha=0.5)
    plt.title("LSTM PDF", fontsize=14)
    plt.xlabel("Log Return", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(loc="upper right", fontsize=10)

    # Adjust layout to leave space for the main title
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # type: ignore
    plt.savefig(os.path.join("plots", f'{asset}_pdf_cdf.eps'), format='eps')
    plt.show()
