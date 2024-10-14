"""Module for distribution utilities."""
import numpy as np
from numpy import ndarray
from sklearn.neighbors import KernelDensity


def quantiles_to_pdf_kde(quantiles, grid_points=200, bandwidth=0.01) -> tuple[ndarray, ndarray, ndarray]:
    """
    Convert quantiles to a probability density function (PDF) using Kernel Density Estimation (KDE).

    Parameters:
    - quantiles: Array of quantile values.
    - grid_points: Number of grid points for a denser representation.
    - bandwidth: Bandwidth for the KDE (controls the smoothness of the density).

    Returns:
    - pdf_grid: Probability density function values on the dense grid.
    - cdf_grid: Cumulative distribution function values on the dense grid (obtained by integrating the PDF).
    - dense_grid: The dense grid of quantiles corresponding to the PDF and CDF values.
    """

    # Create a denser grid between the quantiles
    dense_grid = np.linspace(min(quantiles), max(quantiles), grid_points).reshape(-1, 1)

    # Fit KDE to the quantile values
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(quantiles.reshape(-1, 1))

    # Compute log density and convert to regular density
    log_dens = kde.score_samples(dense_grid)
    pdf_grid = np.exp(log_dens)

    # Approximate the CDF by cumulatively summing the PDF
    cdf_grid = np.cumsum(pdf_grid)
    cdf_grid /= cdf_grid[-1]  # Normalize to [0, 1]

    return pdf_grid, cdf_grid, dense_grid
