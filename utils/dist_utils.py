import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.stats import wasserstein_distance, skew, kurtosis, norm


def calculate_wasserstein(cdf, predicted_grid, realized_returns):
    """
    Calculate Wasserstein distance between predicted PDF and realized returns.

    Args:
    cdf: array of predicted cumulative density values (monotonic)
    predicted_grid: array of x values where density is evaluated
    realized_returns: array of actual realized returns

    Returns:
    float: Wasserstein distance
    """

    # Ensure CDF is strictly monotonic
    cdf = np.maximum.accumulate(cdf)

    # Extend grid if necessary to cover realized returns
    grid_min = min(predicted_grid.min(), realized_returns.min())
    grid_max = max(predicted_grid.max(), realized_returns.max())
    extended_grid = np.linspace(grid_min, grid_max, len(predicted_grid))

    # Interpolate CDF to the extended grid
    extended_cdf = np.interp(extended_grid, predicted_grid, cdf)

    # Generate more predicted samples for stability
    random_unif = np.random.uniform(0, 1, size=10 * len(realized_returns))
    predicted_samples = np.interp(random_unif, extended_cdf, extended_grid)

    # Normalize both distributions
    predicted_samples = (predicted_samples - predicted_samples.mean()) / predicted_samples.std()
    realized_returns = (realized_returns - realized_returns.mean()) / realized_returns.std()

    # Calculate Wasserstein distance
    return wasserstein_distance(predicted_samples, realized_returns)


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


# Example usage
taus = np.array([5.0000e-05, 2.5000e-04, 7.5000e-04, 1.2500e-03, 1.7500e-03,
                 2.5000e-03, 5.0000e-03, 1.0000e-02, 1.5000e-02, 2.0000e-02,
                 3.0000e-02, 5.0000e-02, 1.0000e-01, 1.5000e-01, 2.0000e-01,
                 2.5000e-01, 3.0000e-01, 3.5000e-01, 4.0000e-01, 4.5000e-01,
                 5.0000e-01, 5.5000e-01, 6.0000e-01, 6.5000e-01, 7.0000e-01,
                 7.5000e-01, 8.0000e-01, 8.5000e-01, 9.0000e-01, 9.5000e-01,
                 9.8000e-01, 9.9000e-01, 9.9500e-01, 9.9750e-01, 9.9925e-01,
                 9.9975e-01, 9.9995e-01])

predicted_quantiles = np.array([-1.13813117e-01, -9.81501862e-02, -8.39762017e-02, -7.63831213e-02,
                                -7.16307834e-02, -6.58247992e-02, -5.48559278e-02, -4.31412645e-02,
                                -3.67169455e-02, -3.26498821e-02, -2.67682951e-02, -2.01001763e-02,
                                -1.29303625e-02, -9.45830531e-03, -6.80963136e-03, -4.11394192e-03,
                                -3.60043906e-03, -1.40941748e-03, -1.78500224e-04, -3.83639017e-05,
                                9.46297878e-05,  4.34772635e-04,  1.05301116e-03,  1.45855185e-03,
                                3.47404485e-03,  5.12253167e-03,  7.24938512e-03,  1.05004953e-02,
                                1.41525576e-02,  2.11500693e-02,  3.39386910e-02,  4.48816866e-02,
                                5.58338910e-02,  6.65146708e-02,  8.63869637e-02,  1.01823322e-01,
                                1.12997122e-01]
                               )

x, density, cdf = generate_smooth_pdf(predicted_quantiles, taus, plot=True)

# Calculate some basic statistics from the smoothed distribution
mean = np.trapz(x * density, x)
variance = np.trapz((x - mean)**2 * density, x)
skewness = np.trapz(((x - mean)/np.sqrt(variance))**3 * density, x)
est_kurtosis = np.trapz(((x - mean)/np.sqrt(variance))**4 * density, x)

print(f"Mean: {mean:.6f}")
print(f"Std Dev: {np.sqrt(variance):.6f}")
print(f"Skewness: {skewness:.6f}")
print(f"Kurtosis: {est_kurtosis:.6f}")

future_returns = np.array([0.00852014,  0.02728934,  0.02025293,  0.00663525,  0.02271252,
                           -0.00785648, -0.00722774,  0.03362522,  0.03608126, -0.07108913,
                           0.00586313, -0.0093281,  0.00795154, -0.00344934, -0.01954009,
                           -0.00035124, -0.1368499,  0.05677786, -0.01379864, -0.00930612,
                           -0.02804838,  0.01115552, -0.06972892,  0.03874677,  0.03887476,
                           0.00392328,  0.04815599, -0.06274908,  0.00119127,  0.00475051],)

realized_mean = np.mean(future_returns)
realized_std = np.std(future_returns)
realized_skew = skew(future_returns)
realized_kurt = kurtosis(future_returns)

print(f"Realized Mean: {realized_mean:.6f}")
print(f"Realized Std Dev: {realized_std:.6f}")
print(f"Realized Skewness: {realized_skew:.6f}")
print(f"Realized Kurtosis: {realized_kurt:.6f}")

true_dist = np.quantile(future_returns, taus)
quantiles_error = np.sqrt(np.mean((true_dist - predicted_quantiles)**2))
print(f"Quantiles Error: {quantiles_error:.6f}")

# Calculate the Wasserstein distance between the two distributions
wasserstein_dist = calculate_wasserstein(cdf, x, future_returns)
print(f"Wasserstein Distance: {wasserstein_dist:.6f}")

observred_returns = np.array([0.0035663, -0.00428105, -0.00970019, -0.00906619,  0.00544969,
                              0.01152743,  0.00962402,  0.01583709, -0.00665617,  0.00874891,
                              -0.00034842, -0.00454152, -0.01517062, -0.0194731, -0.01694651,
                              0.00543962,  0.00505057, -0.02293087,  0.01788029, -0.01054349,
                              0.02205039,  0.02017417, -0.01411459, -0.01431676,  0.,
                              -0.00868933,  0.01801859,  0.00285289, -0.01362515, -0.03004982],)

observed_mean = np.mean(observred_returns)
observed_std = np.std(observred_returns)


gaussian = norm(loc=observed_mean, scale=observed_std)
gaussian_pdf = gaussian.pdf(x)
gaussian_cdf = gaussian.cdf(x)
gaussian_quantiles = gaussian.ppf(taus)

gaussian_error = np.sqrt(np.mean((gaussian_quantiles - predicted_quantiles)**2))
print(f"Quantiles Error (Gaussian): {gaussian_error:.6f}")
# Calculate the Wasserstein distance between the two distributions
print(f"Wasserstein Distance: {calculate_wasserstein(gaussian_cdf, x, future_returns):.6f}")

plt.figure(figsize=(12, 6))
plt.plot(x, density, label='Predicted PDF')
plt.hist(future_returns, bins=20, density=True, alpha=0.5, label='True PDF')
plt.plot(x, gaussian_pdf, label='Gaussian PDF')
plt.title('Predicted vs. True PDF')
plt.xlabel('Returns')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()
