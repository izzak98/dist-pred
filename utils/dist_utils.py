import numpy as np
from scipy.stats import norm, wasserstein_distance, t, ks_2samp
from typing import Dict, List, Union
import pandas as pd


def wasserstein_metric(quantiles: np.ndarray, real_returns: np.ndarray) -> float:
    """Calculate Wasserstein distance for a single prediction window."""
    real_cdf = np.sort(real_returns)
    pred_cdf = np.sort(quantiles)
    return wasserstein_distance(real_cdf, pred_cdf)


def ks_statistic(quantiles: np.ndarray, real_returns: np.ndarray) -> float:
    """Calculate KS statistic for a single prediction window."""
    return ks_2samp(quantiles, real_returns).statistic


def quantile_coverage_error(quantiles: np.ndarray, real_returns: np.ndarray,
                            quant_probs: np.ndarray) -> float:
    """Calculate QCE for a single prediction window."""
    empirical_proportions = np.array([
        np.mean(real_returns <= q) for q in quantiles
    ])
    errors = np.abs(empirical_proportions - quant_probs)
    return np.mean(errors)


def historical_baseline(real_returns: np.ndarray, quant_probs: np.ndarray) -> np.ndarray:
    """Calculate historical baseline quantiles."""
    return np.percentile(real_returns, quant_probs * 100)


def gaussian_baseline(real_returns: np.ndarray, quant_probs: np.ndarray) -> np.ndarray:
    """Calculate Gaussian baseline quantiles."""
    mu, sigma = np.mean(real_returns), np.std(real_returns)
    return norm.ppf(quant_probs, loc=mu, scale=sigma)


def student_t_baseline(real_returns: np.ndarray, quant_probs: np.ndarray) -> np.ndarray:
    """Calculate Student's t baseline quantiles."""
    df, loc, scale = 5, np.mean(real_returns), np.std(real_returns)
    return t.ppf(quant_probs, df, loc=loc, scale=scale)


def evaluate_predictions_2d(predicted_quantiles: np.ndarray, future_returns: np.ndarray,
                            observed_returns: np.ndarray, quant_probs: np.ndarray) -> Dict:
    """
    Evaluate distributional predictions across multiple time windows.

    Args:
        predicted_quantiles: Shape (n_windows, n_quantiles)
        future_returns: Shape (n_windows, n_future_steps)
        observed_returns: Shape (n_windows, n_observed_steps)
        quant_probs: Shape (n_quantiles,)

    Returns:
        Dict containing mean and std of metrics across windows
    """
    n_windows = predicted_quantiles.shape[0]
    metrics_per_window = []

    for i in range(n_windows):
        # Generate baseline predictions for this window
        hist_baseline = historical_baseline(observed_returns[i], quant_probs)
        gauss_baseline = gaussian_baseline(observed_returns[i], quant_probs)
        t_baseline = student_t_baseline(observed_returns[i], quant_probs)

        # Calculate metrics for this window
        window_metrics = {
            'qce': {
                'historical': quantile_coverage_error(hist_baseline, future_returns[i], quant_probs),
                'gaussian': quantile_coverage_error(gauss_baseline, future_returns[i], quant_probs),
                'student_t': quantile_coverage_error(t_baseline, future_returns[i], quant_probs),
                'qlstm': quantile_coverage_error(predicted_quantiles[i], future_returns[i], quant_probs)
            },
            'wasserstein': {
                'historical': wasserstein_metric(hist_baseline, future_returns[i]),
                'gaussian': wasserstein_metric(gauss_baseline, future_returns[i]),
                'student_t': wasserstein_metric(t_baseline, future_returns[i]),
                'qlstm': wasserstein_metric(predicted_quantiles[i], future_returns[i])
            },
            'ks': {
                'historical': ks_statistic(hist_baseline, future_returns[i]),
                'gaussian': ks_statistic(gauss_baseline, future_returns[i]),
                'student_t': ks_statistic(t_baseline, future_returns[i]),
                'qlstm': ks_statistic(predicted_quantiles[i], future_returns[i])
            }
        }
        metrics_per_window.append(window_metrics)

    # Calculate aggregate statistics
    aggregate_metrics = {
        'mean': {},
        'std': {},
        'min': {},
        'max': {},
        'median': {}
    }

    # Helper function to extract specific metric across windows
    def extract_metric(metrics_list, metric_path):
        values = []
        for m in metrics_list:
            curr = m
            for key in metric_path:
                curr = curr[key]
            values.append(curr)
        return np.array(values)

    # Metrics to aggregate
    metric_paths = [
        ('qce', model) for model in ['historical', 'gaussian', 'student_t', 'qlstm']
    ] + [
        ('wasserstein', model) for model in ['historical', 'gaussian', 'student_t', 'qlstm']
    ] + [
        ('ks', model) for model in ['historical', 'gaussian', 'student_t', 'qlstm']
    ]

    # Calculate statistics for each metric
    for path in metric_paths:
        values = extract_metric(metrics_per_window, path)
        metric_name = '_'.join(path)

        aggregate_metrics['mean'][metric_name] = np.mean(values)
        aggregate_metrics['std'][metric_name] = np.std(values)
        aggregate_metrics['min'][metric_name] = np.min(values)
        aggregate_metrics['max'][metric_name] = np.max(values)
        aggregate_metrics['median'][metric_name] = np.median(values)

    return aggregate_metrics


def print_aggregate_metrics(metrics: Dict):
    """Pretty print the aggregate metrics."""
    stats = ['mean', 'std', 'median', 'min', 'max']
    metric_types = ['qce', 'wasserstein', 'ks']

    for metric_type in metric_types:
        print(f"\n{metric_type.upper()} Metrics:")
        relevant_metrics = {k: v for k, v in metrics['mean'].items() if metric_type in k}

        for metric_name, mean_value in relevant_metrics.items():
            print(f"\n{metric_name}:")
            for stat in stats:
                print(f"  {stat}: {metrics[stat][metric_name]:.4f}")


def format_distribution_results(results_by_market):
    """
    Format distribution metrics results into a pretty DataFrame.

    Args:
        results_by_market (dict): Dictionary with market names as keys and metric DataFrames as values
                                 Each DataFrame should contain 'metric', 'mean', 'std' columns

    Returns:
        pd.DataFrame: Beautifully formatted results table
    """
    # Initialize empty lists to store formatted results
    rows = []

    # Define metric display names and order
    metric_order = {
        'qLSTM_wasserstein': 'Wasserstein Distance',
        'qLSTM_ks': 'KS Statistic',
        'qLSTM_qce': 'Quantile Coverage Error',
        'historical_wasserstein': 'Historical Wasserstein',
        'historical_ks': 'Historical KS',
        'historical_qce': 'Historical QCE',
        'gaussian_wasserstein': 'Gaussian Wasserstein',
        'gaussian_ks': 'Gaussian KS',
        'gaussian_qce': 'Gaussian QCE',
        'student_t_wasserstein': 'Student-t Wasserstein',
        'student_t_ks': 'Student-t KS',
        'student_t_qce': 'Student-t QCE'
    }

    # Group metrics by type
    metric_groups = {
        'LSTM Metrics': ['qLSTM_wasserstein', 'qLSTM_ks', 'qLSTM_qce'],
        'Historical Baseline': ['historical_wasserstein', 'historical_ks', 'historical_qce'],
        'Gaussian Baseline': ['gaussian_wasserstein', 'gaussian_ks', 'gaussian_qce'],
        'Student-t Baseline': ['student_t_wasserstein', 'student_t_ks', 'student_t_qce']
    }

    # Create multi-level table
    for group_name, metrics in metric_groups.items():
        # Add group header
        rows.append(pd.Series({'metric': group_name}))

        for metric in metrics:
            metric_values = {}
            metric_values['metric'] = f"  {metric_order[metric]}"  # Indent metric names

            # Add values for each market
            for market, results in results_by_market.items():
                if metric in results['metric'].values:
                    row = results[results['metric'] == metric].iloc[0]
                    metric_values[f"{market}_mean"] = f"{row['mean']:.4f}"
                    metric_values[f"{market}_std"] = f"({row['std']:.4f})"

            rows.append(pd.Series(metric_values))

    # Create DataFrame
    results_df = pd.DataFrame(rows)

    # Style the DataFrame
    def highlight_groups(s):
        return ['font-weight: bold' if not str(s['metric']).startswith('  ') else ''
                for _ in s]

    styled_df = results_df.style\
        .apply(highlight_groups, axis=1)\
        .format(precision=4)\
        .set_properties(**{
            'text-align': 'right',
            'padding': '3px 10px'
        })\
        .set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'center'),
                                         ('font-weight', 'bold'),
                                         ('background-color', '#f0f0f0')]},
            {'selector': 'td', 'props': [('text-align', 'right')]},
            {'selector': 'td:first-child', 'props': [('text-align', 'left')]}
        ])

    return styled_df


class DistributionAnalyzer:
    def __init__(self, quant_probs: np.ndarray):
        """
        Initialize analyzer with quantile probabilities.

        Args:
            quant_probs: Array of quantile probabilities
        """
        self.quant_probs = quant_probs

    def evaluate_market(self,
                        predicted_quantiles: np.ndarray,
                        future_returns: np.ndarray,
                        observed_returns: np.ndarray) -> Dict:
        """
        Evaluate predictions for a single market.

        Args:
            predicted_quantiles: Shape (n_windows, n_quantiles)
            future_returns: Shape (n_windows, n_future_steps)
            observed_returns: Shape (n_windows, n_observed_steps)

        Returns:
            Dict with aggregated metrics
        """
        return evaluate_predictions_2d(
            predicted_quantiles,
            future_returns,
            observed_returns,
            self.quant_probs
        )

    def evaluate_markets(self,
                         predictions_by_market: Dict[str, Dict[str, np.ndarray]]) -> pd.DataFrame:
        """
        Evaluate predictions across multiple markets and format results.

        Args:
            predictions_by_market: Dict with keys as market names and values as dicts containing:
                - 'predicted_quantiles': Shape (n_windows, n_quantiles)
                - 'future_returns': Shape (n_windows, n_future_steps)
                - 'observed_returns': Shape (n_windows, n_observed_steps)

        Returns:
            Styled pandas DataFrame with formatted results
        """
        results_by_market = {}

        for market_name, data in predictions_by_market.items():
            # Evaluate metrics for this market
            metrics = self.evaluate_market(
                data['predicted_quantiles'],
                data['future_returns'],
                data['observed_returns']
            )

            # Convert to DataFrame format
            market_results = []
            for metric_type in ['wasserstein', 'ks', 'qce']:
                for model in ['qlstm', 'historical', 'gaussian', 'student_t']:
                    metric_name = f"{model}_{metric_type}"
                    market_results.append({
                        'metric': metric_name,
                        'mean': metrics['mean'][f"{metric_type}_{model}"],
                        'std': metrics['std'][f"{metric_type}_{model}"],
                        'median': metrics['median'][f"{metric_type}_{model}"],
                        'min': metrics['min'][f"{metric_type}_{model}"],
                        'max': metrics['max'][f"{metric_type}_{model}"]
                    })

            results_by_market[market_name] = pd.DataFrame(market_results)

        # Format results into pretty table
        return format_distribution_results(results_by_market)


# Example usage:
if __name__ == "__main__":
    # Example data setup
    n_windows = 100
    n_quantiles = 37
    n_future_steps = 22
    n_observed_steps = 250

    quant_probs = np.linspace(0.0001, 0.9999, n_quantiles)

    # Generate sample data for different markets
    markets = ['Crypto', 'Forex', 'Stocks', 'Commodities']
    predictions_by_market = {}

    for market in markets:
        predictions_by_market[market] = {
            'predicted_quantiles': np.random.normal(0, 1, (n_windows, n_quantiles)),
            'future_returns': np.random.normal(0, 1, (n_windows, n_future_steps)),
            'observed_returns': np.random.normal(0, 1, (n_windows, n_observed_steps))
        }

    # Create analyzer and evaluate
    analyzer = DistributionAnalyzer(quant_probs)
    formatted_results = analyzer.evaluate_markets(predictions_by_market)

    # Display results
    print("\nFormatted Results Table:")
    # print(formatted_results)  # For Jupyter notebook
    # For non-Jupyter environments:
    print(formatted_results.to_string())
