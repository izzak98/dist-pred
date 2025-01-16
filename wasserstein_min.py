import json
import numpy as np
from scipy.stats import norm
import optuna

from utils.result_utils import calculate_wasserstein, generate_smooth_pdf

with open("config.json", encoding="utf8") as f:
    CONFIG = json.load(f)


def wasserstein_objective(trial, quant_values, use_hybrid=False):
    quantiles = CONFIG["general"]["quantiles"]
    min_density = trial.suggest_float("min_density", 1e-6, 1e-1, step=1e-6)
    eps = trial.suggest_float("eps", 1e-6, 1e-1, step=1e-6)
    window = trial.suggest_int("window", 3, 201, step=2)
    group_names = [
        "cryptocurrencies",
        "currency pairs",
        "commodities",
        "euro stoxx 50",
        "s&p 500",
        "nikkei 225"
    ]
    all_quantiles = []
    all_future_returns = []
    all_observed_returns = []
    for group_name in group_names:
        if group_name not in quant_values:
            continue
        all_quantiles.extend(quant_values[group_name]["all_pred_quantiles"])
        all_future_returns.extend(quant_values[group_name]["future_returns"])
        all_observed_returns.extend(quant_values[group_name]["observed_returns"])
    all_quantiles = np.array(all_quantiles)
    all_future_returns = np.array(all_future_returns)

    total_wasserstein = 0
    for i in range(len(all_quantiles)):
        estimated_quantiles = all_quantiles[i]
        future_returns = all_future_returns[i]
        if use_hybrid:
            observed_returns = all_observed_returns[i]
            observed_mean = np.mean(observed_returns)
            observed_std = np.std(observed_returns)
            baseline_quantiles = norm.ppf(quantiles, loc=observed_mean, scale=observed_std)
            hybrid_quantiles = (baseline_quantiles + estimated_quantiles)/2
            estimated_quantiles = hybrid_quantiles
        grid, _, cdf = generate_smooth_pdf(estimated_quantiles, np.array(
            quantiles), min_density=min_density, eps=eps, window=window)
        wasserstein = calculate_wasserstein(cdf, grid, future_returns)
        total_wasserstein += wasserstein
    return total_wasserstein/len(all_quantiles)


def get_best_lstm_pdf_params(val_values):
    qlstm_pdf_study = optuna.create_study(direction="minimize",
                                          study_name="wasserstein_distance_qlstm",
                                          storage="sqlite:///wasserstein_distance.db",
                                          load_if_exists=True)
    completed_trials = qlstm_pdf_study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    n_trials = 100 - len(completed_trials)
    def objective(trial): return wasserstein_objective(trial, val_values, use_hybrid=False)
    qlstm_pdf_study.optimize(objective, n_trials=n_trials, n_jobs=1)

    return qlstm_pdf_study.best_params


def get_best_hybrid_pdf_params(val_values):
    hybrid_qlstm_pdf_study = optuna.create_study(direction="minimize",
                                                 study_name="wasserstein_distance_hybrid_qlstm",
                                                 storage="sqlite:///wasserstein_distance.db",
                                                 load_if_exists=True)
    completed_trials = hybrid_qlstm_pdf_study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    n_trials = 100 - len(completed_trials)
    def objective(trial): return wasserstein_objective(trial, val_values, use_hybrid=True)
    hybrid_qlstm_pdf_study.optimize(objective, n_trials=n_trials, n_jobs=1)

    return hybrid_qlstm_pdf_study.best_params
