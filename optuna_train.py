import json
import optuna

from LSTM import objective as lstm_objective
from Dense import objective as dense_objective

with open('config.json', encoding="utf8") as f:
    CONFIG = json.load(f)


def train_lstm():
    n_trials = 100
    sampler = optuna.samplers.TPESampler(seed=271198)
    study = optuna.create_study(
        direction="minimize",
        study_name="LSTM",
        storage=CONFIG["general"]["db_path"],
        load_if_exists=True,
        sampler=sampler,
    )
    n_trials = n_trials - len(study.trials)
    study.optimize(lstm_objective, n_trials=n_trials, n_jobs=1)
    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)


def train_dense():
    n_trials = 100
    sampler = optuna.samplers.TPESampler(seed=271198)
    study = optuna.create_study(
        direction="minimize",
        study_name="Dense",
        storage=CONFIG["general"]["db_path"],
        load_if_exists=True,
        sampler=sampler,
    )
    finished_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    n_trials = n_trials - len(finished_trials)
    study.optimize(dense_objective, n_trials=n_trials, n_jobs=1)
    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)


if __name__ == "__main__":
    train_lstm()
    train_dense()
