import optuna

from LSTM import objective as lstm_objective
from Dense import objective as dense_objective


def train_lstm():
    n_trials = 100
    sampler = optuna.samplers.TPESampler(seed=271198)
    study = optuna.create_study(
        direction="minimize",
        study_name="LSTM",
        storage="sqlite:///LSTM.db",
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
        storage="sqlite:///Dense.db",
        load_if_exists=True,
        sampler=sampler,
        )
    n_trials = n_trials - len(study.trials)
    study.optimize(dense_objective, n_trials=n_trials, n_jobs=1)
    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)

if __name__ == "__main__":
    train_lstm()
    train_dense()
