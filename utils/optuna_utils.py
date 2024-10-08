import optuna
import torch
from LSTM import LSTM_Model
from Dense import QuantileDense

def load_best_model(model_name):
    """
    Fetch the best parameters from the Optuna study and load the specified model.
    
    Args:
    model_name (str): Name of the model to load. Either 'lstm' or 'dense'.
    
    Returns:
    model: The loaded model with best parameters.
    best_params (dict): The best parameters found by Optuna.
    """
    if model_name.lower() not in ['lstm', 'dense']:
        raise ValueError("model_name must be either 'lstm' or 'dense'")

    # Load the appropriate study
    study_name = "LSTM" if model_name.lower() == 'lstm' else "Dense"
    study = optuna.load_study(
        study_name=study_name,
        storage=f"sqlite:///{study_name}.db"
    )

    best_params = study.best_params

    # Create and load the model
    if model_name.lower() == 'lstm':
        model = LSTM_Model(
            lstm_layers=best_params['raw_lstm_layers'],
            lstm_h=best_params['raw_lstm_h'],
            hidden_layers=[best_params[f'raw_hidden_layer_{i}'] for i in range(best_params['raw_hidden_layers'])],
            hidden_activation=best_params['hidden_activation'],
            market_lstm_layers=best_params['market_lstm_layers'],
            market_lstm_h=best_params['market_lstm_h'],
            market_hidden_layers=[best_params[f'market_hidden_layer_{i}'] for i in range(best_params['market_hidden_layers'])],
            market_hidden_activation=best_params['market_activation'],
            dropout=best_params['dropout'],
            layer_norm=best_params['use_layer_norm']
        )
    else:  # Dense model
        model = QuantileDense(
            raw_hidden_layers=[best_params[f'raw_hidden_layer_{i}'] for i in range(best_params['n_raw_hidden_layers'])],
            hidden_activation=best_params['hidden_activation'],
            market_hidden_layers=[best_params[f'market_hidden_layer_{i}'] for i in range(best_params['n_market_hidden_layers'])],
            market_activation=best_params['market_activation'],
            dropout=best_params['dropout'],
            layer_norm=best_params['layer_norm']
        )

    return model, best_params

# Example usage
if __name__ == "__main__":
    lstm_model, lstm_params = load_best_model('lstm')
    print("LSTM Best Parameters:", lstm_params)
    
    dense_model, dense_params = load_best_model('dense')
    print("Dense Best Parameters:", dense_params)