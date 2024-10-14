# Distribution Prediction

This repository contains the code for the paper [Generalized Distribution Prediction for Asset Returns]().

## Installation
After cloning the repository, install the required packages by running:
```bash
pip install -r requirements.txt
```

Additionally, you need to install PyTorch. Follow the official instructions [here](https://pytorch.org/get-started/locally/).

## Usage

### Data
To fetch the necessary data, run the `data_fetch.py` module. This only needs to be done once, as the script retrieves data via the Yahoo Finance API:
```bash
python3 data_fetch.py
```

### Reproducing Results
Note: Results may vary slightly due to hardware differences. The models were initially trained on an NVIDIA GeForce RTX 2060 SUPER.

You can reproduce the results at three levels:

#### 1. Pre-Trained Models
For the fastest reproduction, use the pre-trained models included in the repository. Simply run all the cells in `results.ipynb`.

#### 2. Training Models
To retrain the models with optimal hyperparameters from Optuna, first delete the existing models (`dense_model.pth` and `lstm_model.pth`) from the directory, then run all cells in `results.ipynb`.

#### 3. Rerun Optuna Trials
Rerunning the Optuna trials requires significant time (approximately 2 days as per the current configuration). To start fresh, delete `Dense.db` and `LSTM.db` from your directory, then run:
```bash
python3 optuna_train.py
```
If training is interrupted, rerun the script to resume from where it stopped. Once the trials are complete, rerun all cells in `results.ipynb`.

## License

[MIT License](https://choosealicense.com/licenses/mit/)