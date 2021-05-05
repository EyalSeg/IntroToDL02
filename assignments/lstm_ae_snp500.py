import torch as T
import torch.nn as nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os

from torch.utils.data import DataLoader

import utils

from data.synthetic_data import SyntheticDataset
from grid_search import tune
from utils import LstmAEHyperparameters

sns.set_theme(style="darkgrid")

file = "../data/cache/S&P500.csv"
DEVICE = T.device('cuda' if T.cuda.is_available() else 'cpu')
T.set_default_dtype(T.double)

import math


def daily_max_stock(stocks, title="Stocks"):
    sd = stocks['date'][:round(len(stocks) * .1)]
    sdv = stocks['dvolume'][:round(len(stocks) * .1)]

    plt.plot(range(len(sdv)), sdv)
    plt.title(title)
    plt.xlabel("Days")
    plt.ylabel("Max Dollar Volume")
    plt.show()

    print(f"First Date is: {sd.values[0]}\nLast Date is: {sd.values[math.floor(len(stocks) * .1)]}")


def plot_max_stocks():
    stocks = pd.read_csv('../data/cache/S&P500.csv')

    google_stocks = stocks[stocks['symbol'] == 'GOOGL']
    amazon_stocks = stocks[stocks['symbol'] == 'AMZN']

    daily_max_stock(google_stocks, "Google Stocks")
    daily_max_stock(amazon_stocks, "Amazon Stocks")


if __name__ == "__main__":
    plot_max_stocks()

    dataset = SyntheticDataset(file)

    train_data, valid_data, test_data = utils.train_validate_test_split(dataset, 0.6, 0.2, 0.2, seed=None)

    mse = nn.MSELoss()
    criterion = lambda output, input: mse(output.output_sequence, input)

    should_tune = False  # change to false to use predefined hyperparameters
    if should_tune:
        param_choices = {
            'epochs': [700],
            'seq_dim': [1],
            'batch_size': [128],
            'num_layers': [2],
            'latent_size': [256],
            'lr': [0.0001, 0.001],
            # 'grad_clipping': [None, 0.01, 0.1, 0.5, 1, 2],
            'grad_clipping': [None, 1],
        }


        def tune_objective(**params):
            hyperparameters = LstmAEHyperparameters(**params)
            return utils.evaluate_hyperparameters(train_data, valid_data, criterion, hyperparameters)


        best_params, best_loss = tune(tune_objective, param_choices, "minimize", workers=4)
        best_params = LstmAEHyperparameters(**best_params)

        print("Best parameters are:")
        print(f"\tlatent size: {best_params.latent_size}")
        print(f"\tlr: {best_params.lr}")
        print(f"\tgrad_clipping: {best_params.grad_clipping}")
        print(f"\tnum_layers: {best_params.num_layers}")

    else:
        best_params = LstmAEHyperparameters(
            epochs=700,
            seq_dim=1,
            batch_size=128,

            num_layers=2,
            lr=0.001,
            latent_size=256,
            grad_clipping=None
        )

    best_seed = utils.cross_validation(dataset, best_params, iterations=1000)

    # Re-Build Data Out of "Best-Seed"
    train_data, valid_data, test_data = utils.train_validate_test_split(dataset, 0.6, 0.2, 0.2, seed=best_seed)

    ae = best_params.create_ae()

    train_dataloader = DataLoader(train_data, batch_size=best_params.batch_size, shuffle=True)
    validate_loader = DataLoader(valid_data, batch_size=len(valid_data))
    test_loader = DataLoader(test_data, batch_size=len(test_data))

    train_losses, validate_losses, train_accuracies, validation_accuracies = \
        utils.train_and_measure(ae, train_dataloader, validate_loader, criterion, best_params, verbose=True)

    utils.draw_sample(ae, test_data, n_samples=2, title="S&P500 Data Set")

    df = pd.DataFrame.from_dict({"training set": train_losses,
                                 "validation set": validate_losses})
    df.index.name = "Epoch"

    sns.lineplot(data=df, dashes=False)
    lr_str = "{:12.7f}".format(best_params.lr)
    plt.title("Learn Loss")
    plt.ylabel("Loss")
    plt.show()

    df = pd.DataFrame.from_dict({"training set": train_accuracies,
                                 "validation set": validation_accuracies})
    df.index.name = "Epoch"

    sns.lineplot(data=df, dashes=False)
    lr_str = "{:12.7f}".format(best_params.lr)
    plt.title("Learn Loss")
    plt.ylabel("Accuracy")
    plt.show()

    test_set = next(iter(test_loader)).to(DEVICE)

    with T.no_grad():
        output = ae.forward(test_set)
        test_loss = criterion(output, test_set).item()

    print(f"Test loss: {test_loss}")
