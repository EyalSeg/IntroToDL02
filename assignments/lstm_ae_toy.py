import torch as T
import torch.nn as nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

import utils

from data.synthetic_data import SyntheticDataset
from grid_search import tune
from utils import LstmAEHyperparameters

sns.set_theme(style="darkgrid")

file = "../data/cache/synthetic.csv"
DEVICE = T.device('cuda' if T.cuda.is_available() else 'cpu')
T.set_default_dtype(T.double)


if __name__ == "__main__":
    dataset = SyntheticDataset(file)

    train_data, valid_data, test_data = utils.train_validate_test_split(dataset, 0.6, 0.2, 0.2)

    mse = nn.MSELoss()
    criterion = lambda output, input: mse(output.output_sequence, input)

    should_tune = False # change to false to use predefined hyperparameters
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
            batch_size=256,

            num_layers=2,
            lr=0.001,
            latent_size=256,
            grad_clipping=2 ** 8
        )

    ae = best_params.create_ae()

    train_dataloader = DataLoader(train_data, batch_size=best_params.batch_size, shuffle=True)
    validate_loader = DataLoader(valid_data, batch_size=len(valid_data))
    test_loader = DataLoader(test_data, batch_size=len(test_data))

    train_losses, validate_losses = \
        utils.train_and_measure(ae, train_dataloader, validate_loader, criterion, best_params)

    utils.draw_reconstruction_sample(ae, test_data, n_samples=2)
    utils.plot_metric(train_losses, validate_losses, "Loss")

    test_set = next(iter(test_loader)).to(DEVICE)

    with T.no_grad():
        output = ae.forward(test_set)
        test_loss = criterion(output, test_set).item()

    print(f"Test loss: {test_loss}")

