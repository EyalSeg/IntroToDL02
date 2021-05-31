import torch as T
import torch.nn as nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset

import utils

from data.synthetic_data import SyntheticDataset
from grid_search import tune
from utils import LstmAEHyperparameters
from experiment import Experiment

sns.set_theme(style="darkgrid")

file = "../data/cache/synthetic.csv"
DEVICE = T.device('cuda' if T.cuda.is_available() else 'cpu')
T.set_default_dtype(T.double)


if __name__ == "__main__":
    dataset = SyntheticDataset(file)
    # dataset = Subset(dataset, list(range(100)))

    train_data, valid_data, test_data = utils.train_validate_test_split(dataset, 0.6, 0.2, 0.2)

    mse = nn.MSELoss()
    criterion = lambda output, input: mse(output.output_sequence, input)
    experiment = Experiment(criterion)

    should_tune = False # change to false to use predefined hyperparameters
    if should_tune:
        param_choices = {
            'epochs': [1000],
            'seq_dim': [1],
            'batch_size': [1024],
            'num_layers': [2],
            'latent_size': [16, 32, 64, 256],
            'lr': [0.1, 0.001, 0.0001],
            'grad_clipping': [None, 1, 0.01, 0.001, 0.0001],
        }

        def tune_objective(**params):
            hyperparameters = LstmAEHyperparameters(**params)
            return utils.evaluate_hyperparameters(train_data, valid_data, criterion, hyperparameters)

        best_params, best_loss = tune(tune_objective, param_choices, "minimize", workers=1)
        best_params = LstmAEHyperparameters(**best_params)

        print("Best parameters are:")
        print(best_params)

    else:
        best_params = LstmAEHyperparameters(
            epochs=3000,
            seq_dim=1,
            batch_size=1024,

            num_layers=1,
            lr=0.01,
            latent_size=32,
            grad_clipping=0.5
        )

    ae = best_params.create_ae()

    train_dataloader = DataLoader(train_data, batch_size=best_params.batch_size, shuffle=True)
    validate_loader = DataLoader(valid_data, batch_size=len(valid_data))
    test_loader = DataLoader(test_data, batch_size=len(test_data))

    results_df = experiment.run(ae, train_dataloader, test_loader, best_params, verbose=True, measure_every=10)

    utils.draw_reconstruction_sample(ae, test_data, n_samples=2, title="Reconstruction Sample")

    sns.lineplot(data=results_df, dashes=False)
    plt.title("Learning Loss")
    plt.show()

    print(f"Test loss: {results_df.iloc[-1]['test_loss']}")




