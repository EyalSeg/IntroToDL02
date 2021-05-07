import math
from random import random

import torch as T
import torch.nn as nn
import pandas as pd
import seaborn as sns
import numpy as np

from torch.utils.data import DataLoader
from data.synthetic_data import SyntheticDataset

import utils

sns.set_theme(style="darkgrid")

file = "../../data/cache/S&P500.csv"
DEVICE = T.device('cuda' if T.cuda.is_available() else 'cpu')
T.set_default_dtype(T.double)

if __name__ == "__main__":
    train_data, valid_data, test_data = utils.fetch_and_split_data(file)

    mse = nn.MSELoss()
    criterion = lambda output, input: mse(output.output_sequence, input)

    best_params = utils.tune_parameters(train_data, valid_data, criterion, should_tune=False)

    find_best_seed = False

    if find_best_seed:
        best_seed = utils.cross_validation(dataset, best_params, iterations=1000)
        print(f"Best seed is: {best_seed}")

        # Re-Build Data Out of "Best-Seed"
        train_data, valid_data, test_data = utils.train_validate_test_split(dataset, 0.6, 0.2, 0.2, seed=42)

    ae = best_params.create_ae()

    train_dataloader = DataLoader(train_data, batch_size=best_params.batch_size, shuffle=True)
    validate_loader = DataLoader(valid_data, batch_size=len(valid_data))
    test_loader = DataLoader(test_data, batch_size=len(test_data))

    train_losses, validate_losses, _, _ = \
        utils.train_and_measure(ae, train_dataloader, validate_loader, criterion, best_params,
                                verbose=True, make_nans_average_check=True)

    utils.print_results_and_plot_graph(ae, criterion, test_data, test_loader, best_params, train_losses,
                                       validate_losses,
                                       None, None, "lstm_ae_snp500", is_supervised=False)
