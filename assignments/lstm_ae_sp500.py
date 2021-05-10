import torch.nn as nn
from torch.utils.data import DataLoader

import utils

from data.sp500_data import SP500Dataset

file = "../data/cache/sp500.csv"

if __name__ == "__main__":
    dataset = SP500Dataset(file, normalize=True, sample_ratio=.1)

    train_data, valid_data, test_data = utils.train_validate_test_split(dataset, 0.6, 0.2, 0.2)

    mse = nn.MSELoss()
    criterion = lambda output, input: mse(output.output_sequence, input)

    hyperparameters = utils.LstmAEHyperparameters(
        epochs=10,
        seq_dim=1,
        batch_size=64,

        num_layers=2,
        lr=0.001,
        latent_size=256,
        grad_clipping=2 ** 8
    )

    ae = hyperparameters.create_ae()

    train_dataloader = DataLoader(train_data, batch_size=hyperparameters.batch_size, shuffle=True)
    validate_loader = DataLoader(valid_data, batch_size=hyperparameters.batch_size)
    test_loader = DataLoader(test_data, batch_size=hyperparameters.batch_size)

    train_losses, validate_losses = \
        utils.train_and_measure(ae, train_dataloader, validate_loader, criterion, hyperparameters, verbose=True)

    utils.draw_reconstruction_sample(ae, test_data, n_samples=2, type="text")
    utils.plot_metric(train_losses, validate_losses, "Loss")