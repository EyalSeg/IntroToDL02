import sys

import torch as T
import torch.nn as nn
import seaborn as sns
from torch.utils.data import DataLoader

import utils
from data.sp500_data import SP500Dataset

sns.set_theme(style="darkgrid")

file = "../data/cache/sp500.csv"
DEVICE = T.device('cuda' if T.cuda.is_available() else 'cpu')
T.set_default_dtype(T.double)


if __name__ == "__main__":
    dataset = SP500Dataset(file, sample_ratio=.1)

    train_data, valid_data, test_data = utils.train_validate_test_split(dataset, 0.6, 0.2, 0.2)

    mse = nn.MSELoss()
    criterion = lambda output, input: mse(output.output_sequence, input)

    best_params = utils.LstmAEHyperparameters(
        epochs=3,
        seq_dim=1,
        batch_size=64,

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
        utils.train_and_measure(ae, train_dataloader, validate_loader, criterion, best_params,
                                verbose=True)

    utils.draw_reconstruction_sample(ae, test_data, n_samples=2, type="text")
    utils.plot_metric(train_losses, validate_losses, "Loss")

    test_set = next(iter(test_loader)).to(DEVICE)

    with T.no_grad():
        output = ae.forward(test_set)
        test_loss = criterion(output, test_set).item()

    print(f"Test loss: {test_loss}")