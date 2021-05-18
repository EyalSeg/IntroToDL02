import torch as T
import torch.nn as nn
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt

import utils

from data.sp500_data import SP500Dataset
from experiment import Experiment

file = "../data/cache/sp500.csv"

if __name__ == "__main__":
    dataset = SP500Dataset(file, normalize=True)

    train_data, valid_data, test_data = utils.train_validate_test_split(dataset, 0.6, 0.2, 0.2)

    mse = nn.MSELoss()
    cel = nn.CrossEntropyLoss()

    prediction = False

    if prediction:
        criterion = lambda output, input: mse(output.output_sequence[1:], input[:-1])
    else:
        criterion = lambda output, input: mse(output.output_sequence, input)

    experiment = Experiment(criterion)

    hyperparameters = utils.LstmAEHyperparameters(
        epochs=250,
        seq_dim=1,
        batch_size=1024,

        num_layers=1,
        lr=0.0005,
        latent_size=64,
        grad_clipping=1
    )

    ae = hyperparameters.create_ae()

    train_dataloader = DataLoader(train_data, batch_size=hyperparameters.batch_size, shuffle=True)
    validate_loader = DataLoader(valid_data, batch_size=hyperparameters.batch_size)
    test_loader = DataLoader(test_data, batch_size=hyperparameters.batch_size)

    results_df = experiment.run(ae, train_dataloader, test_loader, hyperparameters, verbose=True, measure_every=10)

    utils.draw_reconstruction_sample(ae, test_data, n_samples=2)
    sns.lineplot(data=results_df, dashes=False)

    if prediction:
        plt.title("S&P500 Data Set Prediction Loss")
    else:
        plt.title("S&P500 Data Set Training Loss")

    plt.show()

    print(f"Test loss: {results_df.iloc[-1]['test_loss']}")

