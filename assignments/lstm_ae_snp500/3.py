import torch as T
import torch.nn as nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

import utils
from ae_wrappers.ae_classification_wrapper import AutoencoderClassifierOutput
from assignments.lstm_ae_mnist import AEClassifierHyperparameters

sns.set_theme(style="darkgrid")

file = "../../data/cache/S&P500.csv"
DEVICE = T.device('cuda' if T.cuda.is_available() else 'cpu')
T.set_default_dtype(T.double)

if __name__ == "__main__":
    dataset = (pd.read_csv(file, index_col=False))

    # Change This Number to The Best Seed.
    best_seed = 43

    train_data, valid_data, test_data = utils.train_validate_test_split(dataset, 0.6, 0.2, 0.2, seed=best_seed)

    # Change Those Parameters to The Best Parameters.
    hyperparameters = AEClassifierHyperparameters(
        epochs=700,
        seq_dim=1,
        batch_size=64,
        n_classes=10,

        num_layers=1,
        lr=0.001,
        latent_size=256,
        grad_clipping=None
    )

    ae = hyperparameters.create_ae()

    train_dataloader = DataLoader(train_data, batch_size=hyperparameters.batch_size, shuffle=True)
    validate_dataloader = DataLoader(train_data, batch_size=len(valid_data), shuffle=True)
    test_loader = DataLoader(test_data, batch_size=len(test_data))

    mse = nn.MSELoss()
    cel = nn.CrossEntropyLoss()


    def criterion(output: AutoencoderClassifierOutput, input_sequence, labels):
        reconstruction_loss = mse(output.output_sequence, input_sequence)
        classification_loss = cel(output.label_predictions, labels)

        # return reconstruction_loss + classification_loss
        return reconstruction_loss


    train_losses, validate_losses, train_accuracies, validation_accuracies = \
        utils.train_and_measure(ae, train_dataloader, validate_dataloader, criterion, hyperparameters, supervised=True)

    utils.draw_sample(ae, test_data, n_samples=2, title="S&P500 Data Set")

    df = pd.DataFrame.from_dict({"training set": train_losses,
                                 "validation set": validate_losses})
    df.index.name = "Epoch"

    sns.lineplot(data=df, dashes=False)
    lr_str = "{:12.7f}".format(hyperparameters.lr)
    plt.title("Learn Loss")
    plt.ylabel("Loss")
    plt.show()

    df = pd.DataFrame.from_dict({"training set": train_accuracies,
                                 "validation set": validation_accuracies})
    df.index.name = "Epoch"

    sns.lineplot(data=df, dashes=False)
    lr_str = "{:12.7f}".format(hyperparameters.lr)
    plt.title("Learn Loss")
    plt.ylabel("Accuracy")
    plt.show()

    test_set = next(iter(test_loader)).to(DEVICE)

    with T.no_grad():
        output = ae.forward(test_set)
        test_loss = criterion(output, test_set).item()

    print(f"Test loss: {test_loss}")