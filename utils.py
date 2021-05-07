import collections
import math

import torch as T
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from torchvision.transforms import ToTensor
from dataclasses import dataclass
from torch.utils.data import DataLoader
from typing import Union
from grid_search import tune
from data.synthetic_data import SyntheticDataset

from lstm_ae import LstmAutoEncoder

DEVICE = T.device('cuda' if T.cuda.is_available() else 'cpu')
T.set_default_dtype(T.double)
sns.set_theme(style="darkgrid")


@dataclass(frozen=True)
class LstmAEHyperparameters:
    lr: float
    grad_clipping: Union[float, None]
    seq_dim: int
    latent_size: int
    num_layers: int
    batch_size: int
    epochs: int

    def create_ae(self):
        return LstmAutoEncoder(self.seq_dim, self.latent_size, self.num_layers)


def load_torch_dataset(dataset, transform=None, train_validate_split=(2 / 3, 1 / 3), cache_path='/data/cache'):
    if transform:
        train_data = dataset(
            root=cache_path,
            train=True,
            download=True,
            transform=transform
        )
    else:
        train_data = dataset(
            root=cache_path,
            train=True,
            download=True,
        )
    train_len = int(len(train_data) * train_validate_split[0])
    validate_len = int(len(train_data) * train_validate_split[1])

    train_data, validate_data = T.utils.data.random_split(train_data, [train_len, validate_len])

    test_data = dataset(
        root=cache_path,
        train=False,
        download=True,
        transform=ToTensor()
    )

    return train_data, validate_data, test_data


def train_validate_test_split(dataset, train_ratio=0.6, validate_ratio=0.2, test_ratio=0.2, seed=None):
    train_n = int(train_ratio * len(dataset))
    validate_n = int(validate_ratio * len(dataset))
    test_n = len(dataset) - train_n - validate_n

    generator = None

    if seed is not None:
        generator = T.Generator().manual_seed(seed)

    train_data, valid_data, test_data = T.utils.data.random_split(
        dataset, (train_n, validate_n, test_n), generator=generator)

    return train_data, valid_data, test_data


def cross_validation(dataset, best_params, iterations=1000):
    train_losses_data = [], validate_losses_data = [], train_accuracies_data = [], validation_accuracies_data = [], test_loss_data = []
    seeds = []

    for i in range(iterations):
        seed = np.random.randint(0, iterations * 2)

        while np.any(np.isin(seeds, seed)):
            seed = np.random.randint(0, iterations * 2)

        train_data, valid_data, test_data = train_validate_test_split(dataset, 0.6, 0.2, 0.2, seed=seed)

        mse = nn.MSELoss()
        criterion = lambda output, input: mse(output.output_sequence, input)

        ae = best_params.create_ae()

        train_dataloader = DataLoader(train_data, batch_size=best_params.batch_size, shuffle=True)
        validate_loader = DataLoader(valid_data, batch_size=len(valid_data))
        test_loader = DataLoader(test_data, batch_size=len(test_data))

        train_losses, validate_losses, train_accuracies, validation_accuracies = \
            train_and_measure(ae, train_dataloader, validate_loader, criterion, best_params, verbose=True)

        test_set = next(iter(test_loader)).to(DEVICE)

        with T.no_grad():
            output = ae.forward(test_set)
        test_loss = criterion(output, test_set).item()

        # Last 10% Of the Graphs Results
        train_losses_data.append(np.average(train_losses[:best_params.epochs * .1]))
        validate_losses_data.append(np.average(validate_losses[:best_params.epochs * .1]))
        train_accuracies_data.append(np.average(train_accuracies[:best_params.epochs * .1]))
        validation_accuracies_data.append(np.average(validation_accuracies[:best_params.epochs * .1]))
        test_loss_data.append(test_loss)

        seeds.append(seed)

    # Take Best Graphs Results for Each Measurement
    train_losses_data_best_index = np.argmin(train_losses_data)
    validate_losses_data_best_index = np.argmin(validate_losses_data)
    test_loss_data_best_index = np.argmin(test_loss_data)
    train_accuracies_data_best_index = np.argmax(train_accuracies_data)
    validation_accuracies_data_best_index = np.argmax(validation_accuracies_data)

    best_indices = [train_losses_data_best_index,
                    validate_losses_data_best_index,
                    test_loss_data_best_index,
                    train_accuracies_data_best_index,
                    validation_accuracies_data_best_index]

    frequencies = collections.Counter(best_indices)
    keys = np.fromiter(frequencies.keys(), dtype=float)
    values = np.fromiter(frequencies.values(), dtype=float)
    argmax = np.argmax(values)
    best_index = keys[argmax]

    # Return The "Best Seed"
    best_seed = seeds[best_index]

    return best_seed


def make_nans_average(batch):
    sum = 0
    nan_indices = []
    index = 0

    for x in batch:
        i = 0
        for xi in x:
            j = 0
            for xij in xi:
                if math.isnan(float(xij)):
                    batch[index][i][j] = 0
                    nan_indices.append((index, i, j))
                else:
                    sum += xij
                j += 1
            i += 1
        index += 1

    average = 0
    len_nan = len(nan_indices)

    if len(nan_indices) > 0:
        average = sum / len_nan

    for (index, i, j) in nan_indices:
        batch[index][i][j] = average

    return batch


def fit(ae, train_dataloader, criterion, hyperparameters: LstmAEHyperparameters, epoch_end_callbacks=(),
        supervised=False, make_nans_average_check=False):
    optimizer = optim.Adam(ae.parameters(), lr=hyperparameters.lr)

    for epoch in range(hyperparameters.epochs):
        optimizer.zero_grad()

        epoch_losses = []
        for batch in iter(train_dataloader):
            if make_nans_average_check:
                batch = make_nans_average(batch)

            loss = batch_loss(ae, batch, criterion, supervised=supervised)
            loss.backward()

            if hyperparameters.grad_clipping is not None:
                nn.utils.clip_grad_value_(ae.parameters(), clip_value=hyperparameters.grad_clipping)

            optimizer.step()

            epoch_losses.append(loss.item())

        epoch_loss = sum(epoch_losses) / len(epoch_losses)

        for callback in epoch_end_callbacks:
            callback(epoch, ae, epoch_loss)


def batch_loss(ae, batch, criterion, supervised=False):
    if supervised:
        X, y = batch[0].to(DEVICE), batch[1].to(DEVICE)

        output = ae.forward(X)
        return criterion(output, X, y)
    else:
        X = batch.to(DEVICE)

        output = ae.forward(X)
        return criterion(output, batch)


def train_and_measure(ae, train_dataloader, validate_dataloader, criterion, hyperparameters, supervised=False,
                      verbose=False, make_nans_average_check=False):
    train_losses = []
    validate_losses = []

    store_train_loss = lambda epoch, ae, loss: train_losses.append(loss)

    def verbose_print(epoch, ae, loss):
        print(f"Epoch: {epoch}, Loss: {loss}")

    def store_validation_loss(epoch, ae, train_loss):
        validation_set = next(iter(validate_dataloader))

        with T.no_grad():
            loss = batch_loss(ae, validation_set, criterion, supervised=supervised).item()

        validate_losses.append(loss)

    callbacks = [store_train_loss, store_validation_loss]

    if verbose:
        callbacks.append(verbose_print)

    validation_accuracies = []
    train_accuracies = []

    if supervised:
        def measure_accuracy(epoch, ae, train_loss, verbose=False):
            train_set = next(iter(train_dataloader))
            X_train, y_train = train_set[0].to(DEVICE), train_set[1].to(DEVICE)

            validation_set = next(iter(validate_dataloader))
            X_val, y_val = validation_set[0].to(DEVICE), validation_set[1].to(DEVICE)

            with T.no_grad():
                output_val = ae.forward(X_val)
                predictions_val = T.argmax(output_val.label_predictions, -1)

                correct_val = predictions_val.eq(y_val).sum().item()
                accuracy_val = correct_val / predictions_val.shape[-1]

                output_train = ae.forward(X_train)
                predictions_train = T.argmax(output_train.label_predictions, -1)

                correct_train = predictions_train.eq(y_train).sum().item()
                accuracy_train = correct_train / predictions_train.shape[-1]

                if verbose:
                    print(f"Epoch: {epoch}, Train Accuracy: {accuracy_train}, Validation Accuracy: {accuracy_val}")

            validation_accuracies.append(accuracy_val)
            train_accuracies.append(accuracy_train)

        callbacks.append(measure_accuracy)
    fit(ae,
        train_dataloader,
        criterion,
        hyperparameters,
        epoch_end_callbacks=callbacks,
        supervised=supervised,
        make_nans_average_check=make_nans_average_check)

    if not supervised:
        return train_losses, validate_losses

    return train_losses, validate_losses, train_accuracies, validation_accuracies


def evaluate_hyperparameters(train_data, validate_data, criterion, hyperparameters: LstmAEHyperparameters,
                             supervised=False):
    train_dataloader = DataLoader(train_data, batch_size=hyperparameters.batch_size, shuffle=True)
    ae = hyperparameters.create_ae()

    fit(ae, train_dataloader, criterion, hyperparameters, supervised=supervised)

    validate_loader = DataLoader(validate_data, batch_size=len(validate_data))
    validation_set = next(iter(validate_loader)).to(DEVICE)

    with T.no_grad():
        output = ae.forward(validation_set)
        loss = criterion(output, validation_set).item()

    return loss


def draw_sample(ae, data, n_samples=1, title="example"):
    with T.no_grad():
        for _ in range(n_samples):
            idx = T.randint(len(data), (1,))
            sample = data[idx].to(DEVICE).unsqueeze(0)

            output = ae.forward(sample).output_sequence

            df = pd.DataFrame.from_dict({'actual': sample.squeeze().tolist(),
                                         'predicted': output.squeeze().tolist()})
            df.index.name = "t"

            sns.lineplot(data=df, dashes=False)
            plt.title(title)
            plt.ylabel("y")
            plt.show()


def tune_parameters(train_data, valid_data, criterion, should_tune=False):
    if should_tune:
        param_choices = {
            'epochs': [700],
            'seq_dim': [1],
            'batch_size': [128],
            'num_layers': [2],
            'latent_size': [256],
            'lr': [0.0001, 0.001],
            'grad_clipping': [None, 1],
        }

        def tune_objective(**params):
            hyperparameters = LstmAEHyperparameters(**params)
            return evaluate_hyperparameters(train_data, valid_data, criterion, hyperparameters)

        best_params, best_loss = tune(tune_objective, param_choices, "minimize", workers=4)
        best_params = LstmAEHyperparameters(**best_params)

        print("Best parameters are:")
        print(f"\tlatent size: {best_params.latent_size}")
        print(f"\tlr: {best_params.lr}")
        print(f"\tgrad_clipping: {best_params.grad_clipping}")
        print(f"\tnum_layers: {best_params.num_layers}")

    else:
        best_params = LstmAEHyperparameters(
            epochs=500,
            seq_dim=1,
            batch_size=128,

            num_layers=5,
            lr=0.001,
            latent_size=256,
            grad_clipping=None
        )

    return best_params


def print_results_and_plot_graph(ae, criterion, test_data, test_loader, best_params, train_losses, validate_losses,
                                 train_accuracy, validate_accuracy,
                                 title, is_supervised=False):
    draw_sample(ae, test_data, n_samples=2, title=title)

    df = pd.DataFrame.from_dict({"training set": train_losses,
                                 "validation set": validate_losses})
    df.index.name = "Epoch"
    sns.lineplot(data=df, dashes=False)
    lr_str = "{:12.7f}".format(best_params.lr)
    plt.title(title)
    plt.ylabel("Loss")
    plt.show()

    if is_supervised:
        df = pd.DataFrame.from_dict({"training set": train_accuracy,
                                     "validation set": validate_accuracy})
        df.index.name = "Epoch"

        sns.lineplot(data=df, dashes=False)
        lr_str = "{:12.7f}".format(best_params.lr)
        plt.title(title)
        plt.ylabel("Accuracy")
        plt.show()

    test_set = next(iter(test_loader)).to(DEVICE)

    with T.no_grad():
        output = ae.forward(test_set)
        test_loss = criterion(output, test_set).item()

    print(f"Test loss: {test_loss}")


def fetch_and_split_data(file):
    dataset = (pd.read_csv(file, index_col=False))
    X = dataset['high']
    Y = dataset['symbol']

    df_train = {}
    df_test = {}
    df_validate = {}

    n_data = np.shape(X)[0]

    n_train = n_data * 0.6
    n_validate = n_data * 0.2
    n_test = n_data - n_train - n_validate

    for i in range(n_data):
        if not math.isnan(X[i]):
            if i < n_train:
                if df_train.get(Y[i]) is None:
                    df_train[Y[i]] = [(X[i])]
                else:
                    df_train[Y[i]].append(X[i])
            elif i < n_validate + n_train:
                if df_validate.get(Y[i]) is None:
                    df_validate[Y[i]] = [(X[i])]
                else:
                    df_validate[Y[i]].append(X[i])
            else:
                if df_test.get(Y[i]) is None:
                    df_test[Y[i]] = [(X[i])]
                else:
                    df_test[Y[i]].append(X[i])

    for key in df_train:
        df_train[key] = pd.Series(df_train[key])

    for key in df_validate:
        df_validate[key] = pd.Series(df_validate[key])

    for key in df_test:
        df_test[key] = pd.Series(df_test[key])

    dataframe_train = pd.DataFrame(df_train)
    train_data = SyntheticDataset(filename=None, df=dataframe_train)

    dataframe_validate = pd.DataFrame(df_validate)
    valid_data = SyntheticDataset(filename=None, df=dataframe_validate)

    dataframe_test = pd.DataFrame(df_test)
    test_data = SyntheticDataset(filename=None, df=dataframe_test)

    return train_data, valid_data, test_data