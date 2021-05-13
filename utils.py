import torch as T
import torch.nn as nn
import numpy as np
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn_image as isns

from pytorch_metric_learning import losses, reducers
from dataclasses import dataclass
from torch.utils.data import DataLoader
from typing import Union

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


def load_torch_dataset(dataset, transform=None, train_validate_split=(2/3, 1/3), cache_path='/data/cache'):

    if transform:
        train_data = dataset(
            root=cache_path,
            train=True,
            download=True,
            transform=transform
        )

        test_data = dataset(
            root=cache_path,
            train=False,
            download=True,
            transform=transform
        )
    else:
        train_data = dataset(
            root=cache_path,
            train=True,
            download=True,
        )
        test_data = dataset(
            root=cache_path,
            train=False,
            download=True,
        )
    train_len = int(len(train_data) * train_validate_split[0])
    validate_len = int(len(train_data) * train_validate_split[1])

    train_data, validate_data = T.utils.data.random_split(train_data, [train_len, validate_len])

    return train_data, validate_data, test_data


def train_validate_test_split(dataset, train_ratio=0.6, validate_ratio=0.2, test_ratio=0.2):
    train_n = int(train_ratio * len(dataset))
    validate_n = int(validate_ratio * len(dataset))
    test_n = int(test_ratio * len(dataset))

    train_data, valid_data, test_data = T.utils.data.random_split(
        dataset, (train_n, validate_n, test_n))

    return train_data, valid_data, test_data


def fit(ae, train_dataloader, criterion, hyperparameters:LstmAEHyperparameters, epoch_end_callbacks=(),
        supervised=False, verbose=False):
    optimizer = optim.Adam(ae.parameters(), lr=hyperparameters.lr)

    for epoch in range(hyperparameters.epochs):
        optimizer.zero_grad()

        epoch_losses, batch_sizes = [], []
        for batch in iter(train_dataloader):
            loss = batch_loss(ae, batch, criterion, supervised=supervised)
            loss.backward()

            if hyperparameters.grad_clipping is not None:
                nn.utils.clip_grad_value_(ae.parameters(), clip_value=hyperparameters.grad_clipping)

            optimizer.step()

            epoch_losses.append(loss.item())
            batch_sizes.append(len(batch))

        epoch_loss = np.average(epoch_losses, weights=batch_sizes)

        if verbose:
            print(f"Epoch: {epoch} loss: {epoch_loss}")

        for callback in epoch_end_callbacks:
            callback(epoch, ae, epoch_loss)


def epoch_loss(ae, dataloder, criterion, supervised=False):
    losses, batch_sizes = [], []

    for batch in iter(dataloder):
        losses.append(batch_loss(ae, batch, criterion, supervised=supervised).item())
        batch_sizes.append(len(batch))

    return np.average(losses, weights=batch_sizes)


def batch_loss(ae, batch, criterion, supervised=False):
    if supervised:
        X, y = batch[0].to(DEVICE), batch[1].to(DEVICE)

        output = ae.forward(X)
        return criterion(output, X, y)
    else:
        X = batch.to(DEVICE)

        output = ae.forward(X)
        return criterion(output, X)


def train_and_measure(ae, train_dataloader, test_dataloader, criterion, hyperparameters, supervised=False,
                      verbose=False):
    train_losses = []
    test_losses = []

    store_train_loss = lambda epoch, ae, loss: train_losses.append(loss)

    def store_test_loss(epoch, ae, train_loss):
        with T.no_grad():
            loss = epoch_loss(ae, test_dataloader, criterion, supervised=supervised)

        test_losses.append(loss)

    callbacks = [store_train_loss, store_test_loss]

    train_accuracies = []
    test_accuracies = []
    if supervised:
        def measure_accuracy(data_loader):
            n_correct = 0
            total = 0
            with T.no_grad():
                for batch in iter(data_loader):

                    X, y = batch[0].to(DEVICE), batch[1].to(DEVICE)

                    output = ae.forward(X)
                    predictions = T.argmax(output.label_predictions, -1)

                    n_correct += predictions.eq(y).sum().item()
                    total += len(batch[0])

            return n_correct / total

        callbacks.append(lambda epoch, ae, train_loss:
                         train_accuracies.append(measure_accuracy(train_dataloader)))
        callbacks.append(lambda epoch, ae, train_loss:
                         test_accuracies.append(measure_accuracy(test_dataloader)))
    fit(ae,
        train_dataloader,
        criterion,
        hyperparameters,
        epoch_end_callbacks=callbacks,
        supervised=supervised,
        verbose=verbose)

    if not supervised:
        return train_losses, test_losses

    return train_losses, test_losses, train_accuracies, test_accuracies


def evaluate_hyperparameters(train_data, validate_data, criterion, hyperparameters:LstmAEHyperparameters, supervised=False):
    train_dataloader = DataLoader(train_data, batch_size=hyperparameters.batch_size, shuffle=True)
    ae = hyperparameters.create_ae()

    fit(ae, train_dataloader, criterion, hyperparameters, supervised=supervised)

    validate_loader = DataLoader(validate_data, batch_size=len(validate_data))

    with T.no_grad():
        loss = epoch_loss(ae, validate_loader, criterion, supervised=supervised)

    return loss


def draw_reconstruction_sample(ae, data, n_samples=1, title="example", type="line"):
    with T.no_grad():
        for _ in range(n_samples):
            idx = T.randint(len(data), (1,))
            sample_cpu = data[idx]
            sample = sample_cpu.to(DEVICE).unsqueeze(0)

            output = ae.forward(sample).output_sequence

            if type == "line":
                df = pd.DataFrame.from_dict({'actual': sample.squeeze().tolist(),
                                             'predicted': output.squeeze().tolist()})
                df.index.name = "t"

                sns.lineplot(data=df, dashes=False)
                plt.ylabel("y")

            elif type == "image":
                images = [sample_cpu, output.squeeze(0).cpu()]
                labels = ["original", "reconstructed"]
                grid = isns.ImageGrid(images, orientation="h", cbar_label=labels)

            else:
                raise Exception(f'type can be either "line" or "image", but was {type}.')

            plt.title(title)
            plt.show()


def draw_classification_sample(ae, data, n_samples=1, title="example", type="line"):
    samples = T.utils.data.Subset(data, list(range(0, n_samples)))
    loader = DataLoader(samples, batch_size=n_samples)
    samples = next(iter(loader))
    X, y = samples[0], samples[1]

    with T.no_grad():
        y_pred = ae.forward(X.to(DEVICE)).label_predictions
        y_pred = T.argmax(y_pred, dim=-1).cpu()

    if type == "image":
        images = list(X)
        labels = [tensor.item() for tensor in list(y_pred)]
        labels = [str(label) for label in labels]

        grid = isns.ImageGrid(images, orientation="h", cbar_label=labels)

    elif type == "line":
        raise NotImplemented(f"No support for drawing lines yet")
    else:
        raise Exception(f'type can be either "line" or "image", but was {type}.')

    plt.title(title)
    plt.show()


def plot_metric(train_values, test_values, metric_name):
    df = pd.DataFrame.from_dict({"training set": train_values,
                                 "test set": test_values})
    df.index.name = "Epoch"

    sns.lineplot(data=df, dashes=False)
    plt.title(f"Learn {metric_name}")
    plt.ylabel(metric_name)
    plt.show()

