import torch as T
import torch.nn as nn
import numpy as np
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn_image as isns

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
    criterion = criterion if isinstance(criterion, dict) else {"loss":criterion}
    optimizer = optim.Adam(ae.parameters(), lr=hyperparameters.lr)

    for epoch in range(hyperparameters.epochs):
        epoch_losses, batch_sizes = [], []

        for batch in iter(train_dataloader):
            optimizer.zero_grad()

            losses = batch_losses(ae, batch, criterion, supervised=supervised)
            loss = sum(losses.values())
            loss.backward()

            if hyperparameters.grad_clipping is not None:
                nn.utils.clip_grad_value_(ae.parameters(), clip_value=hyperparameters.grad_clipping)

            optimizer.step()

            epoch_losses.append(losses)
            batch_sizes.append(len(batch))

        epoch_loss = {name: [loss_dict[name].item() for loss_dict in epoch_losses] for name in criterion.keys()}
        epoch_loss = {name: np.average(loss_list, weights=batch_sizes) for name, loss_list in epoch_loss.items()}

        if verbose:
            print(f"Epoch: {epoch} loss: {epoch_loss}")

        for callback in epoch_end_callbacks:
            callback(epoch, ae, epoch_loss)


def epoch_loss(ae, dataloader, criterion, supervised=False):
    losses, batch_sizes = [], []

    for batch in iter(dataloader):
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


def batch_losses(ae, batch, criterion_dict, supervised=False):
    return {name: batch_loss(ae, batch, criterion, supervised=supervised)
            for name, criterion in criterion_dict.items()}


def epoch_losses(ae, dataloader, criterion_dict, supervised=False):
    loss_dicts, batch_sizes = [], []

    for batch in iter(dataloader):
        loss_dicts.append(batch_losses(ae, batch, criterion_dict, supervised=supervised))
        batch_sizes.append(len(batch))

    losses = {name: [loss_dict[name].item() for loss_dict in loss_dicts] for name in criterion_dict.keys()}
    losses = {name: np.average(loss_list, weights=batch_sizes) for name, loss_list in losses.items()}

    return losses


def evaluate_hyperparameters(train_data, validate_data, criterion, hyperparameters:LstmAEHyperparameters,
                             supervised=False, verbose=False):

    if verbose:
        print(f"Evaluate hyper-parameters: {hyperparameters}")

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
                                             'reconstructed': output.squeeze().tolist()})
                df.index.name = "Timestep"

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


def plot_classification_sample(ae, data, n_samples=1, title="example", type="line"):
    samples = T.utils.data.Subset(data, list(range(0, n_samples)))
    loader = DataLoader(samples, batch_size=n_samples)
    samples = next(iter(loader))
    X, y = samples[0], samples[1]

    with T.no_grad():
        y_pred = ae.forward(X.to(DEVICE)).label_predictions
        y_pred = T.argmax(y_pred, dim=-1).cpu()
        y_pred = [tensor.item() for tensor in list(y_pred)]

    if type == "image":
        images = list(X)
        labels = [f"Predicted: {str(pred)} Actual: {gt}" for pred, gt in zip(y_pred, y)]

        grid = isns.ImageGrid(images, orientation="h", cbar_label=labels)

    elif type == "line":
        raise NotImplemented(f"No support for drawing lines yet")
    else:
        raise Exception(f'type can be either "line" or "image", but was {type}.')

    plt.title(title)
    plt.show()


def plot_prediction_sample(ae, data, n_samples=1, title="example"):
    samples = T.utils.data.Subset(data, list(range(0, n_samples)))
    loader = DataLoader(samples, batch_size=n_samples)
    X = next(iter(loader))

    with T.no_grad():
        prediction = ae.forward(X.to(DEVICE)).predicted_value

    for actual, pred in zip(X, prediction):
        actual, pred = actual.squeeze(-1), pred.squeeze(-1)
        df = pd.DataFrame({'actual': actual[1:].cpu(),
                           'predicted': pred[:-1].cpu()})

        df.index.name = "Timestep"
        graph = sns.lineplot(data=df, dashes=False)
        plt.ylabel("y")
        plt.title(title)
        plt.show()


def plot_metric(df: pd.DataFrame, metric_name, title=None):
    sns.lineplot(data=df[[f"train_{metric_name}", f"test_{metric_name}"]],
                 dashes=False)
    if title:
        plt.title(title)

    plt.show()

