import torch as T
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


def train_validate_test_split(dataset, train_ratio=0.6, validate_ratio=0.2, test_ratio=0.2):
    train_n = int(train_ratio * len(dataset))
    validate_n = int(validate_ratio * len(dataset))
    test_n = int(test_ratio * len(dataset))

    train_data, valid_data, test_data = T.utils.data.random_split(
        dataset, (train_n, validate_n, test_n))

    return train_data, valid_data, test_data


def fit(ae, train_dataloader, criterion, hyperparameters:LstmAEHyperparameters, epoch_end_callbacks=()):
    optimizer = optim.Adam(ae.parameters(), lr=hyperparameters.lr)

    for epoch in range(hyperparameters.epochs):
        optimizer.zero_grad()

        epoch_losses = []
        for batch in iter(train_dataloader):
            batch = batch.to(DEVICE)
            output = ae.forward(batch)

            loss = criterion(output, batch)
            loss.backward()

            if hyperparameters.grad_clipping is not None:
                nn.utils.clip_grad_value_(ae.parameters(), clip_value=hyperparameters.grad_clipping)

            optimizer.step()

            epoch_losses.append(loss.item())

        epoch_loss = sum(epoch_losses) / len(epoch_losses)

        for callback in epoch_end_callbacks:
            callback(epoch, ae, epoch_loss)


def train_and_measure(ae, train_dataloader, validate_dataloader, criterion, hyperparameters):
    train_losses = []
    validate_losses = []

    store_train_loss = lambda epoch, ae, loss: train_losses.append(loss)

    def store_validation_loss(epoch, ae, train_loss):
        validation_set = next(iter(validate_dataloader)).to(DEVICE)

        with T.no_grad():
            output = ae.forward(validation_set)
            loss = criterion(output, validation_set).item()

        validate_losses.append(loss)

    fit(ae,
        train_dataloader,
        criterion,
        hyperparameters,
        epoch_end_callbacks=[store_train_loss, store_validation_loss])

    return train_losses, validate_losses


def evaluate_hyperparameters(train_data, validate_data, criterion, hyperparameters:LstmAEHyperparameters):
    train_dataloader = DataLoader(train_data, batch_size=hyperparameters.batch_size, shuffle=True)
    ae = hyperparameters.create_ae()

    fit(ae, train_dataloader, criterion, hyperparameters)

    validate_loader = DataLoader(validate_data, batch_size=len(validate_data))
    validation_set = next(iter(validate_loader)).to(DEVICE)

    with T.no_grad():
        output = ae.forward(validation_set)
        loss = criterion(output, validation_set).item()

    return loss


def draw_sample(ae, data, n_samples=1):
    with T.no_grad():
        for _ in range(n_samples):
            idx = T.randint(len(data), (1,))
            sample = data[idx].to(DEVICE).unsqueeze(0)

            output = ae.forward(sample).output_sequence

            df = pd.DataFrame.from_dict({'actual': sample.squeeze().tolist(),
                                         'predicted': output.squeeze().tolist()})
            df.index.name = "t"

            sns.lineplot(data=df, dashes=False)
            plt.title("example")
            plt.ylabel("y")
            plt.show()