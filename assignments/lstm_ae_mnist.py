import torch as T
import torch.nn as nn
from torchvision import datasets
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


from torch.utils.data import DataLoader, SubsetRandomSampler
from dataclasses import dataclass

import utils
from ae_wrappers.ae_classification_wrapper import AutoEncoderClassifier, AutoencoderClassifierOutput

from grid_search import tune
from utils import LstmAEHyperparameters

sns.set_theme(style="darkgrid")

DEVICE = T.device('cuda' if T.cuda.is_available() else 'cpu')
T.set_default_dtype(T.double)


@dataclass(frozen=True)
class AEClassifierHyperparameters(LstmAEHyperparameters):
    n_classes: int

    def create_ae(self):
        ae = super().create_ae()
        return AutoEncoderClassifier(ae, self.n_classes).to(DEVICE)


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        #  transforms.Lambda(lambda X: T.ravel(X).unsqueeze(-1))
        transforms.Lambda(lambda X: X.squeeze())
    ])

    train_data, validate_data, test_data = \
        utils.load_torch_dataset(datasets.MNIST, transform=transform, cache_path="../data/cache")

    train_data = T.utils.data.Subset(train_data, list(range(0, 100)))
    validate_data = T.utils.data.Subset(validate_data, list(range(0, 20)))
    test_data = T.utils.data.Subset(test_data, list(range(0, 20)))

    hyperparameters = AEClassifierHyperparameters(
        epochs=500,
        seq_dim=28,
        batch_size=64,
        n_classes=10,

        num_layers=5,
        lr=0.001,
        latent_size=256,
        grad_clipping=None
    )

    ae = hyperparameters.create_ae()

    train_dataloader = DataLoader(train_data, batch_size=hyperparameters.batch_size, shuffle=True)
    validate_dataloader = DataLoader(train_data, batch_size=len(validate_data), shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=True)

    mse = nn.MSELoss()
    cel = nn.CrossEntropyLoss()

    def criterion(output: AutoencoderClassifierOutput, input_sequence, labels):
        reconstruction_loss = mse(output.output_sequence, input_sequence)
        classification_loss = cel(output.label_predictions, labels)

        return reconstruction_loss + classification_loss

    train_losses, validate_losses, accuracies = \
        utils.train_and_measure(ae, train_dataloader, validate_dataloader, criterion, hyperparameters, supervised=True)

    test_images = [tensor.reshape((28, 28)) for tensor, label in test_data]
    # test_sequences_dataloader = DataLoader(test_sequences, batch_size=len(test_data), shuffle=True)

    utils.draw_sample(ae, test_images, n_samples=2, type="image")

