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

    # train_data = T.utils.data.Subset(train_data, list(range(0, 1000)))
    # validate_data = T.utils.data.Subset(validate_data, list(range(0, 200)))
    # test_data = T.utils.data.Subset(test_data, list(range(0, 200)))

    hyperparameters = AEClassifierHyperparameters(
        epochs=250,
        seq_dim=28,
        batch_size=256,
        n_classes=10,

        num_layers=2,
        lr=0.001,
        latent_size=256,
        grad_clipping=None
    )

    ae = hyperparameters.create_ae()
    model_name = "lstm_ae_mnist"

    supervised = True
    load_model = False
    if load_model:
        ae.load_state_dict(T.load(f"../data/model/{model_name}"))

    train_dataloader = DataLoader(train_data, batch_size=hyperparameters.batch_size, shuffle=True)
    validate_dataloader = DataLoader(validate_data, batch_size=hyperparameters.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=hyperparameters.batch_size, shuffle=True)

    mse = nn.MSELoss()
    cel = nn.CrossEntropyLoss()

    def criterion(output: AutoencoderClassifierOutput, input_sequence, labels, supervised=False):
        reconstruction_loss = mse(output.output_sequence, input_sequence)
        classification_loss = cel(output.label_predictions, labels)

        if supervised:
            return classification_loss
        else:
            return reconstruction_loss

    train_losses, test_losses, train_accuracy, test_accuracy = \
        utils.train_and_measure(ae, train_dataloader, test_dataloader, criterion, hyperparameters,
                                supervised=supervised,
                                verbose=True,
                                save_interval=50,
                                model_name=model_name
                                )

    utils.plot_metric(train_losses, test_losses, "Loss")
    utils.plot_metric(train_accuracy, test_accuracy, "Accuracy")

    test_images = [tensor for tensor, label in test_data]

    utils.draw_reconstruction_sample(ae, test_images, n_samples=2, type="image")
    utils.draw_classification_sample(ae, test_data, n_samples=9, type="image")

    print(f"Test loss: {test_losses[-1]}")
