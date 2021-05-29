import torch as T
import torch.nn as nn
from torchvision import datasets
import pandas as pd
import seaborn as sns
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


from torch.utils.data import DataLoader, SubsetRandomSampler
from dataclasses import dataclass

import utils
from ae_wrappers.ae_classification_wrapper import AutoEncoderClassifier, AutoencoderClassifierOutput
from experiment import Experiment

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

    train_data = T.utils.data.Subset(train_data, list(range(0, 1000)))
    validate_data = T.utils.data.Subset(validate_data, list(range(0, 200)))
    test_data = T.utils.data.Subset(test_data, list(range(0, 200)))

    hyperparameters = AEClassifierHyperparameters(
        epochs=5,
        seq_dim=28,
        batch_size=1024,
        n_classes=10,

        num_layers=2,
        lr=0.001,
        latent_size=64,
        grad_clipping=None
    )

    ae = hyperparameters.create_ae()

    train_dataloader = DataLoader(train_data, batch_size=hyperparameters.batch_size, shuffle=True)
    validate_dataloader = DataLoader(validate_data, batch_size=hyperparameters.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=hyperparameters.batch_size, shuffle=True)

    mse = nn.MSELoss()
    cel = nn.CrossEntropyLoss()

    criterion = {
        "reconstruction_loss": lambda output, input, labels: mse(output.output_sequence, input),
        "classification_loss": lambda output, input, labels: cel(output.label_predictions, labels)
    }

    experiment = Experiment(criterion, {'accuracy': Experiment.measure_accuracy}, supervised=True)
    results_df = experiment.run(ae, train_dataloader, test_dataloader, hyperparameters, verbose=True, measure_every=10)

    results_df['train_loss'] = results_df['train_reconstruction_loss'] + results_df['train_classification_loss']
    results_df['test_loss'] = results_df['test_reconstruction_loss'] + results_df['test_classification_loss']

    utils.plot_metric(results_df, "loss", title="Combined Loss")
    utils.plot_metric(results_df, "reconstruction_loss", title="Reconstruction Loss")
    utils.plot_metric(results_df, "classification_loss", title="Classification Loss")
    utils.plot_metric(results_df, "accuracy", title="Accuracy")

    test_images = [tensor for tensor, label in test_data]

    utils.draw_reconstruction_sample(ae, test_images, n_samples=2, type="image", title="")
    utils.plot_classification_sample(ae, test_data, n_samples=9, type="image", title="Classification Sample")

    print(f"Test loss: {results_df.iloc[-1]['test_loss']}")
