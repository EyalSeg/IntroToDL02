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
        epochs=250,
        seq_dim=28,
        batch_size=1024,
        n_classes=10,

        num_layers=2,
        lr=0.001,
        latent_size=256,
        grad_clipping=0.5
    )

    ae = hyperparameters.create_ae()

    train_dataloader = DataLoader(train_data, batch_size=hyperparameters.batch_size, shuffle=True)
    validate_dataloader = DataLoader(validate_data, batch_size=hyperparameters.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=hyperparameters.batch_size, shuffle=True)

    mse = nn.MSELoss()
    cel = nn.CrossEntropyLoss()

    def criterion(output: AutoencoderClassifierOutput, input_sequence, labels):
        reconstruction_loss = mse(output.output_sequence, input_sequence)
        classification_loss = cel(output.label_predictions, labels)

        return reconstruction_loss + classification_loss

    experiment = Experiment(criterion, {'accuracy': Experiment.measure_accuracy}, supervised=True)
    results_df = experiment.run(ae, train_dataloader, test_dataloader, hyperparameters, verbose=True, measure_every=1)

    sns.lineplot(data=results_df[['train_loss', 'test_loss']],
                 dashes=False)
    plt.title("Loss")
    plt.show()

    sns.lineplot(data=results_df[['train_accuracy', 'test_accuracy']],
                 dashes=False)
    plt.title("Accuracy")
    plt.show()

    test_images = [tensor for tensor, label in test_data]

    utils.draw_reconstruction_sample(ae, test_images, n_samples=3, type="image")
    utils.draw_classification_sample(ae, test_data, n_samples=9, type="image")

    print(f"Test loss: {results_df.iloc[-1]['test_loss']}")
