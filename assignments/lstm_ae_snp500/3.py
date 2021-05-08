import torch as T
import torch.nn as nn
import seaborn as sns
from dataclasses import dataclass

from torch.utils.data import DataLoader

import utils
from ae_wrappers.ae_regression_wrapper import AutoEncoderRegression, AutoencoderRegressionOutput

sns.set_theme(style="darkgrid")

file = "../../data/cache/S&P500.csv"
DEVICE = T.device('cuda' if T.cuda.is_available() else 'cpu')
T.set_default_dtype(T.double)


@dataclass(frozen=True)
# AERegressionHyperparameters
# Same Structure as the AE-Classifier Class
class AERegressionHyperparameters(utils.LstmAEHyperparameters):
    output_dimension: int

    def create_ae(self):
        ae = super().create_ae()
        return AutoEncoderRegression(ae, self.output_dimension).to(DEVICE)


if __name__ == "__main__":
    # Fetch the S&P500 Data Set
    train_data, valid_data, test_data = utils.fetch_and_split_data(file, supervised=True, batch_size=64)

    # Change Those Parameters to The Best Parameters.
    hyperparameters = AERegressionHyperparameters(
        epochs=700,
        seq_dim=1,
        batch_size=64,
        output_dimension=1,

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


    def criterion(output: AutoencoderRegressionOutput, input_sequence_regression, labels):
        reconstruction_loss = mse(output.output_sequence, input_sequence_regression)
        classification_loss = cel(output.label_predictions, labels)

        # return reconstruction_loss + classification_loss
        return reconstruction_loss

    # TODO - Fix Data Building for the Regression Model
    # Or, fix the Model data using (adopt for Regression)
    train_losses, validate_losses, train_accuracies, validation_accuracies = \
        utils.train_and_measure(ae, train_dataloader, validate_dataloader, criterion, hyperparameters,
                                verbose=True, make_nans_average_check=True, supervised=True)

    # TODO - Adopt the New Print Methods recently added
    utils.print_results_and_plot_graph(ae, criterion, test_data, test_loader, hyperparameters, train_losses,
                                       validate_losses,
                                       train_accuracies, validation_accuracies,
                                       "lstm_ae_snp500", is_supervised=True)
