import torch as T
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from torch.utils.data import DataLoader, Subset

import utils

from data.sp500_data import SP500Dataset
from ae_wrappers.ae_regression_wrapper import *
from experiment import Experiment

file = "../data/cache/sp500.csv"
sns.set_theme(style="darkgrid")


def create_train_and_evaluate_model(train_dataloader, test_loader, test_data):
    ae = hyperparameters.create_ae()
    ae = AutoEncoderRegressor(ae)

    experiment = Experiment(criterion)
    results_df = experiment.run(ae, train_dataloader, test_loader, hyperparameters, measure_every=10, verbose=True)

    results_df['train_loss'] = results_df['train_reconstruction_loss'] + results_df['train_prediction_loss']
    results_df['test_loss'] = results_df['test_reconstruction_loss'] + results_df['test_prediction_loss']

    utils.plot_metric(results_df, "loss", title="Combined Loss")
    utils.plot_metric(results_df, "reconstruction_loss", title="Reconstruction Loss")
    utils.plot_metric(results_df, "prediction_loss", title="Prediction Loss")

    utils.draw_reconstruction_sample(ae, test_data, n_samples=2, title="Reconstruction Sample")
    utils.plot_prediction_sample(ae, test_data, n_samples=2, title="Prediction Sample")

    print(f"Test Reconstruction Loss: {results_df.iloc[-1]['test_reconstruction_loss']}")
    print(f"Test Prediction Loss: {results_df.iloc[-1]['test_prediction_loss']}")


if __name__ == "__main__":
    dataset = SP500Dataset(file, normalize=True)
    dataset = Subset(dataset, range(100))

    train_data, valid_data, test_data = utils.train_validate_test_split(dataset, 0.6, 0.2, 0.2)

    mse = nn.MSELoss()
    criterion = {
        "reconstruction_loss": lambda output, input: mse(output.output_sequence, input),
        "prediction_loss": lambda output, input: mse(output.predicted_value[:, :-1, :], input[:, 1:, :])
    }

    hyperparameters = utils.LstmAEHyperparameters(
        epochs=250,
        seq_dim=1,
        batch_size=128,

        num_layers=2,
        lr=0.001,
        latent_size=64,
        grad_clipping=0.5
    )

    k_fold_approach = True

    if k_fold_approach:
        k_folds = 5
        k_fold = KFold(n_splits=k_folds, shuffle=True)

        for fold, (train_ids, test_ids) in enumerate(k_fold.split(dataset)):
            train_sub_sampler = T.utils.data.SubsetRandomSampler(train_ids)
            test_sub_sampler = T.utils.data.SubsetRandomSampler(test_ids)
            train_loader = T.utils.data.DataLoader(dataset, batch_size=hyperparameters.batch_size,
                                                   sampler=train_sub_sampler)
            test_loader = T.utils.data.DataLoader(dataset, batch_size=hyperparameters.batch_size,
                                                  sampler=test_sub_sampler)
            print(f"K-Fold {fold} iteration")
            create_train_and_evaluate_model(train_dataloader=train_loader, test_loader=test_loader,
                                            test_data=test_data)
            print()

    else:
        train_dataloader = DataLoader(train_data, batch_size=hyperparameters.batch_size, shuffle=True)
        validate_loader = DataLoader(valid_data, batch_size=hyperparameters.batch_size)
        test_loader = DataLoader(test_data, batch_size=hyperparameters.batch_size)
        create_train_and_evaluate_model(train_dataloader=train_dataloader, test_loader=test_loader,
                                        test_data=test_data)


