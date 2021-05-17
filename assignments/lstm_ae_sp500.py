import torch as T
import torch.nn as nn
from torch.utils.data import DataLoader

import utils

from data.sp500_data import SP500Dataset

file = "../data/cache/sp500.csv"

if __name__ == "__main__":
    dataset = SP500Dataset(file, normalize=True)

    train_data, valid_data, test_data = utils.train_validate_test_split(dataset, 0.6, 0.2, 0.2)

    mse = nn.MSELoss()


    def criterion(output, input):
        return mse(output.output_sequence, input)


    hyperparameters = utils.LstmAEHyperparameters(
        epochs=250,
        seq_dim=1,
        batch_size=256,

        num_layers=1,
        lr=0.0005,
        latent_size=64,
        grad_clipping=1
    )

    ae = hyperparameters.create_ae()

    supervised = False
    regression = True
    load_model = False
    model_name = "lstm_ae_sp500"

    if supervised:
        model_path = f"{model_name}/supervised"
    else:
        model_path = f"{model_name}/un_supervised"

    if load_model:
        ae.load_state_dict(T.load(f"../data/model/{model_path}"))

    train_dataloader = DataLoader(train_data, batch_size=hyperparameters.batch_size, shuffle=True)
    validate_loader = DataLoader(valid_data, batch_size=hyperparameters.batch_size)
    test_loader = DataLoader(test_data, batch_size=hyperparameters.batch_size)

    train_losses, test_losses = \
        utils.train_and_measure(ae, train_dataloader, test_loader, criterion, hyperparameters,
                                verbose=True,
                                supervised=supervised,
                                save_interval=50,
                                model_path=model_path
                                )

    utils.draw_reconstruction_sample(ae, test_data, n_samples=2)
    utils.plot_metric(train_losses, test_losses, "Loss")

    print(f"Test loss: {test_losses[-1]}")
