import torch as T
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from torch import nn, optim
from torch.utils.data import DataLoader, RandomSampler

from data.synthetic_data import SyntheticDataset
from lstm_ae import LstmAutoEncoder
from grid_search import tune

sns.set_theme(style="darkgrid")

file = "../data/cache/synthetic.csv"
DEVICE = T.device('cuda' if T.cuda.is_available() else 'cpu')
T.set_default_dtype(T.double)


def fit(ae, train_dataloader, lr, grad_clipping, epochs, epoch_end_callbacks=()):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ae.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()

        epoch_losses = []
        for batch in iter(train_dataloader):
            batch = batch.to(DEVICE)
            output = ae.forward(batch)

            loss = criterion(output, batch)
            loss.backward()

            nn.utils.clip_grad_value_(ae.parameters(), clip_value=grad_clipping)
            optimizer.step()

            epoch_losses.append(loss.item())

        epoch_loss = sum(epoch_losses) / len(epoch_losses)

        for callback in epoch_end_callbacks:
            callback(epoch, ae, epoch_loss)


def test_hyperparameters(train_data, validate_data, lr, grad_clipping, seq_dim, latent_size, num_layers, batch_size, epochs):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    ae = LstmAutoEncoder(seq_dim, latent_size, num_layers)

    fit(ae, train_dataloader, lr, grad_clipping, epochs)

    validate_loader = DataLoader(validate_data, batch_size=len(validate_data))
    validation_set = next(iter(validate_loader)).to(DEVICE)

    with T.no_grad():
        output = ae.forward(validation_set)
        loss = nn.MSELoss()(output, validation_set).item()

    return loss


def draw_sample(ae, data, n_samples=1):
    with T.no_grad():
        for _ in range(n_samples):
            idx = T.randint(len(data), (1,))
            sample = data[idx].to(DEVICE).unsqueeze(0)

            output = ae.forward(sample)

            df = pd.DataFrame.from_dict({'actual': sample.squeeze().tolist(),
                                         'predicted': output.squeeze().tolist()})
            df.index.name = "t"

            sns.lineplot(data=df, dashes=False)
            plt.title("example")
            plt.ylabel("y")
            plt.show()


if __name__ == "__main__":
    dataset = SyntheticDataset(file)

    train_n = int(0.6 * len(dataset))
    validate_n = int(0.2 * len(dataset))
    test_n = int(len(dataset) - train_n - validate_n)

    train_data, valid_data, test_data = T.utils.data.random_split(
        dataset, (train_n, validate_n, test_n))


    should_tune = True # change to false to use predefined hyperparameters
    if should_tune:
        param_choices = {
            'epochs': [50],
            'seq_dim': [1],
            'train_data': [train_data],
            'validate_data': [valid_data],
            'batch_size': [512],
            'num_layers': [1, 2],

            'latent_size': [1, 5, 20],
            'lr': [0.001, 0.0001, 0.00001],
            'grad_clipping': [1, 0.1, 0.001],
        }

        best_params, best_loss = tune(test_hyperparameters, param_choices, "minimize", workers=4)

        print("Best parameters are:")
        print(f"\tlatent size: {best_params['latent_size']}")
        print(f"\tlr: {best_params['lr']}")
        print(f"\tgrad_clipping: {best_params['grad_clipping']}")
    else:
        best_params = {
            'epochs': 10,
            'seq_dim': 1,
            'train_data': train_data,
            'validate_data': valid_data,
            'batch_size': 1,
            'num_layers': 5,

            'latent_size': 50,
            'lr': 0.0001,
            'grad_clipping': 0.1,
        }

    train_epochs = 100

    train_losses = []
    validate_losses = []

    ae = LstmAutoEncoder(best_params["seq_dim"], best_params["latent_size"], best_params["num_layers"])
    train_dataloader = DataLoader(train_data, batch_size=best_params["batch_size"], shuffle=True)
    validate_loader = DataLoader(valid_data, batch_size=len(valid_data))
    test_loader = DataLoader(test_data, batch_size=len(test_data))

    store_train_loss = lambda epoch, ae, loss: train_losses.append(loss)

    def store_validation_loss(epoch, ae, train_loss):
        validation_set = next(iter(validate_loader)).to(DEVICE)

        with T.no_grad():
            output = ae.forward(validation_set)
            loss = nn.MSELoss()(output, validation_set).item()

        validate_losses.append(loss)

    fit(ae,
        train_dataloader,
        best_params["lr"],
        best_params["grad_clipping"],
        train_epochs,
        epoch_end_callbacks=[store_train_loss, store_validation_loss])

    draw_sample(ae, test_data, n_samples=2)

    df = pd.DataFrame.from_dict({"training set": train_losses,
                                 "validation set": validate_losses})
    df.index.name = "Epoch"

    sns.lineplot(data=df, dashes=False)
    lr_str = "{:12.7f}".format(best_params['lr'])
    #plt.title(f"{kwargs['title']}\n learning rate = {lr_str}")
    plt.title("Learn Loss")
    plt.ylabel("Loss")
    plt.show()

    test_set = next(iter(test_loader)).to(DEVICE)

    with T.no_grad():
        output = ae.forward(test_set)
        test_loss = nn.MSELoss()(output, test_set).item()

    print(f"Test loss: {test_loss}")

