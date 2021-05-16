import argparse
import torch as T

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

default_name = "cache/synthetic.csv"


class SyntheticDataset(Dataset):
    def __init__(self, filename=default_name):
        self.data = (pd.read_csv(filename, index_col=False))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return T.tensor(self.data.loc[index].values).unsqueeze(-1)


def plot_signals(df):
    fig, ax = plt.subplots()

    # put the labels at 45deg since they tend to be too long
    fig.autofmt_xdate()

    locations = [0, 1, 2]
    for loc in locations:
        signals = df.iloc[loc]

        sns.lineplot(x=range(len(signals)), y=signals)
        plt.title(f"Signals data {loc}")
        plt.ylabel("Value")
        plt.xlabel("Time Step")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", "-t", dest='timesteps', type=int, default=50)
    parser.add_argument("--samples", "-n", dest='samples', type=int, default=10 * 1000)
    parser.add_argument("--destination" "-d", dest='destination', default=default_name)

    args = parser.parse_args()

    timesteps, samples = args.timesteps, args.samples

    data = np.random.random((samples, timesteps))
    df = pd.DataFrame(data)
    df.columns = list([f"y_{t}" for t in range(timesteps)])
    # df.index = list([f"x_{i}" for i in range(samples)])

    df.to_csv(args.destination, index=False)

    signals_plot = False

    if signals_plot:
        plot_signals(df)