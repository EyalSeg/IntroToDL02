import argparse
import torch as T
import numpy as np
import pandas as pd

from sklearn import preprocessing
from torch.utils.data import Dataset

default_name = "cache/sp500.csv"
time_format = '%d/%m/%Y'


class SP500Dataset(Dataset):
    def __init__(self, filename=default_name, normalize=False):
        self.data = (pd.read_csv(filename, index_col=False, parse_dates=["date"]))

        if normalize:
            values = self.data['high'].values.astype(float).reshape(-1, 1)
            self.data['high'] = preprocessing.MinMaxScaler().fit_transform(values)

        self.data = self.data.pivot(index="symbol", columns="date", values="high")

        self.data = self.data.fillna(method="ffill", axis=1)
        self.data = self.data.fillna(method="bfill", axis=1)

    def get_dates(self):
        return self.data.columns

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if isinstance(index, int):
            index = self.data.index[index]

        return T.tensor(self.data.loc[index].values).unsqueeze(-1)


def plot_stock(dataset, stock_symbol, stock_name):
    fig, ax = plt.subplots()

    # assign locator and formatter for the xaxis ticks.
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter(time_format))

    # put the labels at 45deg since they tend to be too long
    fig.autofmt_xdate()

    data = dataset[stock_symbol].squeeze(-1)
    sns.lineplot(x=dataset.get_dates(), y=data)
    plt.title(f"{stock_name} Stock")
    plt.show()


if __name__ == "__main__":
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    sns.set_theme(style="darkgrid")

    dataset = SP500Dataset(normalize=True)
    dates = dataset.get_dates()

    plot_stock(dataset, "GOOGL", "Google")
    plot_stock(dataset, "AMZN", "Amazon")

