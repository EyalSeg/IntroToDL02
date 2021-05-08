import argparse
import torch as T

import pandas as pd
import numpy as np

from torch.utils.data import Dataset

default_name = "cache/synthetic.csv"


class SyntheticDataset(Dataset):
    def __init__(self, filename=default_name, df=None, supervised=False):
        if df is None:
            self.data = (pd.read_csv(filename, index_col=False))
        else:
            self.data = df

        self.supervised = supervised

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if not self.supervised:
            return T.tensor(self.data.loc[index].values).unsqueeze(-1)
        else:
            # TODO - Return the "true-value" out of self.data
            # If the Supervised Flag is False
            # We should return the X value out of Data
            # (That contains X and Y, such that, X is x[t], and Y is x[t+1]).
            value = T.tensor(self.data.loc[index].values[0]).unsqueeze(-1)
            return value


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
