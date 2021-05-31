import argparse
import torch as T

import pandas as pd
import numpy as np

from torch.utils.data import Dataset

default_name = "cache/synthetic.csv"

class SyntheticDataset(Dataset):
    def __init__(self, filename=default_name):
        self.data = (pd.read_csv(filename, index_col=False))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return T.tensor(self.data.loc[index].values).unsqueeze(-1)


if __name__ == "__main__":
    import seaborn as sns
    sns.set_theme(style="darkgrid")

    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", "-t", dest='timesteps', type=int, default=50)
    parser.add_argument("--samples", "-n", dest='samples', type=int, default=10 * 1000)
    parser.add_argument("--destination" "-d", dest='destination', default=default_name)
    parser.add_argument("--plot" "-p", dest='plot', type=int, default=0)

    args = parser.parse_args()

    if args.samples > 0:
        timesteps, samples = args.timesteps, args.samples

        data = np.random.random((samples, timesteps))

        df = pd.DataFrame(data)
        df.columns = list([f"y_{t}" for t in range(timesteps)])

        df.to_csv(args.destination, index=False)

    if args.plot > 0:
        dataset = SyntheticDataset(args.destination)

        for i in range(args.plot):
            line = dataset[i]
            df = pd.DataFrame.from_dict({'y': line.squeeze().tolist()})
            df.index.name = "Timestep"

            df.plot(legend=False)
            plt.title(f"Synthetic Sequence Example {i+1}")
            plt.show()


