import torch as T
import torch.nn as nn
from torchvision import datasets
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, SubsetRandomSampler

import utils

from grid_search import tune
from utils import LstmAEHyperparameters

sns.set_theme(style="darkgrid")

DEVICE = T.device('cuda' if T.cuda.is_available() else 'cpu')
T.set_default_dtype(T.double)


if __name__ == "__main__":
    train_data, validate_data, test_data = utils.load_torch_dataset(datasets.MNIST, cache_path="../data/cache")

    pass
