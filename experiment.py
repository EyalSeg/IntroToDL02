import pandas as pd
import torch as T

import utils


class Experiment:
    def __init__(self, criterion, measurements=None):
        criterion = criterion if isinstance(criterion, dict) else {"loss": criterion}
        self.criterion = criterion
        self.measurements = measurements if measurements else {}

        self.columns = \
            list([f"train_{name}" for name in criterion.keys()]) + \
            list([f"test_{name}" for name in criterion.keys()]) + \
            list([f"train_{name}" for name in self.measurements.keys()]) + \
            list([f"test_{name}" for name in self.measurements.keys()])

        self.__results = None

    def run(self, model, train_dataloader, test_dataloader, hyperparameters, supervised=False, verbose=False):
        self.__results = pd.DataFrame(columns=self.columns)

        def on_epoch_end(epoch, ae, train_losses):
            with T.no_grad():
                test_losses = utils.epoch_losses(ae, test_dataloader, self.criterion, supervised=supervised)

                train_losses = {f"train_{key}": value for key, value in train_losses.items()}
                test_losses = {f"test_{key}": value for key, value in test_losses.items()}

                train_measures =\
                    {f"train_{key}": value(ae, train_dataloader) for key, value in self.measurements.items()}
                test_measures = \
                    {f"train_{key}": value(ae, test_dataloader) for key, value in self.measurements.items()}

                row = {**train_losses, **train_measures, **test_measures, **test_losses}

                if verbose:
                    print(f"Epoch: {epoch}, {row}")

                row['epoch'] = epoch
                self.__results = self.__results.append(row, ignore_index=True)

        utils.fit(model,
                  train_dataloader,
                  self.criterion,
                  hyperparameters,
                  epoch_end_callbacks=[on_epoch_end],
                  supervised=supervised,
                  )

        self.__results['epoch'] = self.__results['epoch'].astype(int)
        self.__results = self.__results.set_index('epoch')

        return self.__results

