import pandas as pd
import torch as T

import utils

DEVICE = T.device('cuda' if T.cuda.is_available() else 'cpu')


class Experiment:
    def __init__(self, criterion, measurements=None, supervised=False):
        criterion = criterion if isinstance(criterion, dict) else {"loss": criterion}
        self.criterion = criterion

        self.supervised=supervised
        self.measurements = measurements if measurements else {}

        self.columns = \
            list([f"train_{name}" for name in criterion.keys()]) + \
            list([f"test_{name}" for name in criterion.keys()]) + \
            list([f"train_{name}" for name in self.measurements.keys()]) + \
            list([f"test_{name}" for name in self.measurements.keys()])

        self.__results = None

    def run(self, model, train_dataloader, test_dataloader, hyperparameters, measure_every=1, verbose=False):
        self.__results = pd.DataFrame(columns=self.columns)

        def on_epoch_end(epoch, ae, train_losses):
            if self.__results.shape[0] % measure_every == 0 or epoch == hyperparameters.epochs - 1:
                row = self.__measure(ae, train_dataloader, test_dataloader, train_losses)
            else:
                row = {f"train_{key}": value for key, value in train_losses.items()}

            if verbose:
                print(f"Epoch: {epoch}, {row}")

            row['epoch'] = epoch
            self.__results = self.__results.append(row, ignore_index=True)

        utils.fit(model,
                  train_dataloader,
                  self.criterion,
                  hyperparameters,
                  epoch_end_callbacks=[on_epoch_end],
                  supervised=self.supervised,
                  )

        self.__results['epoch'] = self.__results['epoch'].astype(int)
        self.__results = self.__results.set_index('epoch')
        self.__results.fillna(method="ffill")

        return self.__results

    def __measure(self, ae, train_dataloader, test_dataloader, train_losses):
        with T.no_grad():
            test_losses = utils.epoch_losses(ae, test_dataloader, self.criterion, supervised=self.supervised)

            train_losses = {f"train_{key}": value for key, value in train_losses.items()}
            test_losses = {f"test_{key}": value for key, value in test_losses.items()}

            train_measures = \
                {f"train_{key}": value(ae, train_dataloader) for key, value in self.measurements.items()}
            test_measures = \
                {f"test_{key}": value(ae, test_dataloader) for key, value in self.measurements.items()}

            row = {**train_losses, **train_measures, **test_measures, **test_losses}
            return row

    @staticmethod
    def measure_accuracy(model, data_loader):
        n_correct = 0
        total = 0
        with T.no_grad():
            for batch in iter(data_loader):
                X, y = batch[0].to(DEVICE), batch[1].to(DEVICE)

                output = model.forward(X)
                predictions = T.argmax(output.label_predictions, -1)

                n_correct += predictions.eq(y).sum().item()
                total += len(batch[0])

        return n_correct / total
