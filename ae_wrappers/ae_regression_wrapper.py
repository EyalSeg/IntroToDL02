import torch as T
import torch.nn as nn

from dataclasses import dataclass

from lstm_ae import AutoencoderOutput

DEVICE = T.device('cuda' if T.cuda.is_available() else 'cpu')


@dataclass(frozen=True)
class AutoencoderRegressionOutput(AutoencoderOutput):
    label_predictions: T.Tensor


class AutoEncoderRegression(nn.Module):
    def __init__(self, ae, output_dimension):
        super(AutoEncoderRegression, self).__init__()
        self.ae = ae
        self.output_layer = nn.Linear(self.ae.encoded_dim, output_dimension).to(DEVICE)
        # self.activation = nn.LogSoftmax(dim=-1).to(DEVICE)

        T.nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, X):
        temporal_output, context = self.ae.encode(X)

        labels = self.output_layer(context)
        # labels = self.activation(labels)

        x_ = self.ae.decode(temporal_output, context)

        return AutoencoderRegressionOutput(x_, labels)
