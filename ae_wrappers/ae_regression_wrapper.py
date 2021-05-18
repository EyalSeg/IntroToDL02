import torch as T
import torch.nn as nn

from dataclasses import dataclass

from lstm_ae import AutoencoderOutput

DEVICE = T.device('cuda' if T.cuda.is_available() else 'cpu')


@dataclass(frozen=True)
class AutoencoderRegressorOutput(AutoencoderOutput):
    predicted_value: T.Tensor


class AutoEncoderRegressor(nn.Module):
    def __init__(self, ae):
        super(AutoEncoderRegressor, self).__init__()
        self.ae = ae
        self.output_layer = nn.Linear(self.ae.encoded_dim, 1).to(DEVICE)

        T.nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, X):
        encoded_X, context = self.ae.encode(X)
        predicted_vals = self.output_layer(encoded_X)

        x_ = self.ae.decode(encoded_X, context)
        return AutoencoderRegressorOutput(x_, predicted_vals)

