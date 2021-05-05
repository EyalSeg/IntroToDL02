import torch as T
import torch.nn as nn

from dataclasses import dataclass

from lstm_ae import AutoencoderOutput

DEVICE = T.device('cuda' if T.cuda.is_available() else 'cpu')

@dataclass(frozen=True)
class AutoencoderClassifierOutput(AutoencoderOutput):
    label_predictions: T.Tensor


class AutoEncoderClassifier(nn.Module):
    def __init__(self, ae, n_classes):
        super(AutoEncoderClassifier, self).__init__()
        self.ae = ae
        self.output_layer = nn.Linear(self.ae.encoded_dim, n_classes).to(DEVICE)
        self.activation = nn.LogSoftmax(dim=-1).to(DEVICE)

    def forward(self, X):
        temporal_output, context = self.ae.encode(X)

        labels = self.output_layer(context)
        labels = self.activation(labels)

        x_ = self.ae.decode(temporal_output, context)

        return AutoencoderClassifierOutput(x_, labels)