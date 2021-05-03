import torch as T
import torch.nn as nn


class AutoEncoderClassifier(nn.Module):
    def __init__(self, ae, n_classes):
        super(AutoEncoderClassifier, self).__init__()
        self.ae = ae
        self.output_layer = nn.Linear(self.ae.encoded_dim, n_classes).to(ae.device)
        self.activation = nn.Softmax().to(ae.device)

    def forward(self, X):
        temporal_output, context = self.ae.encoder(X)

        labels = self.output_layer(context)
        labels = self.activation(labels)

        x_ = self.ae.decoder(temporal_output)

        return x_, labels