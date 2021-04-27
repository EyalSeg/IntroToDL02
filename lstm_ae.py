import numpy as np
import torch as T
import torch.nn as nn

DEVICE = T.device('cuda' if T.cuda.is_available() else 'cpu')


def init_weights(lstm:nn.LSTM):
    for name, param in lstm.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            nn.init.xavier_normal_(param)

class LstmAutoEncoder(nn.Module):
    def __init__(self, input_dim, encoded_dim, num_layers=1, device=DEVICE):
        super(LstmAutoEncoder, self).__init__()

        self.input_dim = input_dim
        self.encoded_dim = encoded_dim
        self.num_layers = num_layers
        self.device = None

        self.encoder = nn.LSTM(input_size=input_dim,
                               hidden_size=encoded_dim,
                               num_layers=num_layers,
                               batch_first=True)
        init_weights(self.encoder)

        self.decoder = nn.LSTM(input_size=encoded_dim,
                               hidden_size=input_dim,
                               num_layers=num_layers,
                               batch_first=True)
        init_weights(self.decoder)

        self.encoder_activation = nn.ReLU()
        self.decoder_activation = nn.Sigmoid()

        self.to(device)

    def _encode(self, X):
        h0 = T.randn(self.num_layers, X.size(0), self.encoded_dim)
        c0 = T.randn(self.num_layers, X.size(0), self.encoded_dim)

        h0 = T.autograd.Variable(h0).to(self.device)
        c0 = T.autograd.Variable(c0).to(self.device)

        X, (h_n, c_n) = self.encoder(X, (h0, c0))
        # X, (h_n, c_n) = self.encoder(X)
        # todo: check if an activation method is needed
        h_n = h_n[-1]
        h_n = self.encoder_activation(h_n)

        return h_n

    def _decode(self, encoded_input):
        h0 = T.randn(self.num_layers, encoded_input.size(0), self.input_dim)
        c0 = T.randn(self.num_layers, encoded_input.size(0), self.input_dim)

        h0 = T.autograd.Variable(h0).to(self.device)
        c0 = T.autograd.Variable(c0).to(self.device)

        decoded, hidden_state = self.decoder(encoded_input, (h0, c0))
        # decoded, hidden_state = self.decoder(encoded_input)
        # todo: check if an activation method is needed
        # decoded = self.decoder_activation(decoded)
        return self.decoder_activation(decoded)

    def forward(self, X):
        z = self._encode(X)
        z = z.unsqueeze(1).repeat(1, X.shape[1], 1)
        x_ = self._decode(z)

        return x_

    def to(self, device):
        self.device = device

        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

        return super(LstmAutoEncoder, self).to(device)

