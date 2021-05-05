import torch as T
import torch.nn as nn
from dataclasses import dataclass

DEVICE = T.device('cuda' if T.cuda.is_available() else 'cpu')


def init_weights(lstm):
    for name, param in lstm.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            nn.init.xavier_normal_(param)


@dataclass(frozen=True)
class AutoencoderOutput:
    output_sequence: T.Tensor


class Encoder(nn.Module):
    def __init__(self, seq_dim, encoding_dim, num_layers=1, device=DEVICE):
        super(Encoder, self).__init__()

        self.rnn = nn.LSTM(input_size=seq_dim,
                           hidden_size=encoding_dim,
                           num_layers=num_layers,
                           batch_first=True)
        init_weights(self.rnn)
        self.to(device)

    def forward(self, X):
        X_, (h_n, c_n) = self.rnn(X)

        return X_, h_n[-1]


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers=1, device=DEVICE):
        super(Decoder, self).__init__()

        self.rnn = nn.LSTM(input_size=input_dim,
                           hidden_size=hidden_dim,
                           num_layers=num_layers,
                           batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        init_weights(self.rnn)
        init_weights(self.output_layer)
        self.to(device)

    def forward(self, encoded_X):
        decoded_X, hidden_state = self.rnn(encoded_X)

        return self.output_layer(decoded_X)


class LstmAutoEncoder(nn.Module):
    def __init__(self, input_dim, encoded_dim, num_layers=1, device=DEVICE):
        super(LstmAutoEncoder, self).__init__()
        self.encoded_dim = encoded_dim

        self.encoder = Encoder(input_dim, self.encoded_dim, num_layers, device)
        self.decoder = Decoder(self.encoded_dim, input_dim, self.encoded_dim, num_layers, device)

    def forward(self, X):
        temporal_output, context = self.encode(X)
        x_ = self.decode(temporal_output, context)

        return AutoencoderOutput(x_)

    def encode(self, X):
        return self.encoder(X)

    def decode(self, temporal_output, context):
        # context = context.unsqueeze(1).repeat((1, temporal_output.shape[1], 1))
        # return self.decoder(context)
        return self.decoder(temporal_output)

    def to(self, device):
        self.device = device

        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

        return super(LstmAutoEncoder, self).to(device)

