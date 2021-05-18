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

        self.num_layers = num_layers
        self.encoding_dim = encoding_dim

        self.rnn = nn.LSTM(input_size=seq_dim,
                           hidden_size=self.encoding_dim,
                           num_layers=self.num_layers,
                           batch_first=True)
        init_weights(self.rnn)
        self.to(device)

    def forward(self, X, context=None):
        X_, context = self.rnn(X) if context is None else self.rnn(X, context)

        return X_, context


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

    def encode(self, X, context=None):
        X_encoded, (h_n, c_n) = self.encoder(X, context)
        h_n = h_n[-1]

        return X_encoded, h_n

    def encode_stepwise(self, X):
        encoded_X = None
        context = None
        hidden_states = None
        for i in range(X.shape[1]):
            encoded, context = self.encoder.forward(X[:, i, :].unsqueeze(1), context)
            encoded_X = encoded if encoded_X is None else T.cat((encoded_X, encoded), 1)

            state = context[0][-1]
            state = state.unsqueeze(1) # add a temporal dimension
            hidden_states = state if hidden_states is None else T.cat((hidden_states, state), 1)

        return encoded_X, hidden_states

    def decode(self, temporal_output, context):
        return self.decoder(temporal_output)

    def to(self, device):
        self.device = device

        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

        return super(LstmAutoEncoder, self).to(device)

