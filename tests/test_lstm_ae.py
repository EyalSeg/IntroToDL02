import pytest

import numpy as np
import torch as T

from torch import nn, optim
from sklearn.linear_model import LinearRegression

from lstm_ae import LstmAutoEncoder


device = 'cuda:0' if T.cuda.is_available() else 'cpu'


class Test_Lstm_AE:

    @pytest.fixture()
    def ae(self, seq_dim, latent_size, num_layers):
        return LstmAutoEncoder(seq_dim, latent_size, num_layers).to(device)

    @pytest.mark.parametrize("seq_length", [50])
    @pytest.mark.parametrize("latent_size", [20])
    @pytest.mark.parametrize("batch_size", [1, 100])
    @pytest.mark.parametrize("seq_dim", [1, 5])
    @pytest.mark.parametrize("num_layers", [1, 2])
    def test_output_shape(self, seq_length, latent_size, batch_size, seq_dim, num_layers, ae):
        input = T.randn(batch_size, seq_length, seq_dim).to(device)

        with T.no_grad():
            output = ae.forward(input).output_sequence

        assert input.shape == output.shape, "Output is not at the same shape of the input!"

    @pytest.mark.parametrize("seq_length", [50])
    @pytest.mark.parametrize("latent_size", [20])
    @pytest.mark.parametrize("batch_size", [1, 100])
    @pytest.mark.parametrize("seq_dim", [1, 5])
    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.parametrize("input", [
        lambda batch, length, dim: T.randn(batch, length, dim, dtype=T.float32).to(device),
    ])
    def test_fit(self, seq_length, latent_size, batch_size, seq_dim, num_layers, input, ae):
        # x = np.linspace(-np.pi, np.pi, seq_length)
        # input = T.tensor([[np.sin(x)]], dtype=T.float32).to(device)

        input = input(batch_size, seq_length, seq_dim)

        criterion = nn.MSELoss()
        optimizer = optim.SGD(ae.parameters(), lr=0.2)

        losses = []
        for epoch in range(500):
            optimizer.zero_grad()

            output = ae.forward(input).output_sequence
            loss = criterion(output, input)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        x = [i for i in range(len(losses))]
        coeffs = np.polyfit(x, losses, 1)
        slope = coeffs[0]

        assert slope < -0, "Learning does not decrease the training loss!"
